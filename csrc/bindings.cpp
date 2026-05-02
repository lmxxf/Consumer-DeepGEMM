#include <torch/extension.h>

#include <pybind11/pybind11.h>

#include <string>

bool cutlass_sm120_probe_compiled();
std::string cutlass_sm120_probe_arch();

namespace py = pybind11;

namespace {

torch::Tensor tensor_from_object(const py::object& value, const char* name) {
  try {
    return value.cast<torch::Tensor>();
  } catch (const py::cast_error&) {
    throw std::invalid_argument(std::string(name) + " must be a torch.Tensor");
  }
}

std::pair<torch::Tensor, torch::Tensor> tensor_scale_pair_from_object(
    const py::object& value,
    const char* name) {
  if (!py::isinstance<py::tuple>(value)) {
    throw std::invalid_argument(std::string(name) + " must be a (tensor, scale) tuple");
  }
  auto tuple = value.cast<py::tuple>();
  if (tuple.size() != 2) {
    throw std::invalid_argument(std::string(name) + " must have exactly two items");
  }
  auto tensor = tensor_from_object(tuple[0].cast<py::object>(), name);
  auto scale_name = std::string(name) + " scale";
  auto scale = tensor_from_object(tuple[1].cast<py::object>(), scale_name.c_str());
  return {tensor, scale};
}

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_grouped_fp8_fp4_shapes(
    const torch::Tensor& a,
    const torch::Tensor& a_scale,
    const torch::Tensor& b,
    const torch::Tensor& b_scale,
    const torch::Tensor& d,
    const py::object& m_indices) {
  check_cuda_tensor(a, "a");
  check_cuda_tensor(a_scale, "a_scale");
  check_cuda_tensor(b, "b");
  check_cuda_tensor(b_scale, "b_scale");
  check_cuda_tensor(d, "d");

  TORCH_CHECK(a.dim() == 2, "a must be [M, K], got dim ", a.dim());
  TORCH_CHECK(b.dim() == 3, "b must be [G, N, K/2] packed FP4, got dim ", b.dim());
  TORCH_CHECK(d.dim() == 2, "d must be [M, N], got dim ", d.dim());
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn ||
                  a.scalar_type() == torch::kBFloat16 ||
                  a.scalar_type() == torch::kFloat32,
              "a must be FP8/BF16/FP32, got ", a.scalar_type());
  TORCH_CHECK(b.scalar_type() == torch::kInt8 || b.scalar_type() == torch::kUInt8,
              "b must be int8/uint8 packed FP4, got ", b.scalar_type());
  TORCH_CHECK(d.scalar_type() == torch::kBFloat16 || d.scalar_type() == torch::kFloat32,
              "d must be BF16/FP32, got ", d.scalar_type());

  const auto m = a.size(0);
  const auto k = a.size(1);
  const auto groups = b.size(0);
  const auto n = b.size(1);
  TORCH_CHECK(b.size(2) * 2 == k,
              "b packed K must equal a K/2, got b.size(2)=",
              b.size(2), " and a K=", k);
  TORCH_CHECK(d.size(0) == m, "d M must match a M");
  TORCH_CHECK(d.size(1) == n, "d N must match b N");

  if (!m_indices.is_none()) {
    auto indices = tensor_from_object(m_indices, "m_indices");
    check_cuda_tensor(indices, "m_indices");
    TORCH_CHECK(indices.dim() == 1, "m_indices must be 1D");
    TORCH_CHECK(indices.scalar_type() == torch::kInt32 || indices.scalar_type() == torch::kInt64,
                "m_indices must be int32/int64, got ", indices.scalar_type());
    TORCH_CHECK(indices.numel() == m || indices.numel() == groups,
                "m_indices length must be M or G, got ", indices.numel(),
                ", M=", m, ", G=", groups);
  }
}

py::object m_grouped_fp8_fp4_gemm_nt_contiguous_stub(
    py::object a_obj,
    py::object b_obj,
    torch::Tensor d,
    py::object m_indices,
    py::kwargs) {
  auto [a, a_scale] = tensor_scale_pair_from_object(a_obj, "a");
  auto [b, b_scale] = tensor_scale_pair_from_object(b_obj, "b");
  check_grouped_fp8_fp4_shapes(a, a_scale, b, b_scale, d, m_indices);

  // ABI is wired. Returning None lets Python use the correctness fallback until
  // the CUTLASS 79d SM120 grouped FP4 kernel is connected here.
  return py::none();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_available", []() { return true; });
  m.def("cutlass_sm120_probe_compiled", &cutlass_sm120_probe_compiled);
  m.def("cutlass_sm120_probe_arch", &cutlass_sm120_probe_arch);
  m.def(
      "m_grouped_fp8_fp4_gemm_nt_contiguous",
      &m_grouped_fp8_fp4_gemm_nt_contiguous_stub,
      py::arg("a"),
      py::arg("b"),
      py::arg("d"),
      py::arg("m_indices") = py::none());
}
