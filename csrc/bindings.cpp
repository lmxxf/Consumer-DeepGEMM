#include <torch/extension.h>

#include <string>

bool cutlass_sm120_probe_compiled();
std::string cutlass_sm120_probe_arch();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_available", []() { return true; });
  m.def("cutlass_sm120_probe_compiled", &cutlass_sm120_probe_compiled);
  m.def("cutlass_sm120_probe_arch", &cutlass_sm120_probe_arch);
}
