#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"
#include "c10/cuda/CUDAStream.h"
#include "torch/extension.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;

using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128;

using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using GroupProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

using GroupedCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag*, AlignmentC,
    ElementD, LayoutDTag*, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using GroupedCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag*, AlignmentA,
    ElementB, LayoutBTag*, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename GroupedCollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GroupedGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    GroupProblemShape,
    GroupedCollectiveMainloop,
    GroupedCollectiveEpilogue>;

using GroupedGemm = cutlass::gemm::device::GemmUniversalAdapter<GroupedGemmKernel>;
using GroupedStrideA = typename GroupedGemm::GemmKernel::InternalStrideA;
using GroupedStrideB = typename GroupedGemm::GemmKernel::InternalStrideB;
using GroupedStrideC = typename GroupedGemm::GemmKernel::InternalStrideC;
using GroupedStrideD = typename GroupedGemm::GemmKernel::InternalStrideD;
using GroupedLayoutSFA = typename GroupedGemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using GroupedLayoutSFB = typename GroupedGemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;

typename GroupedGemm::Arguments make_grouped_arguments_probe() {
  typename GroupedGemm::ElementA const** ptr_a = nullptr;
  typename GroupedGemm::ElementB const** ptr_b = nullptr;
  typename GroupedGemm::ElementC const** ptr_c = nullptr;
  typename GroupedGemm::EpilogueOutputOp::ElementOutput** ptr_d = nullptr;
  typename GroupedGemm::GemmKernel::CollectiveMainloop::ElementSF const** ptr_sfa = nullptr;
  typename GroupedGemm::GemmKernel::CollectiveMainloop::ElementSF const** ptr_sfb = nullptr;

  GroupProblemShape::UnderlyingProblemShape* problem_sizes = nullptr;
  GroupedStrideA* stride_a = nullptr;
  GroupedStrideB* stride_b = nullptr;
  GroupedStrideC* stride_c = nullptr;
  GroupedStrideD* stride_d = nullptr;
  GroupedLayoutSFA* layout_sfa = nullptr;
  GroupedLayoutSFB* layout_sfb = nullptr;

  decltype(std::declval<typename GroupedGemm::Arguments>().epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.dBeta = {_0{}, _0{}, 0};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = 1;

  typename GroupedGemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;

  return typename GroupedGemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {0, problem_sizes, nullptr},
      {ptr_a, stride_a, ptr_b, stride_b, ptr_sfa, layout_sfa, ptr_sfb, layout_sfb},
      {fusion_args, ptr_c, stride_c, ptr_d, stride_d},
      hw_info,
      scheduler};
}

bool can_implement_grouped_probe(torch::Tensor a, torch::Tensor b, torch::Tensor d) {
  const int groups = static_cast<int>(b.size(0));
  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));

  std::vector<GroupProblemShape::UnderlyingProblemShape> problem_sizes;
  std::vector<GroupedStrideA> stride_a;
  std::vector<GroupedStrideB> stride_b;
  std::vector<GroupedStrideC> stride_c;
  std::vector<GroupedStrideD> stride_d;
  std::vector<GroupedLayoutSFA> layout_sfa;
  std::vector<GroupedLayoutSFB> layout_sfb;

  problem_sizes.reserve(groups);
  stride_a.reserve(groups);
  stride_b.reserve(groups);
  stride_c.reserve(groups);
  stride_d.reserve(groups);
  layout_sfa.reserve(groups);
  layout_sfb.reserve(groups);

  for (int i = 0; i < groups; ++i) {
    problem_sizes.push_back({m, n, k});
    stride_a.push_back(cutlass::make_cute_packed_stride(GroupedStrideA{}, {m, k, 1}));
    stride_b.push_back(cutlass::make_cute_packed_stride(GroupedStrideB{}, {n, k, 1}));
    stride_c.push_back(cutlass::make_cute_packed_stride(GroupedStrideC{}, {m, n, 1}));
    stride_d.push_back(cutlass::make_cute_packed_stride(GroupedStrideD{}, {m, n, 1}));
    layout_sfa.push_back(
        GroupedGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
            tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1)));
    layout_sfb.push_back(
        GroupedGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
            tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1)));
  }

  typename GroupedGemm::ElementA const** ptr_a = nullptr;
  typename GroupedGemm::ElementB const** ptr_b = nullptr;
  typename GroupedGemm::ElementC const** ptr_c = nullptr;
  typename GroupedGemm::EpilogueOutputOp::ElementOutput** ptr_d = nullptr;
  typename GroupedGemm::GemmKernel::CollectiveMainloop::ElementSF const** ptr_sfa = nullptr;
  typename GroupedGemm::GemmKernel::CollectiveMainloop::ElementSF const** ptr_sfb = nullptr;

  decltype(std::declval<typename GroupedGemm::Arguments>().epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.dBeta = {_0{}, _0{}, 0};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a.get_device();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename GroupedGemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;

  typename GroupedGemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {groups, problem_sizes.data(), problem_sizes.data()},
      {ptr_a, stride_a.data(), ptr_b, stride_b.data(),
       ptr_sfa, layout_sfa.data(), ptr_sfb, layout_sfb.data()},
      {fusion_args, ptr_c, stride_c.data(), ptr_d, stride_d.data()},
      hw_info,
      scheduler};

  GroupedGemm gemm;
  return gemm.can_implement(arguments) == cutlass::Status::kSuccess;
}

template <typename T>
torch::Tensor device_copy_from_host(std::vector<T> const& values, torch::Device device) {
  auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8);
  auto host = torch::empty({static_cast<int64_t>(values.size() * sizeof(T))}, options);
  std::memcpy(host.data_ptr(), values.data(), values.size() * sizeof(T));
  return host.to(device, /*non_blocking=*/false);
}

torch::Tensor pointer_array_from_host(std::vector<uintptr_t> const& values, torch::Device device) {
  auto host = torch::empty({static_cast<int64_t>(values.size())}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));
  auto* out = host.data_ptr<int64_t>();
  for (size_t i = 0; i < values.size(); ++i) {
    out[i] = static_cast<int64_t>(values[i]);
  }
  return host.to(device, /*non_blocking=*/false);
}

struct GroupSegment {
  int group;
  int start;
  int count;
};

std::vector<GroupSegment> segments_from_indices(torch::Tensor m_indices, int m, int groups) {
  auto cpu = m_indices.to(torch::kCPU, /*non_blocking=*/false).contiguous();
  std::vector<int64_t> vals(cpu.numel());
  if (cpu.scalar_type() == torch::kInt32) {
    auto* p = cpu.data_ptr<int32_t>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
      vals[i] = p[i];
    }
  } else if (cpu.scalar_type() == torch::kInt64) {
    auto* p = cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
      vals[i] = p[i];
    }
  } else {
    return {};
  }

  std::vector<GroupSegment> out;
  if (static_cast<int>(vals.size()) == groups) {
    int start = 0;
    for (int group = 0; group < groups; ++group) {
      int end = static_cast<int>(vals[group]);
      if (end < start || end > m) {
        return {};
      }
      if (end > start) {
        out.push_back({group, start, end - start});
      }
      start = end;
    }
    return out;
  }

  if (static_cast<int>(vals.size()) != m) {
    return {};
  }

  for (int group = 0; group < groups; ++group) {
    int start = -1;
    int count = 0;
    bool closed = false;
    for (int row = 0; row < m; ++row) {
      if (vals[row] == group) {
        if (closed) {
          return {};
        }
        if (start < 0) {
          start = row;
        }
        ++count;
      } else if (start >= 0) {
        closed = true;
      } else if (vals[row] < -1 || vals[row] >= groups) {
        return {};
      }
    }
    if (count > 0) {
      out.push_back({group, start, count});
    }
  }
  return out;
}

bool launch_grouped_fp8_fp4(torch::Tensor a, torch::Tensor a_scale, torch::Tensor b,
                            torch::Tensor b_scale, torch::Tensor d, torch::Tensor m_indices) {
  const int groups = static_cast<int>(b.size(0));
  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));
  auto segments = segments_from_indices(m_indices, m, groups);
  if (segments.empty()) {
    return false;
  }

  std::vector<GroupProblemShape::UnderlyingProblemShape> problem_sizes;
  std::vector<GroupedStrideA> stride_a;
  std::vector<GroupedStrideB> stride_b;
  std::vector<GroupedStrideC> stride_c;
  std::vector<GroupedStrideD> stride_d;
  std::vector<GroupedLayoutSFA> layout_sfa;
  std::vector<GroupedLayoutSFB> layout_sfb;
  std::vector<uintptr_t> ptr_a;
  std::vector<uintptr_t> ptr_b;
  std::vector<uintptr_t> ptr_c;
  std::vector<uintptr_t> ptr_d;
  std::vector<uintptr_t> ptr_sfa;
  std::vector<uintptr_t> ptr_sfb;

  const int64_t a_row_stride = a.stride(0);
  const int64_t d_row_stride = d.stride(0);
  const int64_t b_group_stride = b.stride(0);
  const int64_t b_scale_group_stride = b_scale.dim() > 0 ? b_scale.stride(0) : 0;
  const int64_t a_scale_row_stride = a_scale.dim() > 0 ? a_scale.stride(0) : 0;

  auto* a_base = reinterpret_cast<uint8_t*>(a.data_ptr());
  auto* b_base = reinterpret_cast<uint8_t*>(b.data_ptr());
  auto* d_base = reinterpret_cast<uint8_t*>(d.data_ptr());
  auto* a_scale_base = reinterpret_cast<uint8_t*>(a_scale.data_ptr());
  auto* b_scale_base = reinterpret_cast<uint8_t*>(b_scale.data_ptr());

  for (auto const& seg : segments) {
    problem_sizes.push_back({seg.count, n, k});
    stride_a.push_back(cutlass::make_cute_packed_stride(GroupedStrideA{}, {seg.count, k, 1}));
    stride_b.push_back(cutlass::make_cute_packed_stride(GroupedStrideB{}, {n, k, 1}));
    stride_c.push_back(cutlass::make_cute_packed_stride(GroupedStrideC{}, {seg.count, n, 1}));
    stride_d.push_back(cutlass::make_cute_packed_stride(GroupedStrideD{}, {seg.count, n, 1}));
    layout_sfa.push_back(
        GroupedGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
            tile_atom_to_shape_SFA(cute::make_shape(seg.count, n, k, 1)));
    layout_sfb.push_back(
        GroupedGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
            tile_atom_to_shape_SFB(cute::make_shape(seg.count, n, k, 1)));

    ptr_a.push_back(reinterpret_cast<uintptr_t>(a_base + seg.start * a_row_stride * a.element_size()));
    ptr_b.push_back(reinterpret_cast<uintptr_t>(b_base + seg.group * b_group_stride * b.element_size()));
    ptr_c.push_back(0);
    ptr_d.push_back(reinterpret_cast<uintptr_t>(d_base + seg.start * d_row_stride * d.element_size()));
    ptr_sfa.push_back(reinterpret_cast<uintptr_t>(a_scale_base + seg.start * a_scale_row_stride * a_scale.element_size()));
    ptr_sfb.push_back(reinterpret_cast<uintptr_t>(b_scale_base + seg.group * b_scale_group_stride * b_scale.element_size()));
  }

  const int active = static_cast<int>(segments.size());
  auto device = a.device();
  auto problem_sizes_dev = device_copy_from_host(problem_sizes, device);
  auto stride_a_dev = device_copy_from_host(stride_a, device);
  auto stride_b_dev = device_copy_from_host(stride_b, device);
  auto stride_c_dev = device_copy_from_host(stride_c, device);
  auto stride_d_dev = device_copy_from_host(stride_d, device);
  auto layout_sfa_dev = device_copy_from_host(layout_sfa, device);
  auto layout_sfb_dev = device_copy_from_host(layout_sfb, device);
  auto ptr_a_dev = pointer_array_from_host(ptr_a, device);
  auto ptr_b_dev = pointer_array_from_host(ptr_b, device);
  auto ptr_c_dev = pointer_array_from_host(ptr_c, device);
  auto ptr_d_dev = pointer_array_from_host(ptr_d, device);
  auto ptr_sfa_dev = pointer_array_from_host(ptr_sfa, device);
  auto ptr_sfb_dev = pointer_array_from_host(ptr_sfb, device);

  decltype(std::declval<typename GroupedGemm::Arguments>().epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.dBeta = {_0{}, _0{}, 0};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a.get_device();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename GroupedGemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;

  typename GroupedGemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {active,
       reinterpret_cast<GroupProblemShape::UnderlyingProblemShape*>(problem_sizes_dev.data_ptr()),
       problem_sizes.data()},
      {reinterpret_cast<typename GroupedGemm::ElementA const**>(ptr_a_dev.data_ptr<int64_t>()),
       reinterpret_cast<GroupedStrideA*>(stride_a_dev.data_ptr()),
       reinterpret_cast<typename GroupedGemm::ElementB const**>(ptr_b_dev.data_ptr<int64_t>()),
       reinterpret_cast<GroupedStrideB*>(stride_b_dev.data_ptr()),
       reinterpret_cast<typename GroupedGemm::GemmKernel::CollectiveMainloop::ElementSF const**>(ptr_sfa_dev.data_ptr<int64_t>()),
       reinterpret_cast<GroupedLayoutSFA*>(layout_sfa_dev.data_ptr()),
       reinterpret_cast<typename GroupedGemm::GemmKernel::CollectiveMainloop::ElementSF const**>(ptr_sfb_dev.data_ptr<int64_t>()),
       reinterpret_cast<GroupedLayoutSFB*>(layout_sfb_dev.data_ptr())},
      {fusion_args,
       reinterpret_cast<typename GroupedGemm::ElementC const**>(ptr_c_dev.data_ptr<int64_t>()),
       reinterpret_cast<GroupedStrideC*>(stride_c_dev.data_ptr()),
       reinterpret_cast<typename GroupedGemm::EpilogueOutputOp::ElementOutput**>(ptr_d_dev.data_ptr<int64_t>()),
       reinterpret_cast<GroupedStrideD*>(stride_d_dev.data_ptr())},
      hw_info,
      scheduler};

  GroupedGemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    return false;
  }

  d.zero_();
  const size_t workspace_size = GroupedGemm::get_workspace_size(arguments);
  auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                torch::TensorOptions().device(device).dtype(torch::kUInt8));
  void* workspace_ptr = workspace_size == 0 ? nullptr : workspace.data_ptr();
  auto stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  auto status = gemm.initialize(arguments, workspace_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return false;
  }
  status = gemm.run(stream);
  return status == cutlass::Status::kSuccess;
}

#endif

}  // namespace

bool cutlass_mxfp8_mxfp4_probe_compiled() {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  (void)sizeof(Gemm);
  (void)sizeof(GroupedGemm);
  (void)sizeof(make_grouped_arguments_probe());
  return true;
#else
  return false;
#endif
}

bool cutlass_mxfp8_mxfp4_can_implement_probe(torch::Tensor a, torch::Tensor b, torch::Tensor d) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  return can_implement_grouped_probe(a, b, d);
#else
  return false;
#endif
}

std::vector<int64_t> cutlass_mxfp8_mxfp4_scale_layout_sizes(int64_t m, int64_t n, int64_t k) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  auto layout_sfa =
      GroupedGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
          tile_atom_to_shape_SFA(cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  auto layout_sfb =
      GroupedGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
          tile_atom_to_shape_SFB(cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  return {
      static_cast<int64_t>(cute::size(cute::filter_zeros(layout_sfa))),
      static_cast<int64_t>(cute::size(cute::filter_zeros(layout_sfb))),
  };
#else
  return {0, 0};
#endif
}

bool cutlass_mxfp8_mxfp4_grouped_launch(
    torch::Tensor a,
    torch::Tensor a_scale,
    torch::Tensor b,
    torch::Tensor b_scale,
    torch::Tensor d,
    torch::Tensor m_indices) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  return launch_grouped_fp8_fp4(a, a_scale, b, b_scale, d, m_indices);
#else
  return false;
#endif
}
