#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"
#include "torch/extension.h"

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
