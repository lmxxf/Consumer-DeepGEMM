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

struct SegmentResult {
  std::vector<GroupSegment> segments;
  std::vector<int> sorted_rows;
  bool needs_scatter;
};

SegmentResult segments_from_indices(torch::Tensor m_indices, int m, int groups) {
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

  if (static_cast<int>(vals.size()) == groups) {
    std::vector<GroupSegment> out;
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
    return {std::move(out), {}, false};
  }

  if (static_cast<int>(vals.size()) != m) {
    return {};
  }

  std::vector<std::vector<int>> rows_per_group(groups);
  for (int row = 0; row < m; ++row) {
    int gid = static_cast<int>(vals[row]);
    if (gid == -1) continue;
    if (gid < 0 || gid >= groups) return {};
    rows_per_group[gid].push_back(row);
  }

  std::vector<int> sorted_rows;
  sorted_rows.reserve(m);
  std::vector<GroupSegment> out;
  int offset = 0;
  for (int group = 0; group < groups; ++group) {
    auto& rows = rows_per_group[group];
    if (rows.empty()) continue;
    out.push_back({group, offset, static_cast<int>(rows.size())});
    sorted_rows.insert(sorted_rows.end(), rows.begin(), rows.end());
    offset += static_cast<int>(rows.size());
  }

  bool already_packed = true;
  for (int i = 0; i < static_cast<int>(sorted_rows.size()); ++i) {
    if (sorted_rows[i] != i) { already_packed = false; break; }
  }

  if (already_packed && static_cast<int>(sorted_rows.size()) == m) {
    return {std::move(out), {}, false};
  }

  return {std::move(out), std::move(sorted_rows), true};
}

}  // close anonymous namespace for kernel definition

__global__ void reorder_scale_kernel(const uint8_t* __restrict__ src,
                                     uint8_t* __restrict__ dst,
                                     int rows, int k_blocks,
                                     int k_tiles, int atom_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_src = rows * k_blocks;
  if (idx >= total_src) return;

  int r = idx / k_blocks;
  int c = idx % k_blocks;
  int mt = r / 128;
  int m_in_tile = r % 128;
  int m_32 = m_in_tile / 32;
  int m_in_32 = m_in_tile % 32;
  int kt = c / 4;
  int k_in_4 = c % 4;
  int tile_offset = (mt * k_tiles + kt) * atom_size;
  int in_tile = m_in_32 * 16 + m_32 * 4 + k_in_4;
  dst[tile_offset + in_tile] = src[idx];
}

torch::Tensor reorder_scale_on_gpu(torch::Tensor src_gpu, int rows, int k_blocks) {
  const int m_tiles = (rows + 127) / 128;
  const int k_tiles = (k_blocks + 3) / 4;
  const int atom_size = 32 * 4 * 4;
  const int total_out = m_tiles * k_tiles * atom_size;

  auto result = torch::full({total_out}, 127,
                            torch::TensorOptions().dtype(torch::kUInt8).device(src_gpu.device()));
  int total_src = rows * k_blocks;
  int threads = 256;
  int blocks = (total_src + threads - 1) / threads;
  auto stream = c10::cuda::getCurrentCUDAStream(src_gpu.get_device()).stream();
  reorder_scale_kernel<<<blocks, threads, 0, stream>>>(
      src_gpu.data_ptr<uint8_t>(), result.data_ptr<uint8_t>(),
      rows, k_blocks, k_tiles, atom_size);
  return result;
}

namespace {  // re-open anonymous namespace

torch::Tensor reorder_scale_for_cutlass(const uint8_t* src, int rows, int k_blocks,
                                        torch::Device device) {
  const int m_tiles = (rows + 127) / 128;
  const int k_tiles = (k_blocks + 3) / 4;
  const int atom_size = 32 * 4 * 4;
  const int total = m_tiles * k_tiles * atom_size;

  auto result = torch::full({total}, 127, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  auto* dst = result.data_ptr<uint8_t>();

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < k_blocks; ++c) {
      int mt = r / 128;
      int m_in_tile = r % 128;
      int m_32 = m_in_tile / 32;
      int m_in_32 = m_in_tile % 32;
      int kt = c / 4;
      int k_in_4 = c % 4;
      int tile_offset = (mt * k_tiles + kt) * atom_size;
      int in_tile = m_in_32 * 16 + m_32 * 4 + k_in_4;
      dst[tile_offset + in_tile] = src[r * k_blocks + c];
    }
  }

  return result.to(device, /*non_blocking=*/false);
}

torch::Tensor gather_rows(torch::Tensor src, std::vector<int> const& row_indices,
                          torch::Device device) {
  auto idx = torch::from_blob(const_cast<int*>(row_indices.data()),
                              {static_cast<int64_t>(row_indices.size())},
                              torch::TensorOptions().dtype(torch::kInt32))
                 .to(device, torch::kLong, /*non_blocking=*/false);
  return src.index_select(0, idx).contiguous();
}

void scatter_rows(torch::Tensor dst, torch::Tensor packed_src,
                  std::vector<int> const& row_indices, torch::Device device) {
  auto idx = torch::from_blob(const_cast<int*>(row_indices.data()),
                              {static_cast<int64_t>(row_indices.size())},
                              torch::TensorOptions().dtype(torch::kInt32))
                 .to(device, torch::kLong, /*non_blocking=*/false);
  dst.index_copy_(0, idx, packed_src);
}

bool launch_grouped_fp8_fp4(torch::Tensor a, torch::Tensor a_scale, torch::Tensor b,
                            torch::Tensor b_scale, torch::Tensor d, torch::Tensor m_indices) {
  const int groups = static_cast<int>(b.size(0));
  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));
  auto seg_result = segments_from_indices(m_indices, m, groups);
  if (seg_result.segments.empty()) {
    return false;
  }

  auto device = a.device();
  auto const& segments = seg_result.segments;

  torch::Tensor a_work = a;
  torch::Tensor a_scale_work = a_scale;
  torch::Tensor d_work;

  if (seg_result.needs_scatter) {
    a_work = gather_rows(a, seg_result.sorted_rows, device);
    a_scale_work = gather_rows(a_scale, seg_result.sorted_rows, device);
    int total_active = static_cast<int>(seg_result.sorted_rows.size());
    d_work = torch::empty({total_active, n},
                          torch::TensorOptions().device(device).dtype(d.scalar_type()));
  } else {
    d_work = d;
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

  const int64_t a_row_stride = a_work.stride(0);
  const int64_t d_row_stride = d_work.stride(0);
  const int64_t b_group_stride = b.stride(0);

  auto* a_base = reinterpret_cast<uint8_t*>(a_work.data_ptr());
  auto* b_base = reinterpret_cast<uint8_t*>(b.data_ptr());
  auto* d_base = reinterpret_cast<uint8_t*>(d_work.data_ptr());

  const int a_scale_cols = a_scale_work.dim() >= 2 ? static_cast<int>(a_scale_work.size(1)) : static_cast<int>(a_scale_work.numel()) / m;
  const int b_scale_cols = b_scale.dim() >= 2 ? static_cast<int>(b_scale.size(-1)) : 1;

  std::vector<torch::Tensor> sfa_buffers;
  std::vector<torch::Tensor> sfb_buffers;

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

    ptr_a.push_back(reinterpret_cast<uintptr_t>(a_base + seg.start * a_row_stride * a_work.element_size()));
    ptr_b.push_back(reinterpret_cast<uintptr_t>(b_base + seg.group * b_group_stride * b.element_size()));
    ptr_c.push_back(0);
    ptr_d.push_back(reinterpret_cast<uintptr_t>(d_base + seg.start * d_row_stride * d_work.element_size()));

    {
      auto sfa_slice = a_scale_work.is_cuda()
          ? a_scale_work.narrow(0, seg.start, seg.count).contiguous().view(-1).to(torch::kUInt8)
          : a_scale_work.narrow(0, seg.start, seg.count).contiguous().view(-1).to(torch::kUInt8).to(device);
      auto sfa_buf = reorder_scale_on_gpu(sfa_slice, seg.count, a_scale_cols);
      sfa_buffers.push_back(sfa_buf);
      ptr_sfa.push_back(reinterpret_cast<uintptr_t>(sfa_buf.data_ptr()));
    }

    {
      torch::Tensor sfb_flat;
      if (b_scale.dim() == 3) {
        sfb_flat = b_scale.select(0, seg.group).contiguous().view(-1).to(torch::kUInt8);
      } else {
        sfb_flat = b_scale.contiguous().view(-1).to(torch::kUInt8);
      }
      if (!sfb_flat.is_cuda()) {
        sfb_flat = sfb_flat.to(device);
      }
      auto sfb_buf = reorder_scale_on_gpu(sfb_flat, n, b_scale_cols);
      sfb_buffers.push_back(sfb_buf);
      ptr_sfb.push_back(reinterpret_cast<uintptr_t>(sfb_buf.data_ptr()));
    }
  }

  const int active = static_cast<int>(segments.size());
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

  if (!seg_result.needs_scatter) {
    d.zero_();
  }
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
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  if (seg_result.needs_scatter) {
    d.zero_();
    scatter_rows(d, d_work, seg_result.sorted_rows, device);
  }
  return true;
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
