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
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

// ============================================================
// CUTLASS type aliases (unchanged)
// ============================================================

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
// Keep 128x128 — 64x128 needs CUTLASS v4.4.2+ (version mismatch with DeepGEMM's bundled CUTLASS)
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

// ============================================================
// Probe helpers (unchanged)
// ============================================================

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

// ============================================================
// GPU helpers
// ============================================================

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

// ============================================================
// GPU scale reorder kernel
// ============================================================

}  // close anonymous namespace

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

// ============================================================
// GPU segment extraction kernel — replaces CPU segments_from_indices
// ============================================================

__global__ void extract_segments_kernel(
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ seg_group,   // [max_segments] group id
    int32_t* __restrict__ seg_start,   // [max_segments] start row
    int32_t* __restrict__ seg_count,   // [max_segments] row count
    int32_t* __restrict__ num_segments,// [1] actual number of segments
    int m, int max_segments) {
  // Single-thread kernel — m is typically small (384-16192) and this
  // eliminates GPU→CPU sync. Runs in <10μs even for m=16192.
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  int nseg = 0;
  int cur_group = -2;  // impossible value
  for (int i = 0; i < m; ++i) {
    int gid = expert_ids[i];
    if (gid < 0) {
      cur_group = -2;
      continue;
    }
    if (gid != cur_group) {
      if (nseg >= max_segments) break;
      seg_group[nseg] = gid;
      seg_start[nseg] = i;
      seg_count[nseg] = 1;
      cur_group = gid;
      nseg++;
    } else {
      seg_count[nseg - 1]++;
    }
  }
  *num_segments = nseg;
}

namespace {  // re-open anonymous namespace

// ============================================================
// SFB cache — pre-reordered weight scales, computed once per weight tensor
// ============================================================

struct SFBCacheEntry {
  std::vector<torch::Tensor> per_group_sfb;  // [groups] each is reordered SFB on GPU
  int n;
  int k;
};

static std::mutex sfb_cache_mutex;
static std::unordered_map<uintptr_t, SFBCacheEntry> sfb_cache;

SFBCacheEntry const& get_or_create_sfb_cache(
    torch::Tensor b_scale, int n, int k, int groups, torch::Device device) {
  uintptr_t key = reinterpret_cast<uintptr_t>(b_scale.data_ptr());
  {
    std::lock_guard<std::mutex> lock(sfb_cache_mutex);
    auto it = sfb_cache.find(key);
    if (it != sfb_cache.end()) {
      return it->second;
    }
  }

  const int b_scale_cols = b_scale.dim() >= 2
      ? static_cast<int>(b_scale.size(-1)) : 1;

  SFBCacheEntry entry;
  entry.n = n;
  entry.k = k;
  entry.per_group_sfb.reserve(groups);

  for (int g = 0; g < groups; ++g) {
    torch::Tensor sfb_flat;
    if (b_scale.dim() == 3) {
      sfb_flat = b_scale.select(0, g).contiguous().view(-1).to(torch::kUInt8);
    } else {
      sfb_flat = b_scale.contiguous().view(-1).to(torch::kUInt8);
    }
    if (!sfb_flat.is_cuda()) {
      sfb_flat = sfb_flat.to(device);
    }
    auto sfb_buf = reorder_scale_on_gpu(sfb_flat, n, b_scale_cols);
    entry.per_group_sfb.push_back(sfb_buf);
  }

  std::lock_guard<std::mutex> lock(sfb_cache_mutex);
  auto [it, inserted] = sfb_cache.emplace(key, std::move(entry));
  return it->second;
}

// ============================================================
// Workspace cache — grows but never shrinks
// ============================================================

static torch::Tensor cached_workspace;
static int cached_workspace_device = -1;

torch::Tensor get_workspace(size_t needed, torch::Device device) {
  int dev = device.index();
  if (cached_workspace_device == dev && cached_workspace.defined() &&
      static_cast<size_t>(cached_workspace.numel()) >= needed) {
    return cached_workspace;
  }
  size_t alloc = std::max(needed, static_cast<size_t>(1024 * 1024));
  cached_workspace = torch::empty({static_cast<int64_t>(alloc)},
                                  torch::TensorOptions().device(device).dtype(torch::kUInt8));
  cached_workspace_device = dev;
  return cached_workspace;
}

// ============================================================
// Pre-allocated GPU buffers for segment metadata
// ============================================================

struct SegmentBuffers {
  torch::Tensor seg_group;
  torch::Tensor seg_start;
  torch::Tensor seg_count;
  torch::Tensor num_segments;
  int capacity;
  int device_index;

  SegmentBuffers() : capacity(0), device_index(-1) {}
};

static SegmentBuffers seg_bufs;

SegmentBuffers& get_segment_buffers(int max_segs, torch::Device device) {
  int dev = device.index();
  if (seg_bufs.device_index == dev && seg_bufs.capacity >= max_segs) {
    return seg_bufs;
  }
  int cap = std::max(max_segs, 512);
  auto opts = torch::TensorOptions().device(device).dtype(torch::kInt32);
  seg_bufs.seg_group = torch::empty({cap}, opts);
  seg_bufs.seg_start = torch::empty({cap}, opts);
  seg_bufs.seg_count = torch::empty({cap}, opts);
  seg_bufs.num_segments = torch::empty({1}, opts);
  seg_bufs.capacity = cap;
  seg_bufs.device_index = dev;
  return seg_bufs;
}

// ============================================================
// Optimized launch_grouped_fp8_fp4
// ============================================================

bool launch_grouped_fp8_fp4(torch::Tensor a, torch::Tensor a_scale, torch::Tensor b,
                            torch::Tensor b_scale, torch::Tensor d, torch::Tensor m_indices) {
  const int groups = static_cast<int>(b.size(0));
  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));
  auto device = a.device();
  auto stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();

  // --- Step 1: Extract segments on GPU (no GPU→CPU sync) ---
  auto& sbufs = get_segment_buffers(groups + 1, device);
  extract_segments_kernel<<<1, 1, 0, stream>>>(
      m_indices.data_ptr<int32_t>(),
      sbufs.seg_group.data_ptr<int32_t>(),
      sbufs.seg_start.data_ptr<int32_t>(),
      sbufs.seg_count.data_ptr<int32_t>(),
      sbufs.num_segments.data_ptr<int32_t>(),
      m, sbufs.capacity);

  // We need num_segments on CPU to build pointer arrays.
  // This is ONE sync per call instead of copying the entire m_indices.
  auto nseg_cpu = sbufs.num_segments.to(torch::kCPU, /*non_blocking=*/false);
  int active = nseg_cpu.item<int32_t>();
  if (active <= 0) {
    return false;
  }

  // Copy only the small segment metadata (3 * active int32s instead of m int32s)
  auto seg_group_cpu = sbufs.seg_group.narrow(0, 0, active).to(torch::kCPU, /*non_blocking=*/false);
  auto seg_start_cpu = sbufs.seg_start.narrow(0, 0, active).to(torch::kCPU, /*non_blocking=*/false);
  auto seg_count_cpu = sbufs.seg_count.narrow(0, 0, active).to(torch::kCPU, /*non_blocking=*/false);
  auto* sg = seg_group_cpu.data_ptr<int32_t>();
  auto* ss = seg_start_cpu.data_ptr<int32_t>();
  auto* sc = seg_count_cpu.data_ptr<int32_t>();

  // --- Step 2: Get cached SFB (weight scale reorder done once) ---
  auto const& sfb_entry = get_or_create_sfb_cache(b_scale, n, k, groups, device);

  // --- Step 3: Build CUTLASS arguments ---
  const int a_scale_cols = a_scale.dim() >= 2
      ? static_cast<int>(a_scale.size(1))
      : static_cast<int>(a_scale.numel()) / m;

  const int64_t a_row_bytes = a.stride(0) * a.element_size();
  const int64_t d_row_bytes = d.stride(0) * d.element_size();
  const int64_t b_group_bytes = b.stride(0) * b.element_size();

  auto* a_base = reinterpret_cast<uint8_t*>(a.data_ptr());
  auto* b_base = reinterpret_cast<uint8_t*>(b.data_ptr());
  auto* d_base = reinterpret_cast<uint8_t*>(d.data_ptr());

  std::vector<GroupProblemShape::UnderlyingProblemShape> problem_sizes(active);
  std::vector<GroupedStrideA> stride_a(active);
  std::vector<GroupedStrideB> stride_b(active);
  std::vector<GroupedStrideC> stride_c(active);
  std::vector<GroupedStrideD> stride_d(active);
  std::vector<GroupedLayoutSFA> layout_sfa(active);
  std::vector<GroupedLayoutSFB> layout_sfb(active);
  std::vector<uintptr_t> ptr_a(active);
  std::vector<uintptr_t> ptr_b(active);
  std::vector<uintptr_t> ptr_c(active, 0);
  std::vector<uintptr_t> ptr_d(active);
  std::vector<uintptr_t> ptr_sfa(active);
  std::vector<uintptr_t> ptr_sfb(active);

  std::vector<torch::Tensor> sfa_buffers;
  sfa_buffers.reserve(active);

  for (int i = 0; i < active; ++i) {
    int group = sg[i];
    int start = ss[i];
    int count = sc[i];

    problem_sizes[i] = {count, n, k};
    stride_a[i] = cutlass::make_cute_packed_stride(GroupedStrideA{}, {count, k, 1});
    stride_b[i] = cutlass::make_cute_packed_stride(GroupedStrideB{}, {n, k, 1});
    stride_c[i] = cutlass::make_cute_packed_stride(GroupedStrideC{}, {count, n, 1});
    stride_d[i] = cutlass::make_cute_packed_stride(GroupedStrideD{}, {count, n, 1});
    layout_sfa[i] =
        GroupedGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
            tile_atom_to_shape_SFA(cute::make_shape(count, n, k, 1));
    layout_sfb[i] =
        GroupedGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
            tile_atom_to_shape_SFB(cute::make_shape(count, n, k, 1));

    ptr_a[i] = reinterpret_cast<uintptr_t>(a_base + start * a_row_bytes);
    ptr_b[i] = reinterpret_cast<uintptr_t>(b_base + group * b_group_bytes);
    ptr_d[i] = reinterpret_cast<uintptr_t>(d_base + start * d_row_bytes);

    // SFA: activation scale — changes every call, must reorder each time
    auto sfa_slice = a_scale.narrow(0, start, count).contiguous().view(-1).to(torch::kUInt8);
    auto sfa_buf = reorder_scale_on_gpu(sfa_slice, count, a_scale_cols);
    sfa_buffers.push_back(sfa_buf);
    ptr_sfa[i] = reinterpret_cast<uintptr_t>(sfa_buf.data_ptr());

    // SFB: weight scale — from cache, zero-copy
    ptr_sfb[i] = reinterpret_cast<uintptr_t>(sfb_entry.per_group_sfb[group].data_ptr());
  }

  // --- Step 4: Copy metadata to GPU ---
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

  // --- Step 5: Launch CUTLASS grouped GEMM ---
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
  auto workspace = get_workspace(workspace_size, device);
  void* workspace_ptr = workspace_size == 0 ? nullptr : workspace.data_ptr();

  auto status = gemm.initialize(arguments, workspace_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return false;
  }
  status = gemm.run(stream);
  return status == cutlass::Status::kSuccess;
}

#endif

}  // namespace

// ============================================================
// Public API (unchanged signatures)
// ============================================================

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

void clear_sfb_cache() {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  std::lock_guard<std::mutex> lock(sfb_cache_mutex);
  sfb_cache.clear();
#endif
}
