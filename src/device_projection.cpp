#include "pmi/device_projection.hpp"
#include "pmi/ggml_device.hpp"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#ifdef PMI_WITH_VULKAN
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pmi {
namespace {
[[noreturn]] void fail(const char* message) {
  throw std::runtime_error(std::string("device projection: ") + message);
}
uint64_t count(uint64_t value, uint64_t increment = 1) noexcept {
  return increment > UINT64_MAX - value ? UINT64_MAX : value + increment;
}
uint64_t add(uint64_t a, uint64_t b) {
  if (b > UINT64_MAX - a) fail("allocation size overflow");
  return a + b;
}
uint64_t rounded(uint64_t bytes, uint64_t alignment) {
  if (!alignment) fail("zero backend alignment");
  const auto result = add(bytes, alignment - 1) / alignment * alignment;
  if (result > SIZE_MAX) fail("allocation exceeds address space");
  return result;
}
class Timer final {
 public:
  explicit Timer(double* destination) : destination_(destination) {
    if (destination_) start_ = std::chrono::steady_clock::now();
  }
  ~Timer() {
    if (destination_) *destination_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start_).count();
  }
 private:
  double* destination_;
  std::chrono::steady_clock::time_point start_{};
};
void synchronize(ggml_backend_t backend) noexcept {
  // Same lifetime boundary as the allocation adapter: a failed fence cannot
  // safely return and let host or graph buffers disappear.
  try { ggml_backend_synchronize(backend); } catch (...) { std::terminate(); }
}
} // namespace

struct GgmlDeviceProjection::Impl {
  std::shared_ptr<GgmlDeviceBackend> backend;
  DeviceCache cache;
  ProjectionStats counters;
  ProjectionActivationMode activation_mode;
  bool in_call{};
  Impl(PageProvider& provider, std::shared_ptr<GgmlDeviceBackend> b, uint64_t cache_bytes,
       uint64_t graph_bytes, bool profile, ProjectionActivationMode mode, DeviceCacheObserver observer, bool transient_upload_host, HostReadPolicy io)
      : backend(std::move(b)), cache(provider, backend, cache_bytes, profile, std::move(observer), transient_upload_host, io), activation_mode(mode) {
    if (!graph_bytes) fail("graph buffer capacity must be nonzero");
    if (mode != ProjectionActivationMode::BackendNative && mode != ProjectionActivationMode::CpuCompatible)
      fail("unknown activation preparation mode");
    if (mode == ProjectionActivationMode::CpuCompatible) {
#ifdef PMI_WITH_VULKAN
      if (!ggml_backend_is_vk(backend->backend()))
#endif
        fail("CPU-compatible activation preparation requires a Vulkan backend");
    }
    counters.graph_capacity_bytes = graph_bytes;
    counters.profile_enabled = profile;
  }
  double* time(double& value) noexcept { return counters.profile_enabled ? &value : nullptr; }

  struct Call final {
    Impl& owner;
    ggml_context* context{};
    ggml_backend_buffer_t buffer{};
    DeviceLease weight;
    std::vector<float> result;
    std::vector<float> prepared_input;
    std::vector<block_q8_0> packed_q8_0;
    std::vector<block_q8_K> packed_q8_K;
    explicit Call(Impl& value) : owner(value) { owner.in_call = true; }
    ~Call() noexcept {
      Timer timer(owner.time(owner.counters.graph_teardown_ms));
      // Drain even when a kernel/upload API throws; never free graph/weight
      // storage while the selected stream could still consume it.
      if (buffer) synchronize(owner.backend->backend());
      ggml_backend_buffer_free(buffer);
      owner.counters.graph_current_bytes = 0;
      ggml_free(context);
      weight.reset();
      owner.in_call = false;
    }
  };
};

GgmlDeviceProjection::GgmlDeviceProjection(PageProvider& provider,
    std::shared_ptr<GgmlDeviceBackend> backend, uint64_t cache_bytes, uint64_t graph_bytes, bool profile,
    ProjectionActivationMode activation_mode, DeviceCacheObserver observer, bool transient_upload_host, HostReadPolicy io)
    : impl_(std::make_unique<Impl>(provider, std::move(backend), cache_bytes, graph_bytes, profile, activation_mode, std::move(observer), transient_upload_host, io)) {}
GgmlDeviceProjection::~GgmlDeviceProjection() noexcept = default;

std::vector<float> GgmlDeviceProjection::project(const TensorInfo& source, const std::vector<float>& input) {
  return compute(source, input, {});
}
DeviceLease GgmlDeviceProjection::resident(const TensorInfo& source, bool seed) {
  auto& state = *impl_;
  if (state.in_call) fail("resident planning cannot overlap a projection");
  DevicePageRequest request{source, GgmlDeviceBackend::format, GgmlDeviceBackend::layout};
  Timer timer(state.time(state.counters.cache_request_ms));
  return seed ? state.cache.seed(request) : state.cache.lookup(request);
}
std::vector<float> GgmlDeviceProjection::project_resident(const DeviceLease& page, const std::vector<float>& input) {
  if (!page) fail("resident projection requires a device lease");
  return compute(page->descriptor().source, input, page);
}
std::vector<float> GgmlDeviceProjection::compute(const TensorInfo& source, const std::vector<float>& input, DeviceLease resident) {
  auto& state = *impl_;
  state.counters.calls = count(state.counters.calls);
  if (state.in_call) {
    state.counters.failures = count(state.counters.failures);
    fail("reentrant projection is unsupported");
  }
  Impl::Call call(state);
  try {
    DevicePageRequest request{source, GgmlDeviceBackend::format, GgmlDeviceBackend::layout};
    ggml_tensor* x{};
    ggml_tensor* output{};
    const ggml_type_traits_cpu* activation_traits{};
    ggml_type activation_type = GGML_TYPE_F32;
    uint64_t input_charge{}, graph_charge{};
    auto buffer_type = ggml_backend_get_default_buffer_type(state.backend->backend());
    {
      Timer timer(state.time(state.counters.graph_setup_ms));
      if (source.shape.size() != 2 || source.shape[0] != input.size() || input.empty())
        fail("matrix shape disagrees with input vector");
      // This validates quantization type/block/dimensions/range without reading
      // host weights or allocating backend storage.
      state.backend->allocation_bytes(request);
      // The pinned CPU supports_op accepts any F32 rhs, including lhs types
      // with no dot kernel (e.g. I32). Reject those before dispatch can assert.
      if (ggml_backend_is_cpu(state.backend->backend())) {
        const auto* traits = ggml_get_type_traits_cpu(static_cast<ggml_type>(source.ggml_type));
        if (!traits || !traits->vec_dot) fail("CPU backend has no matrix dot kernel for this weight type");
      }
      if (source.shape[0] > SIZE_MAX / sizeof(float) || source.shape[1] > SIZE_MAX / sizeof(float))
        fail("activation or output size overflow");
      if (state.activation_mode == ProjectionActivationMode::CpuCompatible) {
        const auto weight_type = static_cast<ggml_type>(source.ggml_type);
        switch (weight_type) {
          case GGML_TYPE_IQ3_S:
          case GGML_TYPE_IQ4_XS: activation_type = GGML_TYPE_Q8_K; break;
          case GGML_TYPE_IQ4_NL:
          case GGML_TYPE_Q8_0: activation_type = GGML_TYPE_Q8_0; break;
          default: fail("CPU-compatible activation preparation is unsupported for this weight type");
        }
        const auto* weight_traits = ggml_get_type_traits_cpu(weight_type);
        activation_traits = ggml_get_type_traits_cpu(activation_type);
        if (!weight_traits || !weight_traits->vec_dot || weight_traits->vec_dot_type != activation_type ||
            !activation_traits || !activation_traits->from_float || input.size() % ggml_blck_size(activation_type))
          fail("native CPU activation quantization traits disagree with the selected weight type");
      }
      ggml_init_params params{};
      params.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false);
      params.no_alloc = true;
      call.context = ggml_init(params);
      if (!call.context) fail("cannot allocate graph metadata");
      auto* placeholder = ggml_new_tensor_2d(call.context, static_cast<ggml_type>(source.ggml_type),
          static_cast<int64_t>(source.shape[0]), static_cast<int64_t>(source.shape[1]));
      x = ggml_new_tensor_2d(call.context, GGML_TYPE_F32, static_cast<int64_t>(input.size()), 1);
      output = ggml_mul_mat(call.context, placeholder, x);
      if (activation_traits && !ggml_prec_set_src(output, GGML_PREC_F32, 1))
        fail("matrix operation cannot preserve the prepared F32 activation");
      if (!ggml_backend_supports_op(state.backend->backend(), output))
        fail("selected backend does not support this matrix multiplication");
      const auto alignment = ggml_backend_buft_get_alignment(buffer_type);
      input_charge = rounded(ggml_backend_buft_get_alloc_size(buffer_type, x), alignment);
      graph_charge = add(input_charge, rounded(ggml_backend_buft_get_alloc_size(buffer_type, output), alignment));
      if (graph_charge > state.counters.graph_capacity_bytes ||
          graph_charge > ggml_backend_buft_get_max_size(buffer_type))
        fail("input/output graph allocation exceeds capacity or backend limit");
    }
    if (activation_traits) {
      Timer timer(state.time(state.counters.activation_preparation_ms));
      for (const float value : input)
        if (!std::isfinite(value)) fail("CPU-compatible activation input must be finite");
      call.prepared_input.resize(input.size());
      if (activation_type == GGML_TYPE_Q8_K) {
        call.packed_q8_K.resize(input.size() / QK_K);
        activation_traits->from_float(input.data(), call.packed_q8_K.data(), static_cast<int64_t>(input.size()));
        dequantize_row_q8_K(call.packed_q8_K.data(), call.prepared_input.data(), static_cast<int64_t>(input.size()));
      } else {
        call.packed_q8_0.resize(input.size() / QK8_0);
        activation_traits->from_float(input.data(), call.packed_q8_0.data(), static_cast<int64_t>(input.size()));
        dequantize_row_q8_0(call.packed_q8_0.data(), call.prepared_input.data(), static_cast<int64_t>(input.size()));
      }
      const auto preparation_bytes = add(call.prepared_input.capacity() * sizeof(float),
          add(call.packed_q8_0.capacity() * sizeof(block_q8_0), call.packed_q8_K.capacity() * sizeof(block_q8_K)));
      state.counters.activation_preparation_peak_bytes = std::max(
          state.counters.activation_preparation_peak_bytes, preparation_bytes);
      for (const float value : call.prepared_input)
        if (!std::isfinite(value)) fail("CPU-compatible prepared activation must be finite");
      state.counters.activation_preparation_calls = count(state.counters.activation_preparation_calls);
      state.counters.activation_preparation_bytes = count(
          state.counters.activation_preparation_bytes, input.size() * sizeof(float));
    }
    // Allocate CPU output before any weight I/O as well. It is separate from
    // the reported backend graph buffer and remains part of process telemetry.
    // Call owns readback storage: its destructor synchronizes before members
    // are destroyed, including when the backend readback API throws mid-copy.
    call.result.resize(static_cast<size_t>(source.shape[1]));
    if (resident) {
      call.weight = std::move(resident);
    } else {
      Timer timer(state.time(state.counters.cache_request_ms));
      call.weight = state.cache.request(request);
    }
    ggml_cgraph* graph{};
    {
      Timer timer(state.time(state.counters.graph_setup_ms));
      output->src[0] = state.backend->tensor(*call.weight);
      if (!ggml_backend_supports_op(state.backend->backend(), output))
        fail("selected backend rejects the allocated weight layout");
      const auto alignment = ggml_backend_buft_get_alignment(buffer_type);
      if (add(input_charge, rounded(ggml_backend_buft_get_alloc_size(buffer_type, output), alignment)) != graph_charge)
        fail("allocated weight changed the graph allocation quote");
      call.buffer = ggml_backend_buft_alloc_buffer(buffer_type, static_cast<size_t>(graph_charge));
      if (!call.buffer) fail("graph buffer allocation failed");
      const uint64_t reported = ggml_backend_buffer_get_size(call.buffer);
      state.counters.graph_current_bytes = reported;
      state.counters.graph_peak_bytes = std::max(state.counters.graph_peak_bytes, reported);
      if (reported != graph_charge)
        fail("graph buffer size differs from its admitted quote");
      ggml_backend_buffer_set_usage(call.buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
      auto* base = static_cast<uint8_t*>(ggml_backend_buffer_get_base(call.buffer));
      if (!base) fail("graph buffer has no tensor allocation base");
      if (ggml_backend_tensor_alloc(call.buffer, x, base) != GGML_STATUS_SUCCESS ||
          ggml_backend_tensor_alloc(call.buffer, output, base + input_charge) != GGML_STATUS_SUCCESS)
        fail("graph tensor initialization failed");
      graph = ggml_new_graph_custom(call.context, 8, false);
      ggml_build_forward_expand(graph, output);
    }
    {
      Timer timer(state.time(state.counters.activation_upload_ms));
      ggml_backend_tensor_set(x, activation_traits ? call.prepared_input.data() : input.data(),
          0, input.size() * sizeof(float));
      synchronize(state.backend->backend());
      state.counters.activation_upload_bytes = count(state.counters.activation_upload_bytes, input.size() * sizeof(float));
    }
    {
      Timer timer(state.time(state.counters.kernel_ms));
      if (ggml_backend_graph_compute(state.backend->backend(), graph) != GGML_STATUS_SUCCESS)
        fail("selected backend matrix multiplication failed");
    }
    {
      Timer timer(state.time(state.counters.output_readback_ms));
      ggml_backend_tensor_get(output, call.result.data(), 0, call.result.size() * sizeof(float));
      synchronize(state.backend->backend());
      state.counters.output_readback_bytes = count(state.counters.output_readback_bytes, call.result.size() * sizeof(float));
    }
    state.counters.completed = count(state.counters.completed);
    return std::move(call.result);
  } catch (...) {
    state.counters.failures = count(state.counters.failures);
    throw;
  }
}

ProjectionStats GgmlDeviceProjection::stats() const noexcept {
  auto result = impl_->counters;
  result.cache = impl_->cache.stats();
  return result;
}
void GgmlDeviceProjection::clear() noexcept { if (!impl_->in_call) impl_->cache.clear(); }

} // namespace pmi
