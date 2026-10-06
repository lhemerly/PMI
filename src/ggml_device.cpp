#include "pmi/ggml_device.hpp"

#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pmi {
namespace {
[[noreturn]] void fail(const char* message) {
  throw std::runtime_error(std::string("ggml device page: ") + message);
}

// Validate before invoking ggml constructors, which assert on malformed shapes.
void validate(const DevicePageRequest& request) {
  const auto& source = request.source;
  if (request.format != GgmlDeviceBackend::format || request.layout != GgmlDeviceBackend::layout)
    fail("unsupported format or layout");
  if (source.shape.empty() || source.shape.size() > 2)
    fail("expected one compact row or matrix, with no expert batch dimension");
  if (source.ggml_type >= GGML_TYPE_COUNT) fail("invalid ggml type");
  const auto* traits = ggml_get_type_traits(static_cast<ggml_type>(source.ggml_type));
  if (!traits || traits->blck_size <= 0 || !traits->type_size) fail("unsupported ggml type");
  uint64_t elements = 1;
  for (const uint64_t dimension : source.shape) {
    if (!dimension || dimension > static_cast<uint64_t>(INT64_MAX) ||
        elements > static_cast<uint64_t>(INT64_MAX) / dimension)
      fail("invalid or overflowing dimensions");
    elements *= dimension;
  }
  if (source.shape.front() % static_cast<uint64_t>(traits->blck_size))
    fail("row length is not a quantization block multiple");
  const uint64_t blocks = elements / static_cast<uint64_t>(traits->blck_size);
  if (blocks > static_cast<uint64_t>(SIZE_MAX) / traits->type_size)
    fail("payload size overflow");
  const uint64_t bytes = blocks * traits->type_size;
  if (!bytes || bytes != source.byte_length || bytes > static_cast<uint64_t>(INT64_MAX))
    fail("payload length disagrees with compact tensor shape");
  if (source.absolute_offset > UINT64_MAX - bytes) fail("source range overflow");
}

struct Metadata final {
  ggml_context* context{};
  ggml_tensor* tensor{};
  explicit Metadata(const DevicePageRequest& request) {
    validate(request);
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead();
    params.no_alloc = true;
    context = ggml_init(params);
    if (!context) fail("cannot allocate tensor metadata");
    const auto& source = request.source;
    int64_t dimensions[2] = {static_cast<int64_t>(source.shape[0]), 1};
    if (source.shape.size() == 2) dimensions[1] = static_cast<int64_t>(source.shape[1]);
    tensor = ggml_new_tensor(context, static_cast<ggml_type>(source.ggml_type),
                             static_cast<int>(source.shape.size()), dimensions);
    ggml_set_name(tensor, "pmi.compact.weight");
  }
  ~Metadata() { ggml_free(context); }
  Metadata(const Metadata&) = delete;
  Metadata& operator=(const Metadata&) = delete;
};

uint64_t quote(ggml_backend_buffer_type_t type, const Metadata& metadata) {
  const uint64_t alignment = ggml_backend_buft_get_alignment(type);
  const uint64_t bytes = ggml_backend_buft_get_alloc_size(type, metadata.tensor);
  if (!alignment || !bytes || bytes > static_cast<uint64_t>(SIZE_MAX) - (alignment - 1))
    fail("invalid or overflowing allocation alignment");
  const uint64_t aligned = ((bytes + alignment - 1) / alignment) * alignment;
  if (aligned < ggml_nbytes(metadata.tensor) || aligned > ggml_backend_buft_get_max_size(type))
    fail("tensor exceeds backend buffer limit");
  return aligned;
}

bool same_descriptor(const DevicePageRequest& a, const DevicePageRequest& b) {
  return a.format == b.format && a.layout == b.layout &&
      a.source.shard_id == b.source.shard_id && a.source.absolute_offset == b.source.absolute_offset &&
      a.source.byte_length == b.source.byte_length && a.source.ggml_type == b.source.ggml_type &&
      a.source.shape == b.source.shape;
}

// Returning after an unsuccessful fence would let the caller free a host page
// or weight still in use. The public ggml API has no recoverable fence status;
// an exception from synchronization is therefore a fail-stop boundary.
void synchronize_or_terminate(ggml_backend_t backend) noexcept {
  try { ggml_backend_synchronize(backend); }
  catch (...) { std::terminate(); }
}
} // namespace

struct GgmlDeviceBackend::State {
  GgmlBackendOwner owner;
  ggml_backend_buffer_type_t buffer_type{};

  struct Allocation final : DeviceAllocation {
    std::shared_ptr<State> state;
    DevicePageRequest descriptor;
    Metadata metadata;
    ggml_backend_buffer_t buffer{};
    uint64_t charge{};
    Allocation(std::shared_ptr<State> s, const DevicePageRequest& request, uint64_t bytes)
        : state(std::move(s)), descriptor(request), metadata(request), charge(bytes) {}
    ~Allocation() noexcept override {
      // Also protects accidental last-lease release after work was submitted on
      // this single ordered stream. Consumers still own the explicit contract.
      if (buffer) synchronize_or_terminate(state->owner.get());
      ggml_backend_buffer_free(buffer);
    }
    uint64_t charged_bytes() const noexcept override { return charge; }
  };
};

GgmlDeviceBackend::GgmlDeviceBackend(GgmlBackendOwner owner) {
  if (!owner) fail("missing caller-owned backend");
  // A distributed/meta buffer has different allocation and tensor-base
  // semantics. Inspect only the supplied handle; do not enumerate devices.
  const auto device = ggml_backend_get_device(owner.get());
  if (device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_META)
    fail("distributed/meta backends are not supported by the compact-page adapter");
  state_ = std::make_shared<State>();
  state_->owner = std::move(owner);
  state_->buffer_type = ggml_backend_get_default_buffer_type(state_->owner.get());
  if (!state_->buffer_type) fail("backend has no default buffer type");
}
GgmlDeviceBackend::~GgmlDeviceBackend() noexcept = default;

uint64_t GgmlDeviceBackend::allocation_bytes(const DevicePageRequest& request) {
  const Metadata metadata(request);
  return quote(state_->buffer_type, metadata);
}

std::unique_ptr<DeviceAllocation> GgmlDeviceBackend::allocate(
    const DevicePageRequest& request, uint64_t charged_bytes) {
  auto allocation = std::make_unique<State::Allocation>(state_, request, charged_bytes);
  if (charged_bytes != quote(state_->buffer_type, allocation->metadata))
    fail("allocation charge changed or was not quoted by this backend");
  allocation->buffer = ggml_backend_buft_alloc_buffer(state_->buffer_type, static_cast<size_t>(charged_bytes));
  if (!allocation->buffer) fail("backend allocation failed");
  if (ggml_backend_buffer_get_size(allocation->buffer) != charged_bytes)
    fail("backend buffer size differs from reserved charge");
  ggml_backend_buffer_set_usage(allocation->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  void* base = ggml_backend_buffer_get_base(allocation->buffer);
  if (!base) fail("backend does not expose a tensor allocation base");
  if (ggml_backend_tensor_alloc(allocation->buffer, allocation->metadata.tensor, base) != GGML_STATUS_SUCCESS)
    fail("backend tensor initialization failed");
  return allocation;
}

void GgmlDeviceBackend::upload_sync(
    DeviceAllocation& opaque, const Page& source, const DevicePageRequest& request) {
  auto* allocation = dynamic_cast<State::Allocation*>(&opaque);
  if (!allocation || allocation->state != state_) fail("allocation belongs to another adapter");
  if (!same_descriptor(request, allocation->descriptor) || source.size() != request.source.byte_length)
    fail("upload descriptor or host page length mismatch");
  try {
    ggml_backend_tensor_set(allocation->metadata.tensor, source.data(), 0, source.size());
  } catch (...) {
    synchronize_or_terminate(state_->owner.get());
    throw;
  }
  synchronize_or_terminate(state_->owner.get());
}

ggml_tensor* GgmlDeviceBackend::tensor(const DevicePage& page) const {
  const auto* allocation = dynamic_cast<const State::Allocation*>(&page.allocation());
  if (!allocation || allocation->state != state_) fail("page belongs to another adapter");
  return allocation->metadata.tensor;
}

ggml_backend_t GgmlDeviceBackend::backend() const noexcept { return state_->owner.get(); }

} // namespace pmi
