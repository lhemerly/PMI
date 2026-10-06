#include "pmi/page_provider.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
  bool caught = false;
  try { action(); } catch (const std::exception&) { caught = true; }
  require(caught, "expected provider failure");
}
struct Directory {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("pmi-provider-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  Directory() { std::filesystem::create_directories(path); }
  ~Directory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
}
int main() {
  try {
    Directory directory;
    auto a = directory.path / "a.bin", b = directory.path / "b.bin";
    { std::ofstream f(a, std::ios::binary); f << "abcdefgh"; }
    { std::ofstream f(b, std::ios::binary); f << "ABCDEFGH"; }
    pmi::TensorInfo first; first.byte_length = 4;
    pmi::TensorInfo other = first; other.shard_id = 1;
    pmi::TensorInfo overlapping = first; overlapping.byte_length = 3;
    pmi::PageLease survivor;
    {
      pmi::FilePageProvider provider(std::vector<std::filesystem::path>{a, b}, 128);
      auto x = provider.request(first).get();
      auto y = provider.prefetch(first).get();
      require(x->data() == y->data(), "cache hit copied the payload");
      require(reinterpret_cast<uintptr_t>(x->data()) % 64 == 0, "page alignment");
      auto z = provider.request(other).get();
      require((*x)[0] == 'a' && (*z)[0] == 'A', "shard identity collision");
      rejects([&] { provider.request(overlapping).get(); });
      require(provider.stats().current_bytes == 128, "leased bytes escaped budget");
      x.reset(); y.reset(); z.reset();
      require(provider.pin(first) && provider.pin(first), "counted pin acquisition");
      require(provider.release(first), "first pin release");
      auto short_page = provider.request(overlapping).get();
      require(short_page->size() == 3, "range length missing from cache key");
      rejects([&] { provider.request(other).get(); });
      require(provider.release(first), "second pin release");
      auto reloaded = provider.request(other).get();
      require((*reloaded)[3] == 'D', "evicted shard reload mismatch");
      short_page.reset(); reloaded.reset();
      auto future = std::async(std::launch::async, [&] { return provider.request(first).get(); });
      survivor = future.get();
      require(provider.stats().hits >= 1 && provider.stats().evictions >= 2, "missing cache telemetry");
    }
    require((*survivor)[3] == 'd', "lease did not survive provider destruction");
    {
      pmi::FilePageProvider provider(a, 64);
      std::filesystem::resize_file(a, 2);
      rejects([&] { provider.request(first).get(); });
      require(provider.stats().current_bytes == 0 && provider.stats().failures > 0,
              "failed I/O installed a cache entry");
    }
    std::cout << "page provider bridge tests passed (leases, alignment, shards, counted pins, reload, errors)\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
