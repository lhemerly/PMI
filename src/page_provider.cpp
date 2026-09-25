#include "pmi/page_provider.hpp"
#include <fstream>
#include <stdexcept>
namespace pmi {
FilePageProvider::FilePageProvider(std::filesystem::path p,size_t c):path_(std::move(p)),cache_bytes_(c),cache_(c){}
std::shared_ptr<const std::vector<uint8_t>> FilePageProvider::load(const TensorInfo&t){std::lock_guard<std::mutex>g(mutex_);if(auto p=cache_.get(t.absolute_offset))return std::make_shared<const std::vector<uint8_t>>(*p);std::ifstream f(path_,std::ios::binary);if(!f)throw std::runtime_error("cannot open model");f.seekg(static_cast<std::streamoff>(t.absolute_offset));std::vector<uint8_t>b(static_cast<size_t>(t.byte_length));f.read(reinterpret_cast<char*>(b.data()),static_cast<std::streamsize>(b.size()));if(!f)throw std::runtime_error("short tensor read");if(!cache_.put(t.absolute_offset,b))throw std::runtime_error("tensor exceeds cache or all entries pinned");return std::make_shared<const std::vector<uint8_t>>(std::move(b));}
std::shared_future<std::shared_ptr<const std::vector<uint8_t>>> FilePageProvider::request(const TensorInfo&t){auto copy=t;return std::async(std::launch::async,[this,copy=std::move(copy)]{return load(copy);}).share();}std::shared_future<std::shared_ptr<const std::vector<uint8_t>>> FilePageProvider::prefetch(const TensorInfo&t){return request(t);}bool FilePageProvider::pin(const TensorInfo&t){std::lock_guard<std::mutex>g(mutex_);return cache_.pin(t.absolute_offset);}bool FilePageProvider::release(const TensorInfo&t){std::lock_guard<std::mutex>g(mutex_);return cache_.unpin(t.absolute_offset);}
}
