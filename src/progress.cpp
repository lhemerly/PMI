#include "pmi/progress.hpp"
#include <cerrno>
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>
namespace pmi {
ProgressWriter::ProgressWriter(const std::filesystem::path& path, uint64_t bytes, uint64_t rows)
    : byte_limit_(bytes), row_limit_(rows), detail_bytes_(std::max<uint64_t>(1024,bytes/32)), detail_rows_(std::max<uint64_t>(8,rows/64)) {
  if (bytes<4096 || rows<64) throw std::invalid_argument("progress limits too small");
  fd_=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
  if(fd_<0) throw std::system_error(errno,std::generic_category(),"create progress JSONL");
  try {
    struct stat s{};
    if(::fstat(fd_,&s) || !S_ISREG(s.st_mode)) throw std::runtime_error("progress requires regular file");
    append("run_begin","\"process_interruption_scope\":true,\"power_loss_durable\":false");
  } catch (...) {::close(fd_);fd_=-1;throw;}
}
ProgressWriter::~ProgressWriter() noexcept {
  if(fd_<0)return;
  try {if(!finished_ && !failed_) append("incomplete","\"reason\":\"scope_exit\"");} catch (...) {}
  ::close(fd_);
}
void ProgressWriter::write(std::string_view phase,std::string_view fields) {
  const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  std::string row="{\"schema\":\"pmi.progress.v1\",\"sequence\":"+std::to_string(rows_)+
      ",\"monotonic_ns\":"+std::to_string(ns)+",\"phase\":\""+std::string(phase)+"\"";
  if(!fields.empty())row+=","+std::string(fields);
  row+="}\n";
  if(rows_>=row_limit_ || row.size()>byte_limit_-bytes_)
    throw std::runtime_error("progress critical capture limit");
  size_t offset=0;
  while(offset<row.size()) {
    const auto n=::write(fd_,row.data()+offset,row.size()-offset);
    if(n<0 && errno==EINTR)continue;
    if(n<=0){failed_=true;throw std::system_error(n<0?errno:EIO,std::generic_category(),"write progress JSONL");}
    offset+=static_cast<size_t>(n);bytes_+=static_cast<uint64_t>(n);
  }
  ++rows_;
}
bool ProgressWriter::append(std::string_view phase,std::string_view fields,bool detail) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(failed_ || finished_)throw std::logic_error("progress capture failed or finished");
  if(phase.empty() || phase.size()>64 || fields.size()>16384)
    throw std::invalid_argument("progress record size");
  for(char c:phase)if(!((c>='a'&&c<='z') || (c>='0'&&c<='9') || c=='_'))
    throw std::invalid_argument("progress phase framing");
  for(unsigned char c:fields)if(c<0x20)throw std::invalid_argument("progress JSON framing");
  if(detail && (details_truncated_ || rows_>=row_limit_-detail_rows_ ||
      bytes_>=byte_limit_-detail_bytes_ || fields.size()+256>byte_limit_-detail_bytes_-bytes_)) {
    if(!details_truncated_){details_truncated_=true;write("detail_limit","\"details_truncated\":true");}
    return false;
  }
  write(phase,fields);return true;
}
void ProgressWriter::finish() {
  std::lock_guard<std::mutex> lock(mutex_);
  if(finished_)return;
  if(failed_)throw std::runtime_error("progress prior write failure");
  write("run_complete",details_truncated_?"\"details_truncated\":true":"\"details_truncated\":false");
  finished_=true;
}
}
