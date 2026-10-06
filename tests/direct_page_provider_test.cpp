#include "pmi/page_provider.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void require(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,"expected direct provider failure");}
}
int main(int argc,char** argv){
 try {
  auto directory=(argc>1?std::filesystem::path(argv[1]):std::filesystem::temp_directory_path()/"pmi-direct") +=
    "-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  require(std::filesystem::create_directories(directory),"exclusive fixture directory");
  std::vector<std::filesystem::path> shards;std::vector<std::vector<uint8_t>> source;
  for(unsigned shard=0;shard<2;++shard){
   auto path=directory/(std::to_string(shard)+".bin");shards.push_back(path);source.emplace_back(8195);
   uint32_t value=531+shard;
   for(auto& x:source.back()){value^=value<<13;value^=value>>17;value^=value<<5;x=static_cast<uint8_t>(value);}
   std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(source.back().data()),source.back().size());
  }
  pmi::FilePageProvider provider(shards,65536);bool native=false;
  for(uint32_t shard=0;shard<2;++shard)for(auto range:std::vector<std::pair<uint64_t,uint64_t>>{{1,131},{63,65},{4095,129},{8180,15}}){
   pmi::TensorInfo tensor;tensor.shard_id=shard;tensor.absolute_offset=range.first;tensor.byte_length=range.second;
   auto buffered=provider.request_transient(tensor).get();auto direct=provider.request_direct(tensor,true,false).get();
   require(*buffered==*direct,"direct/fallback source bytes differ");
   require(reinterpret_cast<uintptr_t>(direct->data())%64==0,"logical payload lost 64-byte alignment");
   require(std::equal(direct->data(),direct->data()+direct->size(),source[shard].data()+range.first),"wrong shard/range");
   native=provider.io_stats().direct_completed>0;
   direct.reset();buffered.reset();require(provider.stats().current_bytes==0,"transient bytes leaked");
  }
  require(provider.stats().peak_bytes<=65536,"staging escaped quota");
  pmi::TensorInfo tensor;tensor.byte_length=64;
  auto normal=provider.request(tensor).get();require(provider.pin(tensor),"ordinary pin failed");
  auto hit=provider.request_direct(tensor,true,true).get();require(hit->data()==normal->data(),"cached direct hit changed allocation");
  require(provider.stats().pinned_entries==1,"direct hit lost ordinary pin");hit.reset();normal.reset();require(provider.release(tensor),"ordinary pin release failed");
  pmi::PageLease survivor;
  {pmi::FilePageProvider temporary(shards,65536);tensor.absolute_offset=4095;tensor.byte_length=129;
   survivor=temporary.request_direct(tensor,true,false).get();}
  require((*survivor)[0]==source[0][4095],"direct lease did not survive runtime");survivor.reset();
  {pmi::FilePageProvider temporary(shards,65536);tensor.absolute_offset=0;tensor.byte_length=64;
   std::filesystem::resize_file(shards[0],2);rejects([&]{temporary.request_direct(tensor,true,false).get();});
   require(temporary.stats().current_bytes==0 && temporary.stats().entries==0,"partial read published/leaked payload");}
  const auto io=provider.io_stats();
  require(native ? (io.direct_read_bytes>io.direct_logical_bytes && io.fallback_count==0) : io.fallback_count==8,"I/O accounting/fallback mismatch");
  std::cout<<"Direct provider exact bytes/alignment/shards/EOF/budget/pins/lease/failed-read passed; native="<<native
           <<" direct_api_bytes="<<io.direct_read_bytes<<" fallback="<<io.fallback_count<<" staging_peak="<<io.staging_peak_bytes<<"\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
