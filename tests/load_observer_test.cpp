#include "llama.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <vector>
static void* watched=nullptr;
static size_t watched_frees=0;
extern "C" void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
  using Fn=void(*)(ggml_backend_buffer_t);
  static auto actual=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"ggml_backend_buffer_free"));
  if(!actual) std::terminate();
  if(buffer&&watched&&!ggml_backend_buffer_is_multi_buffer(buffer)&&ggml_backend_buffer_get_base(buffer)==watched) ++watched_frees;
  actual(buffer);
}
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
struct Observer {
  llama_load_event_kind reject;
  size_t buffers{},reads{};
  static bool event(const llama_load_event* e,void* p) noexcept {
    auto& o=*static_cast<Observer*>(p);
    if(e->kind==LLAMA_LOAD_BUFFER_READY){++o.buffers;watched=e->owned_cpu_buffer;}
    if(e->kind==LLAMA_LOAD_READ_BEGIN)++o.reads;
    return e->kind!=o.reject;
  }
};
int main(int argc,char** argv){
 try {
  check(argc>1,"needs tiny GGUF splits");
  llama_backend_init();
  auto params=llama_model_default_params();params.n_gpu_layers=0;
  ggml_backend_dev_t devices[]={nullptr};params.devices=devices;
  params.load_mode=LLAMA_LOAD_MODE_DIRECT_IO_REQUIRED;params.lazy_mode=LLAMA_LAZY_MODE_OFF;params.use_extra_bufts=false;
  std::vector<const char*> paths;for(int i=1;i<argc;++i)paths.push_back(argv[i]);
  for(auto rejected:{LLAMA_LOAD_BUFFER_READY,LLAMA_LOAD_READ_BEGIN}){
    watched=nullptr;watched_frees=0;Observer observer{rejected};
    llama_load_observer binding{Observer::event,&observer};
    auto* model=llama_model_load_from_splits_observed(paths.data(),paths.size(),params,&binding);
    check(!model,"rejected observer returned model");
    check(observer.buffers==1&&watched&&watched_frees==1,"observer rejection leaked owned CPU buffer");
    if(rejected==LLAMA_LOAD_READ_BEGIN)check(observer.reads==1,"read rejection failed to exercise chunk boundary");
    const auto calls=observer.buffers;watched=nullptr;
    model=llama_model_load_from_splits(paths.data(),paths.size(),params);
    check(model,"default loading failed after observer rejection");
    check(observer.buffers==calls,"observer scope leaked into default load");
    llama_model_free(model);
  }
  std::cout<<"Observer allocation/read rejection frees CPU ownership; scope resets; default load recovers\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
