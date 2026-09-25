#include "pmi/inference.hpp"
#include <stdexcept>
#ifdef PMI_WITH_LLAMA
#include "llama.h"
#include <memory>
#include <vector>
namespace pmi {
GenerationResult run_reference(const std::string& path,const std::string& prompt,int32_t max_tokens,int32_t gpu_layers){
 if(max_tokens<=0)throw std::invalid_argument("max_tokens must be positive");
 ggml_backend_load_all();
 auto mp=llama_model_default_params();mp.n_gpu_layers=gpu_layers;
 std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(path.c_str(),mp),llama_model_free);
 if(!model)throw std::runtime_error("llama.cpp could not load model");
 const auto* vocab=llama_model_get_vocab(model.get());
 int32_t count=-llama_tokenize(vocab,prompt.data(),static_cast<int32_t>(prompt.size()),nullptr,0,true,true);
 if(count<=0)throw std::runtime_error("prompt tokenization failed");
 std::vector<llama_token> prompt_tokens(static_cast<size_t>(count));
 if(llama_tokenize(vocab,prompt.data(),static_cast<int32_t>(prompt.size()),prompt_tokens.data(),count,true,true)<0)throw std::runtime_error("prompt tokenization failed");
 auto cp=llama_context_default_params();cp.n_ctx=static_cast<uint32_t>(count+max_tokens);cp.n_batch=static_cast<uint32_t>(count);cp.n_ubatch=static_cast<uint32_t>(count);cp.no_perf=false;
 std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
 if(!ctx)throw std::runtime_error("llama.cpp could not create context");
 auto sp=llama_sampler_chain_default_params();std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)> sampler(llama_sampler_chain_init(sp),llama_sampler_free);
 if(!sampler)throw std::runtime_error("could not create greedy sampler");
 llama_sampler_chain_add(sampler.get(),llama_sampler_init_greedy());
 GenerationResult result;
 llama_batch batch=llama_batch_get_one(prompt_tokens.data(),static_cast<int32_t>(prompt_tokens.size()));
 for(int32_t step=0;step<max_tokens;step++){
   if(llama_decode(ctx.get(),batch)!=0)throw std::runtime_error("llama_decode failed");
   auto token=llama_sampler_sample(sampler.get(),ctx.get(),-1);
   if(llama_vocab_is_eog(vocab,token))break;
   result.tokens.push_back(token);
   char piece[256];int n=llama_token_to_piece(vocab,token,piece,sizeof(piece),0,true);
   if(n<0)throw std::runtime_error("token-to-piece conversion failed");
   result.text.append(piece,static_cast<size_t>(n));
   batch=llama_batch_get_one(&token,1);
 }
 return result;
}
}
#else
namespace pmi { GenerationResult run_reference(const std::string&,const std::string&,int32_t,int32_t){throw std::runtime_error("PMI was built without llama.cpp; rebuild with CMake");} }
#endif
