#include "pmi/gguf.hpp"
#include <cstring>
#include <stdexcept>
#include <limits>

namespace pmi {
namespace { constexpr uint32_t MAGIC=0x46554747; uint64_t align_up(uint64_t n,uint64_t a){if(a==0)throw std::runtime_error("GGUF alignment is zero"); if(n>std::numeric_limits<uint64_t>::max()-(a-1))throw std::runtime_error("GGUF alignment overflow"); return (n+a-1)/a*a;} }
GgufReader::GgufReader(const std::filesystem::path& path):path_(path),in_(path,std::ios::binary){
 if(!in_) throw std::runtime_error("cannot open GGUF: "+path.string());
 in_.seekg(0,std::ios::end); file_size_=static_cast<uint64_t>(in_.tellg()); in_.seekg(0);
}
void GgufReader::seek(uint64_t o) const { if(o>file_size_) throw std::runtime_error("GGUF offset outside file"); in_.clear(); in_.seekg(static_cast<std::streamoff>(o)); }
template<class T> T GgufReader::read_scalar() const { T v{}; in_.read(reinterpret_cast<char*>(&v),sizeof(v)); if(!in_) throw std::runtime_error("truncated GGUF"); return v; }
std::string GgufReader::read_string() const { auto n=read_scalar<uint64_t>(); if(n>file_size_ || n>static_cast<uint64_t>(std::numeric_limits<size_t>::max())) throw std::runtime_error("invalid GGUF string length"); std::string s(static_cast<size_t>(n),'\0'); in_.read(s.data(),static_cast<std::streamsize>(n)); if(!in_) throw std::runtime_error("truncated GGUF string"); return s; }
MetadataValue GgufReader::read_value(GgufType t) const { MetadataValue v{t,{}}; switch(t){
 case GgufType::UINT8:v.text=std::to_string(read_scalar<uint8_t>());break; case GgufType::INT8:v.text=std::to_string(read_scalar<int8_t>());break;
 case GgufType::UINT16:v.text=std::to_string(read_scalar<uint16_t>());break; case GgufType::INT16:v.text=std::to_string(read_scalar<int16_t>());break;
 case GgufType::UINT32:v.text=std::to_string(read_scalar<uint32_t>());break; case GgufType::INT32:v.text=std::to_string(read_scalar<int32_t>());break;
 case GgufType::UINT64:v.text=std::to_string(read_scalar<uint64_t>());break; case GgufType::INT64:v.text=std::to_string(read_scalar<int64_t>());break;
 case GgufType::FLOAT32:v.text=std::to_string(read_scalar<float>());break; case GgufType::FLOAT64:v.text=std::to_string(read_scalar<double>());break;
 case GgufType::BOOL:v.text=read_scalar<uint8_t>()?"true":"false";break; case GgufType::STRING:v.text=read_string();break;
 case GgufType::ARRAY:{auto et=static_cast<GgufType>(read_scalar<uint32_t>()); auto n=read_scalar<uint64_t>(); for(uint64_t i=0;i<n;i++){auto x=read_value(et); if(i<16){if(!v.text.empty())v.text+=",";v.text+=x.text;}} v.text="["+v.text+(n>16?",...":"")+"]";break;}
 default: throw std::runtime_error("unknown GGUF metadata type"); } return v; }
GgufFile GgufReader::inspect() const { seek(0); auto magic=read_scalar<uint32_t>(); if(magic!=MAGIC) throw std::runtime_error("not a GGUF file"); GgufFile f; f.version=read_scalar<uint32_t>(); if(f.version<2||f.version>3)throw std::runtime_error("unsupported GGUF version"); auto n_t=read_scalar<uint64_t>(); auto n_k=read_scalar<uint64_t>(); if(n_t>100000000||n_k>1000000) throw std::runtime_error("implausible GGUF counts");
 for(uint64_t i=0;i<n_k;i++){auto k=read_string(); auto t=static_cast<GgufType>(read_scalar<uint32_t>()); f.metadata.emplace(std::move(k),read_value(t));}
 f.tensors.reserve(static_cast<size_t>(n_t)); for(uint64_t i=0;i<n_t;i++){TensorInfo t; t.name=read_string(); auto nd=read_scalar<uint32_t>(); if(nd>64)throw std::runtime_error("invalid GGUF dimensions"); t.shape.resize(nd); for(auto& d:t.shape)d=read_scalar<uint64_t>(); t.ggml_type=read_scalar<uint32_t>(); t.offset=read_scalar<uint64_t>(); f.tensors.push_back(std::move(t));}
 uint64_t alignment=32; if(auto it=f.metadata.find("general.alignment");it!=f.metadata.end()) alignment=std::stoull(it->second.text);
 if(alignment==0 || (alignment&(alignment-1))!=0)throw std::runtime_error("GGUF alignment must be a nonzero power of two");
 f.tensor_data_offset=align_up(static_cast<uint64_t>(in_.tellg()),alignment); f.file_size=file_size_;
 for(auto& t:f.tensors){if(t.offset>std::numeric_limits<uint64_t>::max()-f.tensor_data_offset)throw std::runtime_error("tensor offset overflow");t.absolute_offset=f.tensor_data_offset+t.offset;t.byte_length=ggml_tensor_nbytes(t.ggml_type,t.shape);if(t.absolute_offset>file_size_||t.byte_length>file_size_-t.absolute_offset)throw std::runtime_error("tensor range outside GGUF");}
 for(size_t i=0;i<f.tensors.size();i++)for(size_t j=i+1;j<f.tensors.size();j++){const auto&a=f.tensors[i];const auto&b=f.tensors[j];if(a.absolute_offset<b.absolute_offset+b.byte_length&&b.absolute_offset<a.absolute_offset+a.byte_length)throw std::runtime_error("overlapping GGUF tensor ranges");}
 return f; }
std::string ggml_type_name(uint32_t t){switch(t){case 0:return"F32";case 1:return"F16";case 2:return"Q4_0";case 3:return"Q4_1";case 4:case 5:return"DEPRECATED";case 6:return"Q5_0";case 7:return"Q5_1";case 8:return"Q8_0";case 9:return"Q8_1";case 10:return"Q2_K";case 11:return"Q3_K";case 12:return"Q4_K";case 13:return"Q5_K";case 14:return"Q6_K";case 15:return"Q8_K";case 16:return"IQ2_XXS";case 17:return"IQ2_XS";case 18:return"IQ3_XXS";case 19:return"IQ1_S";case 20:return"IQ4_NL";case 21:return"IQ3_S";case 22:return"IQ2_S";case 23:return"IQ4_XS";case 24:return"I8";case 25:return"I16";case 26:return"I32";case 27:return"I64";case 28:return"F64";case 29:return"IQ1_M";case 30:return"BF16";case 34:return"TQ1_0";case 35:return"TQ2_0";case 39:return"MXFP4";case 40:return"NVFP4";case 41:return"Q1_0";case 42:return"Q2_0";default:return"TYPE_"+std::to_string(t);}}
uint64_t ggml_tensor_nbytes(uint32_t type,const std::vector<uint64_t>& shape){
 if(shape.empty())throw std::runtime_error("GGUF tensor has no dimensions");
 uint64_t block=1,bytes=0;
 switch(type){case 0:block=1;bytes=4;break;case 1:case 30:block=1;bytes=2;break;case 2:block=32;bytes=18;break;case 3:block=32;bytes=20;break;case 6:block=32;bytes=22;break;case 7:block=32;bytes=24;break;case 8:block=32;bytes=34;break;case 9:block=32;bytes=36;break;case 10:block=256;bytes=84;break;case 11:block=256;bytes=110;break;case 12:block=256;bytes=144;break;case 13:block=256;bytes=176;break;case 14:block=256;bytes=210;break;case 15:block=256;bytes=292;break;case 16:block=256;bytes=66;break;case 17:block=256;bytes=74;break;case 18:block=256;bytes=98;break;case 19:block=256;bytes=50;break;case 20:block=32;bytes=18;break;case 21:block=256;bytes=110;break;case 22:block=256;bytes=82;break;case 23:block=256;bytes=136;break;case 24:block=1;bytes=1;break;case 25:block=1;bytes=2;break;case 26:block=1;bytes=4;break;case 27:case 28:block=1;bytes=8;break;case 29:block=256;bytes=56;break;case 34:block=256;bytes=54;break;case 35:block=256;bytes=66;break;case 39:block=32;bytes=17;break;case 40:block=64;bytes=36;break;case 41:block=128;bytes=18;break;case 42:block=64;bytes=18;break;default:throw std::runtime_error("GGML tensor type size is unsupported: "+std::to_string(type));}
 if(shape[0]%block!=0)throw std::runtime_error("tensor row dimension is not divisible by GGML quantization block");
 uint64_t rows=1;for(size_t i=1;i<shape.size();i++){if(shape[i]&&rows>std::numeric_limits<uint64_t>::max()/shape[i])throw std::runtime_error("tensor element count overflow");rows*=shape[i];}
 uint64_t row_bytes=(shape[0]/block)*bytes;if(rows&&row_bytes>std::numeric_limits<uint64_t>::max()/rows)throw std::runtime_error("tensor byte size overflow");return row_bytes*rows;
}
} // namespace pmi
