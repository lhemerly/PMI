#include "pmi/owned_buffer.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
using namespace pmi;
void check(bool ok) { if (!ok) throw std::runtime_error("owned buffer check failed"); }
template<class F> void rejects(F f) { try { f(); } catch (const std::exception&) { return; } throw std::runtime_error("expected range/advice rejection"); }
#ifdef __linux__
uintptr_t seen; size_t seen_bytes; int seen_flag;
int capture(void* p,size_t n,int flag) { seen=reinterpret_cast<uintptr_t>(p);seen_bytes=n;seen_flag=flag;return 0; }
int fail(void*,size_t,int) { errno=ENOMEM;return -1; }
bool no_huge(uintptr_t address) {
  std::ifstream maps("/proc/self/smaps"); std::string line; bool active=false;
  while(std::getline(maps,line)) {
    unsigned long begin,end;
    if(std::sscanf(line.c_str(),"%lx-%lx",&begin,&end)==2) active=address>=begin&&address<end;
    if(active&&line.rfind("VmFlags:",0)==0) return line.find(" nh")!=std::string::npos;
  }
  throw std::runtime_error("owned mapping unavailable");
}
#endif
int main() {
 try {
  auto r=owned_page_interior(reinterpret_cast<void*>(4097),12288,4096);
  check(r.begin==8192&&r.bytes==8192);
  check(owned_page_interior(reinterpret_cast<void*>(4097),4094,4096).bytes==0);
  check(owned_page_interior(nullptr,0,4096).bytes==0);
  rejects([]{owned_page_interior(nullptr,1,4096);});
  rejects([]{owned_page_interior(reinterpret_cast<void*>(1),8,3);});
  rejects([]{owned_page_interior(reinterpret_cast<void*>(UINTPTR_MAX-1),8,4096);});
#ifdef __linux__
  const auto page=static_cast<size_t>(sysconf(_SC_PAGESIZE));
  void* ptr=mmap(nullptr,4*page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  check(ptr!=MAP_FAILED);
  struct Owner {void* p;size_t n;~Owner(){munmap(p,n);}} owner{ptr,4*page};
  std::memset(ptr,0xa5,4*page);
  auto* interior=static_cast<char*>(ptr)+13;
  r=advise_owned_base_pages(interior,3*page-19,capture);
  check(seen==reinterpret_cast<uintptr_t>(ptr)+page&&seen_bytes==page&&seen_flag==MADV_NOHUGEPAGE);
  rejects([&]{advise_owned_base_pages(interior,3*page-19,fail);});
  check(!no_huge(reinterpret_cast<uintptr_t>(ptr)));
  r=advise_owned_base_pages(interior,3*page-19);
  check(no_huge(r.begin)&&!no_huge(reinterpret_cast<uintptr_t>(ptr))&&
        !no_huge(reinterpret_cast<uintptr_t>(ptr)+3*page));
  for(size_t i=0;i<4*page;++i) check(static_cast<unsigned char*>(ptr)[i]==0xa5);
#endif
  std::cout<<"Owned advice: inward alignment, empty/overflow rejection, injected syscall failure, mapping boundaries and unchanged bytes passed\n";
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
