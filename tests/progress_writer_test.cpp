#ifdef NDEBUG
#undef NDEBUG
#endif
#include "pmi/progress.hpp"
#include <cassert>
#include <fstream>
#include <iterator>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/resource.h>
static std::string read(const std::string& p) {std::ifstream f(p);return {std::istreambuf_iterator<char>(f),{}};}
int main(int argc,char** argv) {
  assert(argc==2);std::string base=std::string(argv[1])+std::to_string(getpid());
  {pmi::ProgressWriter w(base+"-bounded",4096,64);
   for(int i=0;i<100;++i)w.append("upload_begin","\"bytes\":123",true);
   w.append("evaluation_complete","\"evaluation\":0");w.finish();}
  auto x=read(base+"-bounded");assert(x.find("detail_limit")!=std::string::npos);
  assert(x.find("evaluation_complete")!=std::string::npos && x.find("run_complete")!=std::string::npos);
  struct stat st{};assert(!stat((base+"-bounded").c_str(),&st));assert(st.st_size<=4096 && (st.st_mode&0777)==0600);
  bool refused=false;try{pmi::ProgressWriter w(base+"-bounded");}catch(...){refused=true;}assert(refused);
  assert(!symlink((base+"-bounded").c_str(),(base+"-symlink").c_str()));
  refused=false;try{pmi::ProgressWriter w(base+"-symlink");}catch(...){refused=true;}assert(refused);
  for(int sig:{SIGTERM,SIGKILL}) {
    std::string p=base+"-signal"+std::to_string(sig);int pipefd[2];assert(!pipe(pipefd));auto pid=fork();assert(pid>=0);
    if(!pid){close(pipefd[0]);pmi::ProgressWriter w(p);w.append("evaluation_begin","\"evaluation\":0");w.append("upload_begin");char c=1;assert(write(pipefd[1],&c,1)==1);for(;;)pause();}
    close(pipefd[1]);char c;assert(::read(pipefd[0],&c,1)==1);close(pipefd[0]);assert(!kill(pid,sig));int status;assert(waitpid(pid,&status,0)==pid && WIFSIGNALED(status));
    x=read(p);assert(x.find("evaluation_begin")!=std::string::npos && x.find("upload_begin")!=std::string::npos && x.find("run_complete")==std::string::npos && x.back()=='\n');
  }
  {
    auto pid=fork();assert(pid>=0);
    if(!pid){
      signal(SIGXFSZ,SIG_IGN);pmi::ProgressWriter w(base+"-io-failure");
      rlimit limit{64,64};assert(!setrlimit(RLIMIT_FSIZE,&limit));
      try{w.append("upload_begin");}catch(...){_exit(0);}_exit(1);
    }
    int status;assert(waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0);
    x=read(base+"-io-failure");assert(x.find("run_complete")==std::string::npos);
  }
  {pmi::ProgressWriter w(base+"-threads");std::vector<std::thread> ts;
   for(int i=0;i<4;++i)ts.emplace_back([&]{for(int j=0;j<50;++j)w.append("worker","\"bytes\":1");});
   for(auto& t:ts)t.join();w.finish();}
  x=read(base+"-threads");size_t n=0;for(char c:x)if(c=='\n')++n;assert(n==202);
}
