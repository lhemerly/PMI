"""Native Jinja model-free regression using the exact completion initial-render block."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path('/home/luizh/Documents/Codex/2026-10-08/task-8/real-content-factory')
SRC=Path('/home/luizh/Documents/Codex/2026-10-06/task-3/smaller-conversation-prep-v1/source/prism/PrismML-Eng-llama.cpp-adfffbe')
sys.path.insert(0,str(ROOT));from prompt_fit import SYSTEM_PROMPT,BONSAI_TEMPLATE_SHA256
source=(SRC/'tools/completion/completion.cpp').read_text()
block=source[source.index('    std::string prompt;\n    {',source.index('auto chat_add_and_format')):source.index('        if (params.interactive_first',source.index('auto chat_add_and_format'))]+'    }\n'
assert 'chat_add_and_format(' not in block
assert 'inputs.enable_thinking = g_params->enable_reasoning != 0;' in block
header='''#include "chat.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
std::string read(const char * p) {std::ifstream f(p);std::ostringstream s;s<<f.rdbuf();return s.str();}
std::string initial(const std::string & t, const std::string & user, const std::string & system, int reasoning) {
 auto chat_templates=common_chat_templates_init(nullptr,t);
 std::vector<common_chat_msg> chat_msgs;
 struct Params {bool conversation_mode=true,enable_chat_template=true,use_jinja=true,force_pure_content_parser=false;int enable_reasoning=0;std::string prompt,system_prompt;} params;
 params.prompt=user;params.system_prompt=system;params.enable_reasoning=reasoning;auto * g_params=&params;
 bool waiting_for_first_input=false;
'''
tail='''return prompt;
}
int main(int argc,char **argv) {
 if(argc!=4)return 2;
 auto t=read(argv[1]),user=read(argv[2]),expected=read(argv[3]);
 auto tmpls=common_chat_templates_init(nullptr,t);
 common_chat_msg system;system.role="system";system.content="Answer concisely in plain text.";
 bool failed=false;
 try {common_chat_format_single(tmpls.get(),{},system,false,true,false);}catch(const std::exception &e){failed=std::string(e.what()).find("No user query found in messages.")!=std::string::npos;}
 if(!failed)throw std::runtime_error("original system-only failure not reproduced");
 auto actual=initial(t,user,system.content,0);
 if(actual!=expected)throw std::runtime_error("native and fit renderer differ");
 if(initial(t,user,system.content,1)==expected)throw std::runtime_error("reasoning flag ignored");
 auto plain=initial(t,"A different arbitrary user query.","",0);
 if(plain.find("A different arbitrary user query.")==std::string::npos)throw std::runtime_error("arbitrary user missing");
 std::cout<<actual;return 0;
}
'''
Path('test_native_prompt.cpp').write_text(header+block+tail)
template=Path('/home/luizh/Documents/Codex/2026-10-06/task-3/smaller-conversation-prep-v1/templates/bonsai2-27b-official.jinja')
raw=template.read_bytes();assert hashlib.sha256(raw).hexdigest()==BONSAI_TEMPLATE_SHA256
from jinja2 import Environment,StrictUndefined
expected=Environment(undefined=StrictUndefined).from_string(raw.decode()).render(messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':(ROOT/'first-run-prompt.txt').read_text()}],tools=None,add_generation_prompt=True,enable_thinking=False)
Path('expected-native-welcome-prompt.txt').write_text(expected)
bin_dir=ROOT/'.runtime/prism-vulkan/bin'
cmd=['/usr/bin/c++','-std=c++17','-O0','-I'+str(SRC/'common'),'-I'+str(SRC/'include'),'-I'+str(SRC/'ggml/include'),'-I'+str(SRC/'vendor'),'test_native_prompt.cpp','-L'+str(bin_dir),'-Wl,-rpath,'+str(bin_dir),'-lllama-common','-lllama','-lggml','-lggml-base','-o','test_native_prompt']
subprocess.run(cmd,check=True,timeout=90)
r=subprocess.run([str(Path('test_native_prompt').resolve()),str(template),str(ROOT/'first-run-prompt.txt'),str(Path('expected-native-welcome-prompt.txt').resolve())],capture_output=True,timeout=15)
Path('native-prompt-test.stderr').write_bytes(r.stderr)
if r.returncode:raise RuntimeError(str(r.returncode)+': '+r.stderr.decode())
assert r.stdout==expected.encode();Path('actual-native-welcome-prompt.txt').write_bytes(r.stdout)
record={'status':'NativeModelFreePromptRegressionPassed','checks':['Original system-only native Jinja exception reproduced','Exact production initial-render block tested','Both initial roles rendered together','Native output byte-identical to pinned Python fit renderer','Reasoning off honored; on produces different prefix','Different arbitrary user with no system renders'], 'model_loaded':False,'backend_init_called':False,'device_workload':False,'template_sha256':BONSAI_TEMPLATE_SHA256,'completion_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'prompt_bytes':len(r.stdout),'prompt_sha256':hashlib.sha256(r.stdout).hexdigest(),'input_upper_bound':len(r.stdout)+2,'context':512,'output_reserve':64,'system':SYSTEM_PROMPT,'user':(ROOT/'first-run-prompt.txt').read_text(),'final_prompt':expected}
Path('native-prompt-regression.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
