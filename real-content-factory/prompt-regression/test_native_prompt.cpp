#include "chat.h"
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
    std::string prompt;
    {
        if (params.conversation_mode && params.enable_chat_template) {
            if (!params.system_prompt.empty()) {
                // Collect initial roles before rendering: templates may require a user.
                common_chat_msg message;
                message.role = "system";
                message.content = params.system_prompt;
                chat_msgs.push_back(message);
            }

            if (!params.prompt.empty()) {
                // Add the exact CLI user prompt to the initial conversation.
                common_chat_msg message;
                message.role = "user";
                message.content = params.prompt;
                chat_msgs.push_back(message);
            } else {
                waiting_for_first_input = true;
            }

            if (!params.system_prompt.empty() || !params.prompt.empty()) {
                common_chat_templates_inputs inputs;
                inputs.use_jinja = g_params->use_jinja;
                inputs.messages = chat_msgs;
                inputs.enable_thinking = g_params->enable_reasoning != 0;
                inputs.add_generation_prompt = !params.prompt.empty();
                inputs.force_pure_content = params.force_pure_content_parser;

                prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            }
        } else {
            // otherwise use the prompt as is
            prompt = params.prompt;
        }

    }
return prompt;
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
