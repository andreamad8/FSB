from prompts.persona_chat import convert_sample_to_shot_persona
from prompts.persona_chat_memory import convert_sample_to_shot_msc, convert_sample_to_shot_msc_interact
from prompts.persona_parser import convert_sample_to_shot_msc as convert_sample_to_shot_msc_parse
from prompts.emphatetic_dialogue import convert_sample_to_shot_ed
from prompts.daily_dialogue import convert_sample_to_shot_DD_prefix, convert_sample_to_shot_DD_inference
from prompts.skill_selector import convert_sample_to_shot_selector
from prompts.generic_prompt import load_prefix, generate_response_interactive, select_prompt_interactive
from prompts.wizard_of_wikipedia import convert_sample_to_shot_wow, convert_sample_to_shot_wow_interact
from prompts.wizard_of_wikipedia_parse import convert_sample_to_shot_wow as convert_sample_to_shot_wow_parse
import wikipedia
import html
import random

## This is the config dictionary used to select the template converter
mapper = {
          "persona": {"shot_converter":convert_sample_to_shot_persona, 
                    "shot_converter_inference": convert_sample_to_shot_persona,
                     "file_data":"../data/persona/","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5,6]},"max_shot":{1024:2,2048:6},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
        #   "msc": {"shot_converter":convert_sample_to_shot_msc, 
        #             "shot_converter_inference": convert_sample_to_shot_msc_interact,
        #              "file_data":"../data/msc/session-2-","with_knowledge":None,
        #              "shots":{1024:[0,1],2048:[0,1,3]},"max_shot":{1024:1,2048:3},
        #              "shot_separator":"\n\n",
        #              "meta_type":"all","gen_len":50,"max_number_turns":3},
          "ed": {"shot_converter":convert_sample_to_shot_ed, 
                 "shot_converter_inference": convert_sample_to_shot_ed,
                 "file_data":"../data/ed/","with_knowledge":None,
                  "shots":{1024:[0,1,7],2048:[0,1,17]},"max_shot":{1024:7,2048:17},
                  "shot_separator":"\n\n",
                  "meta_type":"none","gen_len":50,"max_number_turns":5},
          "DD": {"shot_converter":convert_sample_to_shot_DD_prefix, 
                 "shot_converter_inference": convert_sample_to_shot_DD_inference,
                 "file_data":"../data/dailydialog/","with_knowledge":False,
                  "shots":{1024:[0,1,2],2048:[0,1,6]},"max_shot":{1024:2,2048:6},
                  "shot_separator":"\n\n",
                  "meta_type":"all_turns","gen_len":50,"max_number_turns":5},
        #   "msc-parse": {"shot_converter":convert_sample_to_shot_msc_parse, "max_shot":{1024:1,2048:2},
        #          "file_data":"../data/msc/parse-session-1-","level":"dialogue", "retriever":"none",
        #           "shots":{1024:[0,1],2048:[0, 1, 2]},"shot_separator":"\n\n",
        #           "meta_type":"incremental","gen_len":50,"max_number_turns":3}, 
           "safe": {"shot_converter":convert_sample_to_shot_persona, 
                 "shot_converter_inference": convert_sample_to_shot_persona,
                 "file_data":"../data/safety_layers/safety_safe_adv_","with_knowledge":None,
                  "shots":{1024:[0,1,5],2048:[0,1,10]},"max_shot":{1024:5,2048:10},
                  "shot_separator":"\n\n",
                  "meta_type":"none","gen_len":50,"max_number_turns":5},
          "wow": {"shot_converter":convert_sample_to_shot_wow, 
                 "shot_converter_inference": convert_sample_to_shot_wow_interact,
                 "file_data":"../data/wow/","with_knowledge":True,
                  "shots":{1024:[0,1,2],2048:[4,3,2,1,0]},"max_shot":{1024:1,2048:1},
                  "shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":60,"max_number_turns":5},
          "wow-parse": {"shot_converter":convert_sample_to_shot_wow_parse, 
                 "file_data":"../data/wow/parse-","level":"dialogue", "retriever":"wiki",
                  "shots":{1024:[0, 1, 5],2048:[0, 1, 5, 10]},"max_shot":{1024:5,2048:10},
                  "shot_separator":"\n\n", "meta_type":"sentence","gen_len":50,"max_number_turns":2},
         }
         
## This is the config dictionary used to select the template converter
mapper_safety = {
          "unsa_topic": {"file_data":"../data/safety_layers/safety_topic.json","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":2},
          "unsa_nonadv": {"file_data":"../data/safety_layers/safety_nonadv.json","with_knowledge":None,
                     "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
                     "shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":2},
        ## THIS MAKE IT VERY VERY SAFE
        #   "unsa_adv": {"file_data":"../data/safety_layers/safety_adv.json","with_knowledge":None,
        #              "shots":{1024:[0,1,2],2048:[0,1,2,3,4,5]},"max_shot":{1024:2,2048:3},
        #              "shot_separator":"\n\n",
        #              "meta_type":"all","gen_len":50,"max_number_turns":2},
         }

def run_parsers(args, model, tokenizer, device, max_seq, dialogue, skill, prefix_dict, api, api_key):

    if skill not in ["wow"]: return dialogue

    ### parse 
    d_p = f"{skill}-parse"
    print(f"Parse with {d_p}")

    prefix = prefix_dict[d_p].get(mapper[d_p]["max_shot"][max_seq])
    query = generate_response_interactive(model, tokenizer, shot_converter=mapper[d_p]["shot_converter"], 
                                                    dialogue=dialogue, prefix=prefix, 
                                                    device=device,  with_knowledge=None, 
                                                    meta_type=None, gen_len=50, 
                                                    beam=1, max_seq=max_seq, eos_token_id=198, 
                                                    do_sample=False, multigpu=False, api=api, api_key=api_key)

    
    print(f"Query: {query}")
    if query.lower() == "none": return dialogue
    dialogue["query"] = query

    if skill == "wow" and query not in dialogue["query_mem"]:
        dialogue["query_mem"].append(query)
        ## Try first with Wiki
        retrieve_K = "None"
        try:
            retrieve_K = wikipedia.summary(query, sentences=1, auto_suggest=False)
            print(f"Retrieved: {retrieve_K}")
        except wikipedia.DisambiguationError as e:
            s = random.choice(e.options)
            print("New query: ", s)
            try:
                retrieve_K = wikipedia.summary(s, sentences=1, auto_suggest=False)
            except:
                print("Error retrieving wikipedia")
                retrieve_K = "None"
        except:
            print("Error retrieving wikipedia")
            retrieve_K = "None"
        dialogue["KB_wiki"][-1] = [retrieve_K]
    elif skill == "msc":
        dialogue["user"].append(query)
        dialogue["user_memory"][-1] = [query]
    return dialogue




header =  """
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
                        <link rel="stylesheet" href="https://unpkg.com/botui@0.3.9/build/botui.min.css" />
                        <link rel="stylesheet" href="https://unpkg.com/botui@0.3.9/build/botui-theme-default.css" />
                        <style>
                        .botui-container {
                            background-color: inherit;
                        }

                        .botui-messages-container {
                        }

                        .botui-actions-container {
                            padding: 0px 30px;
                        }

                        .botui-message {
                        }

                        .botui-message-content.text {
                            background-color: #e1fafc;
                            color: #404040;
                        }

                        .botui-message-content.human {
                            background-color: #f7ff61;
                            color: #404040;
                        }

                        .botui-message-content.embed {
                        }

                        .botui-message-content-link {
                        }

                        .botui-actions-text-input {
                        }

                        .botui-actions-text-submit {
                        }

                        button.botui-actions-buttons-button {
                            margin-top: 0px;
                        }
                        </style>
                    </head>
                    <body>
                        <div class="botui-app-container" id="hello-world">
                        <bot-ui></bot-ui>
                        </div>
                        <script src="https://cdn.jsdelivr.net/vue/latest/vue.min.js"></script>
                        <script src="https://cdn.jsdelivr.net/gh/andreamad8/Abotui/botui1.js"></script>
                        
                        <script>
                        var botui = new BotUI('hello-world');
        """

footer = """
        </script>
    </body>
    </html>
    """

def render(list_turn, query_info):
    # print(query_info)
    # means no query has been run
    q = None
    if query_info: 
        if query_info and query_info["query"] == "": 
            q = None
        elif len(query_info["wiki"]):
            kb = query_info["wiki"][0]
            q = query_info["query"]
        else:
            q = query_info["query"]
            kb = None
             
    string = header 
    for id_t, turn in enumerate(list_turn):
        u, s =  html.escape(turn[0]), html.escape(turn[1])
        if id_t == 0 and s == '':
            # print("""botui.message.add({human: true,content: '""" +list_turn[i]+"'})")
            string += """botui.message.add({human: true, content: '""" +u+"'})"
            if q:
                query = f"Query: {q}" 
                string += """.then(function () {return botui.message.add({type: 'html', content: '""" + html.escape(query)+"""'});})"""
        elif id_t == 0 and s != '':
            string += """botui.message.add({human: true, content: '""" +u+"'})"
            if id_t == len(list_turn)-1 and q:

                query = f"Query: {q}" 
                string += """.then(function () {return botui.message.add({type: 'html', content: '""" + html.escape(query)+"""'});})"""
                if kb: 
                    string += """.then(function () {return botui.message.add({type: 'html', content: '""" + html.escape(f"KB: {kb}")+"""'});})"""
            string += """.then(function () {return botui.message.add({content: '""" +s+"""'});})"""
        else:
            string += """.then(function () {return botui.message.add({human: true, content: '""" +u+"""'});})"""
            if id_t == len(list_turn)-1 and q:
                query = f"Query: {q}" 
                string += """.then(function () {return botui.message.add({type: 'html', content: '""" + html.escape(query)+"""'});})"""
                if kb: 
                    string += """.then(function () {return botui.message.add({type: 'html', content: '""" + html.escape(f"KB: {kb}")+"""'});})"""
            if s != '':
                string += """.then(function () {return botui.message.add({content: '""" +s+"""'});})"""

    string += ";"
    string += footer
    return string
# <script src="https://cdn.jsdelivr.net/npm/botui/build/botui.js"></script>

# botui.message.add({human: true, content: 'Hi! Welcome to my website'})
# .then(function () {return botui.message.add({content: 'How can I help?'});})
# .then(function () {return botui.message.add({human: true, content: 'Hi! Welcome to my website'});})
# .then(function () {return botui.message.add({content: 'How can I help?'});})
# .then(function () {return botui.message.add({human: true, content: 'Hi! Welcome to my website'});})
# .then(function () {return botui.message.add({content: 'How can I help?'});});


def display_app_header(main_txt,sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    Parameters
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <div style = "background.color:#3c403f  ; padding:15px">
    <h2 style = "color:black; text_align:center;"> {main_txt} </h2>
    <p style = "color:black; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)

