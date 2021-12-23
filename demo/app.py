import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import SessionState
from dialogue_helper import header, footer, render, mapper, mapper_safety, run_parsers
from prompts.generic_prompt import load_prefix, generate_response_interactive, select_prompt_interactive
from prompts.generic_prompt_parser import load_prefix as load_prefix_parse
from prompts.skill_selector import convert_sample_to_shot_selector
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os


st.set_page_config(page_title="Few-Shot Bot", layout='centered', initial_sidebar_state='auto', page_icon="🤖")


@st.cache(allow_output_mutation=True, max_entries=1) #ttl=1200,
def load_model(args, model_checkpoint, device, shot_selector, safety_level):
    if "gpt-j"in model_checkpoint or "neo"in model_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if args.multigpu:
            from parallelformers import parallelize
            parallelize(model, num_gpus=4, fp16=True, verbose='detail')
        else:
            model.half().to(device)
        max_seq = 2048
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        tokenizer.bos_token = ":"
        tokenizer.eos_token = "\n"
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        max_seq = 1024
        model.half().to(device)
    
    available_datasets = mapper.keys()
    prompt_dict = {}
    prompt_parse = {}
    prompt_skill_selector = {}
    for d in available_datasets:
        if "parse" in d:
            prompt_parse[d] = load_prefix_parse(tokenizer=tokenizer, shots_value=mapper[d]["shots"][max_seq], 
                                    shot_converter=mapper[d]["shot_converter"], 
                                    file_shot=mapper[d]["file_data"]+"valid.json", 
                                    name_dataset=d, level=mapper[d]["level"], 
                                    shot_separator=mapper[d]["shot_separator"],sample_times=1)[0]
        else:
            if "safe" != d:
                prompt_skill_selector[d] = load_prefix(tokenizer=tokenizer, shots_value=[shot_selector], 
                            shot_converter=convert_sample_to_shot_selector, 
                            file_shot= mapper[d]["file_data"]+"train.json" if "smd" in d else mapper[d]["file_data"]+"valid.json", 
                            name_dataset=d, with_knowledge=None, 
                            shot_separator=mapper[d]["shot_separator"],sample_times=1)[0]
            prompt_dict[d] = load_prefix(tokenizer=tokenizer, shots_value=mapper[d]["shots"][max_seq], 
                        shot_converter=mapper[d]["shot_converter"], 
                        file_shot=mapper[d]["file_data"]+"valid.json", 
                        name_dataset=d, with_knowledge=mapper[d]["with_knowledge"], 
                        shot_separator=mapper[d]["shot_separator"],sample_times=1)[0]
        
    ## add safety prompts
    for d in mapper_safety.keys():
        prompt_skill_selector[d] = load_prefix(tokenizer=tokenizer, shots_value=[safety_level], 
                shot_converter=convert_sample_to_shot_selector, 
                file_shot= mapper_safety[d]["file_data"], 
                name_dataset=d, with_knowledge=None, 
                shot_separator=mapper_safety[d]["shot_separator"],sample_times=1)[0]

    return model, tokenizer, max_seq, prompt_dict, prompt_parse, prompt_skill_selector



def get_session_state():
    session_state = SessionState.get(sessionstep = 0, dialogue =[],
                                     meta = [], user=[], assistant=[],
                                     user_memory=[], length_gen='', KB_wiki=[], query_mem=[],
                                     prompt=[], skill=[],
                                     temperature='', topk='', topp='', api_key='', api=False)
    return session_state


def mychat(): 
    args = type('', (), {})()
    args.multigpu = False
    device = 0
    safety_level = 6
    shot_selector = 6
    sample_skill = False
    # model_checkpoint = "gpt2"
    # model_checkpoint = "EleutherAI/gpt-neo-1.3B"
    model_checkpoint = "EleutherAI/gpt-j-6B"
    dialogue_ss = get_session_state()
    

    form_model = st.sidebar.form(key='my_form')
    api_key = form_model.text_input("Insert an API key from AI21 to interact with the 175B model")
    submit_button_model = form_model.form_submit_button(label='Submit')
    if submit_button_model:
        dialogue_ss.api = True
        dialogue_ss.api_key = api_key
        st.sidebar.write("You are using the API")   

    max_number_turns = 3

    dialogue_ss.meta = [
                    "I am a chatbot.",
                    "My name is FSB.",
                    "I love chatting with people.",
                    "I am less than 1 years old."
                    ]
    persona_used = "#### Persona\n"+"<br>".join(dialogue_ss.meta)
    st.sidebar.markdown(f"{persona_used}", unsafe_allow_html=True)
    
    # dialogue_ss.length_gen = st.sidebar.slider("Max Length", value=50, min_value = 10, max_value=100)
    temperature = st.sidebar.slider("Temperature", value = 1.0, min_value = 0.0, max_value=1.0, step=0.05)
    topp = st.sidebar.slider("Top-p", min_value = 0.0, max_value=1.0, step = 0.05, value = 0.9)


    with st.spinner("Initial models loading, please be patient"):
        model, tokenizer, max_seq, prompt_dict, prompt_parse, prompt_skill_selector = load_model(args, model_checkpoint, device, shot_selector, safety_level) #  bad_word_ids

    chatlogholder = st.empty()
    with chatlogholder:
        if len(dialogue_ss.dialogue)==0:
            components.html(header+footer, height=400)
        else:
            components.html(render(dialogue_ss.dialogue, None), height=400, scrolling=True)

    form = st.form(key='chatinput', clear_on_submit=True)
    chatinput = form.text_input("", placeholder="Type a message...", key='chatinput')
    submit = form.form_submit_button('Send')
    try:
        if submit:
            print("API key:", dialogue_ss.api_key)
            dialogue_ss.user_memory.append([])
            dialogue_ss.KB_wiki.append([])
            dialogue_ss.sessionstep += 1
            dialogue_ss.dialogue.append([chatinput,""])

            # prepare the input for the model
            dialogue = {"dialogue":[],"meta":[],"user":[],
                        "assistant":[],"user_memory":[], 
                        "KB_wiki": [], "query_mem":[]}

            dialogue["dialogue"] = dialogue_ss.dialogue
            dialogue["meta"] = dialogue_ss.meta
            dialogue["assistant"] = dialogue_ss.meta
            dialogue["user"] = dialogue_ss.user
            dialogue["user_memory"] = dialogue_ss.user_memory
            dialogue["KB_wiki"] = dialogue_ss.KB_wiki
            dialogue["query"] = ""

            with chatlogholder:
                components.html(render(dialogue_ss.dialogue, None), height=400, scrolling=True)

            skill, skill_dist = select_prompt_interactive(model, tokenizer, 
                                            shot_converter=convert_sample_to_shot_selector, 
                                            dialogue=dialogue, prompt_dict=prompt_skill_selector, 
                                            device=device, max_seq=max_seq, max_shot=shot_selector)
            
            if "unsa" in skill: 
                skill = "safe"
                # print(f"FSB (Safety) >>> {response}")
            print(f"Skill: {skill}")
            print(skill_dist)
            ## parse user dialogue dialogue ==> msc-parse
            dialogue = run_parsers(args, model, tokenizer, device=device, max_seq=max_seq,
                                    dialogue=dialogue, skill=skill,  
                                    prefix_dict=prompt_parse, api=dialogue_ss.api, api_key=dialogue_ss.api_key)

            ## generate response based on skills
            prompt = prompt_dict[skill].get(mapper[skill]["max_shot"][max_seq])
            response = generate_response_interactive(model, tokenizer, shot_converter=mapper[skill]["shot_converter_inference"], 
                                                        dialogue=dialogue, prefix=prompt, 
                                                        device=device, with_knowledge=mapper[skill]["with_knowledge"], 
                                                        meta_type=mapper[skill]["meta_type"], gen_len=50, 
                                                        beam=1, max_seq=max_seq, eos_token_id=198, 
                                                        do_sample=True, multigpu=False, api=dialogue_ss.api, api_key=dialogue_ss.api_key,
                                                        temperature=temperature, topp=topp)


            print(f"FSB ({skill}) >>> {response}")
            dialogue_ss.skill.append(skill)
            dialogue_ss.prompt.append(prompt + f'*{mapper[skill]["shot_converter_inference"](dialogue)}*' + f" ***{response}***")
            dialogue_ss.dialogue[-1][1] = response
            dialogue_ss.dialogue = dialogue_ss.dialogue[-max_number_turns:]
            dialogue_ss.user_memory = dialogue["user_memory"][-max_number_turns:]
            dialogue_ss.KB_wiki = dialogue["KB_wiki"][-max_number_turns:]
            dialogue_ss.user = dialogue["user"]
            
            with chatlogholder:
                components.html(render(dialogue_ss.dialogue, {"query":dialogue["query"],"wiki":dialogue["KB_wiki"][-1]}), height=400, scrolling=True)

            with st.expander("Full chat dial"):
                for i_p, prompt in enumerate(dialogue_ss.prompt):
                    st.markdown(f"### Turn {i_p}: Prompt of the {dialogue_ss.skill[i_p]} skill.")
                    st.markdown(prompt.replace("\n", "<br>"), unsafe_allow_html=True)
       
    except:
        raise


def main():
    
    main_txt = """Welcome To Few-Shot Bot"""
    sub_txt = "Just have fun"
    subtitle = """**Instructions:** Type in some text and click "Chat" to generate a response. Optionally, adjust settings on the left.
        """

    # display_app_header(main_txt,sub_txt,is_sidebar = False)
    # st.markdown(subtitle)
    st.sidebar.markdown(f'## Generation Settings')

    # st.sidebar.markdown("""TEST TEST""")
    mychat()

if __name__ == "__main__":
    main()