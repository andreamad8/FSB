import streamlit as st
import pandas as pd
import json
import random
import streamlit.components.v1 as components
import SessionState
from dialogue_helper import header, footer, render, display_app_header
from prompts.generic_prompt import load_prefix, generate_response_interactive, select_prompt_interactive
from prompts.generic_prompt_parser import load_prefix as load_prefix_parse
from prompts.skill_selector import convert_sample_to_shot_selector
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts.semantic_parser import convert_sample_to_shot_semantic_parser
from sentence_transformers import SentenceTransformer, util
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
    
    # load index of prompts from TOP
    
    embedder = SentenceTransformer('all-mpnet-base-v2')
    with open("../data/TOP/train.json") as json_file:
        sub_data_dialogue_raw = json.load(json_file)
    # sub_data_dialogue_raw = random.sample(sub_data_dialogue_raw, int(len(sub_data_dialogue_raw)*0.01))
    data_all = [d['dialogue'] for d in sub_data_dialogue_raw]
    print(len(data_all))
    corpus_embeddings = embedder.encode(data_all, convert_to_tensor=True)

    return model, tokenizer, max_seq, corpus_embeddings, embedder, sub_data_dialogue_raw


def retrive_closest_shots(corpus_embeddings, embedder, sub_data_dialogue_raw, dialogue, n_shots):
    query_embedding = embedder.encode([dialogue], convert_to_tensor=True)
    
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=n_shots)
    hits = hits[0]      #Get the hits for the first query
    top_shots = []
    for hit in hits:
        top_shots.append({"score":hit['score'],
                            "dialogue":sub_data_dialogue_raw[hit['corpus_id']]['dialogue'],
                            "query":sub_data_dialogue_raw[hit['corpus_id']]['query']})
    return top_shots


def get_prompt(top_shots,shot_converter):
    prompt = ""
    for shot in top_shots:
        prompt += shot_converter(sample=shot) + "\n\n"
    return prompt

def get_session_state():
    session_state = SessionState.get(sessionstep = 0, dialogue =[], prompt=[],
                                     temperature='', topk='', topp='', api_key='', api=False)
    return session_state


def mychat(): 
    args = type('', (), {})()
    args.multigpu = False
    device = 0
    safety_level = 6
    shot_selector = 6
    sample_skill = False
    model_checkpoint = "gpt2"
    # model_checkpoint = "EleutherAI/gpt-neo-1.3B"
    # model_checkpoint = "EleutherAI/gpt-j-6B"
    dialogue_ss = get_session_state()
    

    form_model = st.sidebar.form(key='my_form')
    api_key = form_model.text_input("Insert an API key from AI21 to interact with the 175B model")
    submit_button_model = form_model.form_submit_button(label='Submit')
    if submit_button_model:
        dialogue_ss.api = True
        dialogue_ss.api_key = api_key
        st.sidebar.write("You are using the AI21 API")   
    
    # dialogue_ss.length_gen = st.sidebar.slider("Max Length", value=50, min_value = 10, max_value=100)
    temperature = st.sidebar.slider("Temperature", value = 1.0, min_value = 0.0, max_value=1.0, step=0.05)
    topp = st.sidebar.slider("Top-p", min_value = 0.0, max_value=1.0, step = 0.05, value = 0.9)
    shots_number = st.sidebar.slider("N-Shots", min_value = 1, max_value=30, step = 1, value = 10)


    with st.spinner("Initial models loading, please be patient"):
        model, tokenizer, max_seq, corpus_embeddings, embedder, sub_data_dialogue_raw = load_model(args, model_checkpoint, device, shot_selector, safety_level) #  bad_word_ids

    chatlogholder = st.empty()
    with chatlogholder:
        if len(dialogue_ss.dialogue)==0:
            components.html(header+footer, height=400)
        else:
            components.html(render(dialogue_ss.dialogue, None), height=400, scrolling=True)

    form = st.form(key='chatinput', clear_on_submit=True)
    example_set = sub_data_dialogue_raw[random.randint(0,10)]['dialogue']
    chatinput = st.selectbox('Select a query for the model',('Email', 'Home phone', 'Mobile phone'))

    # For example: {sub_data_dialogue_raw[random.randint(0,1000)]['dialogue']}
    # chatinput = form.text_input("", placeholder=f"Type a command..." , key='chatinput')
    submit = form.form_submit_button('Send')
    try:
        if submit:
            dialogue_ss.sessionstep += 1
            dialogue_ss.dialogue.append([chatinput,""])

            # prepare the input for the model
            dialogue = {"dialogue": chatinput, "query":""}
            print(chatinput)

            with chatlogholder:
                components.html(render(dialogue_ss.dialogue, None), height=400, scrolling=True)

            # retrive closest shots from the training set
            top_shots = retrive_closest_shots(corpus_embeddings, embedder, sub_data_dialogue_raw, chatinput, n_shots=int(shots_number))
            print(dialogue)


            ## generate response based on skills
            prompt = get_prompt(top_shots,convert_sample_to_shot_semantic_parser)
            print(dialogue)
            response = generate_response_interactive(model, tokenizer, shot_converter=convert_sample_to_shot_semantic_parser, 
                                                        dialogue=dialogue, prefix=prompt, 
                                                        device=device, with_knowledge=None, 
                                                        meta_type=None, gen_len=50, 
                                                        beam=1, max_seq=max_seq, eos_token_id=198, 
                                                        do_sample=True, multigpu=False, api=dialogue_ss.api, api_key=dialogue_ss.api_key,
                                                        temperature=temperature, topp=topp)


            print(dialogue)
            print(convert_sample_to_shot_semantic_parser(dialogue))
            print(f"FSB >>> {response}")
            dialogue_ss.prompt.append(prompt + f'*{convert_sample_to_shot_semantic_parser(dialogue)}*' + f" ***{response}***")
            # dialogue_ss.dialogue[-1][1] = response
            with chatlogholder:
                components.html(render(dialogue_ss.dialogue, {"query":response,"wiki":[]}), height=400, scrolling=True)

            with st.expander("Prompt"):
                for i_p, prompt in enumerate(dialogue_ss.prompt):
                    st.markdown(f"### Request {i_p}")
                    st.markdown(prompt.replace("\n", "<br>"), unsafe_allow_html=True)
       
    except:
        raise


def main():
    
    main_txt = """Few-Shot Bot: Prompt-Based Learning for Dialogue Systems"""
    sub_txt = "Conversational Semantic Parsing"
    subtitle = """FSB demo for Conversational Semantic Parsing (CSP). In this case we use TOP dataset.  
                    Firstly, the retriever model gets the closest examples from the training set and then use it as prompt to parse the user's input. 
        """

    html_temp = f"""
    <div style = " padding:10px">
    <h2 style = "color:#3c403f ; text_align:center;"> {main_txt} </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html = True)
    st.markdown(subtitle)
    st.sidebar.markdown(f'## Generation Settings')

    # st.sidebar.markdown("""TEST TEST""")
    mychat()

if __name__ == "__main__":
    main()