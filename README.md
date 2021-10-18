# Few-Shot Bot: Prompt-Based Learning for Dialogue Systems

This repository includes the dataset, experiments results, and code for the paper:

**Few-Shot Bot: Prompt-Based Learning for Dialogue Systems** [PDF](https://arxiv.org/abs/2110.08118). 

**Authors**: [Andrea Madotto](https://andreamad8.github.io), [Zhaojiang Lin](https://zlinao.github.io), [Genta Indra Winata](https://gentawinata.com/), [Pascale Fung](https://pascale.home.ece.ust.hk/)


## Abstract
Learning to converse using only a few examples is a great challenge in conversational AI. The current best conversational models, which are either good chit-chatters (e.g., BlenderBot) or goal-oriented systems (e.g., MinTL), are language models (LMs) fine-tuned on large conversational datasets. Training these models is expensive, both in terms of computational resources and time, and it is hard to keep them up to date with new conversational skills. A simple yet unexplored solution is prompt-based few-shot learning (Brown et al. 2020) which does not require gradient-based fine-tuning but instead uses a few examples in the LM context as the only source of learning. In this paper, we explore prompt-based few-shot learning in dialogue tasks. We benchmark LMs of different sizes in nine response generation tasks, which include four knowledge-grounded tasks, a task-oriented generations task, three open-chat tasks, and controlled stylistic generation, and five conversational parsing tasks, which include dialogue state tracking, graph path generation, persona information extraction, document retrieval, and internet query generation. The current largest released LM (GPT-J-6B) using prompt-based few-shot learning, and thus requiring no training, achieves competitive performance to fully trained state-of-the-art models. Moreover, we propose a novel prompt-based few-shot classifier, that also does not require any fine-tuning, to select the most appropriate prompt given a dialogue history. Finally, by combining the power of prompt-based few-shot learning and a Skill Selector, we create an end-to-end chatbot named the Few-Shot Bot (FSB), which automatically selects the most appropriate conversational skill, queries different knowledge bases or the internet, and uses the retrieved knowledge to generate a human-like response, all using only few dialogue examples per skill.

## Installation
In this repo, we load all the validation and test sets used in the evaluation. For running the experiments and the demo, you should install the following requirements:
```
pip install -r requirements.txt
```

## Basic Running

### Reproducing the results and plots
The ```generation``` folder stores the generated responses of the experiments in all datasets. To generate the tables and the plots in the paper, run:
```
python generate_plots_tables.py
```
This script loads all the files computes the mean between different runs and generates the plots. Note that this script is very custom for each dataset, but it can serve as a guideline for future extensions. 


### Running the experiments
There are three main files to run 1) response generation (```main_response_generation.py```), 2) conversational parsing (```main_conversational_parsing.py```), and 3) skill-selector (```main_skill_selector.py```). In these files, we load the necessary prompt (```load_prefix```), and we run the generation (```generate_response```) for each sample in the test set. Since each dialogue skill require a different template, as shown in the paper, we create a function that converts structured data into the correct shot prompt. An example of this function can be found in ```prompts/persona_chat.py```, and in ```generic_prompts.py``` we store the generation functions. 

In each main file, there is a configuration object (```mapper```) that specifies meta-information about the task (i.e., number of shots, generation length, decoding type, prompt converter). Especially for conversational parsing, there are different decoding types. For example, in MWOZ, the model generates the dialogue state, which is further looped into the next turn. 


#### How to run?
For example, to run the persona chat experiments (0, 1, k-shots), you can use the following command:
```
python main_response_generation.py --model_checkpoint EleutherAI/gpt-j-6B --dataset persona --gpu 0
```
In case your GPU has less that 16GB, then you could add ```--multigpu``` to spawn 4 GPUs (e.g., 1080Ti) and do inference in parallel. Similarly, for conversational parsing tasks, you could use:
```
python main_conversational_parsing.py --model_checkpoint EleutherAI/gpt-j-6B --dataset wow-parse --gpu 0
```
Notice that some parsing task requires a knowledge base (e.g., dialKG-parse requires the KG in neo4j). 
Finally, to run the skill-selector task, you could use:
```
python main_skill_selector.py --model_checkpoint EleutherAI/gpt-j-6B --shots_k 6 --repetition 1 --gpu 0
```
where repetition is the seed for selecting random samples in the prompts. 

#### Runners
In the ```runners``` folder, we provide a rudimental runner to run all the experiments and reproduce the results in the paper. 

## Few-Shot Bot
There are two FSB modes: 1) controlled style generation (FSB-CG) and 2) full-model. 

### FSB-CG 
Check the ```FSB-CG.ipynb``` to try to interact with FSB in your local machine, or try directly in colab at 
```
https://colab.research.google.com/drive/15hQv1V3Cs5kQVfLOE_FZc1VCWQ3YpWVd?usp=sharing
```
Remember to select the environment with GPU!! 

### FSB 
Check the ```FSB.ipynb``` to try to interact with FSB in your local machine, or try directly in colab at 
```
https://colab.research.google.com/drive/1JkPeX-6oXikiwWKqW5QkibEy8Oq6KM9g?usp=sharing
```
Remember to select the environment with GPU!! This current version does not query the Internet, Wiki and KGs, but only parse the dialogue history with MSC-parse. We implement only 4 skills for now. 

## Safety Bench
TODO