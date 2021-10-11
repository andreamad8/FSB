# Few-Shot Bot: Prompt-Based Learning for Dialogue Systems

This repository includes the dataset and baselines of the paper:

**Few-Shot Bot: Prompt-Based Learning for Dialogue Systems** [PDF]](https://arxiv.org/abs/2012.15504). 

**Authors**: [Andrea Madotto](https://andreamad8.github.io), [Zhaojiang Lin](https://zlinao.github.io), [Genta Indra Winata](https://gentawinata.com/), [Pascale Fung](https://pascale.home.ece.ust.hk/)


## Abstract
Learning to converse using only a few examples is a grand challenge in Conversational AI. The current best conversational models, which are either good chit-chatters (e.g., BlenderBot) or goal-oriented systems (e.g., MinTL), are language models (LMs) fine-tuned on large conversational datasets. Training these models is expensive, both in terms of computational resources and time, and it is hard to keep these models up to date with new conversational skills. A simple yet unexplored solution is prompt-based few-shot learning~\citep{brown2020language} which does not require gradient-based fine-tuning but instead uses a few examples in the LM context as the only source of learning. In this paper, we explore prompt-based few-shot learning in dialogue tasks. We benchmark LMs of different sizes in 9 response generation tasks, which include a variety of knowledge-grounded tasks, task-oriented generations, general open-chat, and controlled stylistic generation, and 5 conversational parsing tasks, which include dialogue state tracking, graph path generation, persona information extraction, and document retrieval. The current largest, released, LM (GPT-J-6B) achieves competitive performance to full-training state-of-the-art models by using the prompt-based few-shot learning, thus \textit{no training}. Moreover, we proposed a novel perplexity-based classifier, that also does not require any fine-tuning, to select the most appropriate prompt given a dialogue history, as to create an all-in-one model with multiple dialogue skills. Finally, by combining the power of prompt-based few-shot learning and the skill selector, we create an end-to-end chatbot named the \textbf{Few-Shot Bot}, which automatically selects the most appropriate conversational skill, queries different KBs or the internet, and uses it to generate a human-like response, all by using only one dialogue example per skill.   

## Installation
In this repo, we load all the validation and test sets used in the evaluation. For running the experiments and the demo, you should install the following requirements:
```
pip install -r requirements.txt
```

## Basic Running

### Reproducing the results and plots
The ```generation``` folder stores the generated responses of the experiments in all datasets. To generate the tables and the plots in the paper, run 
```
python generate_plots_tables.py
```
This script loads all the files and computes the mean between different runs and it generates the plots. Note that this script is very custum for each datasets, but it can serve as guide line for future extentions. 


