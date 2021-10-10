import glob
import json
from collections import defaultdict
from tabulate import tabulate
import numpy as np
import pandas as pd
import matplotlib

# comment this line if you don't have a latex compiler 
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

tables_BLEU = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_B4 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_F1 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_KF1 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_RL = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_ppl = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_JGA = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_SACC = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_path1 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_path5 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_path25 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_tgt1 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_tgt5 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_tgt25 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_F1_smd = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_F1_navigate = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_F1_schedule = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_F1_weather = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_BLEU_smd = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_Rprec = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_accSKILL = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_F1SKILL = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
tables_FeQA = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

data_plot_ppl = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
folder_names = ["DD","dialKG-parse","IC","MSC","MWOZ",
                "skill-selector","WiT","WoW","dialKG",
                "ED","IC-IMG","MSC-parse","persona",
                "SMD","WiT-parse","WoW-parse"]

for folder_name in folder_names:
    for name in glob.glob(f'generations/{folder_name}/*.json'):
        # print(name)
        name_for_parsing = name
        if "msc_dialogue_2" in name_for_parsing:
            name_for_parsing = name.replace("msc_dialogue_2","msc-dialogue-2")
        if "generations/gpt" in name_for_parsing: 
            dataset = "Prompt Selection"
            parse = name_for_parsing.replace(f"generations/{folder_name}/","").replace(".json","").split("_")
            model, shot, id_exp = parse
            # print(model)
        else:
            parse = name_for_parsing.replace(f"generation/{folder_name}/","").replace(".json","").split("_")
            if len(parse) == 5:
                dataset, shot, model, beam, id_exp = parse
            else:
                dataset, shot, model, beam, id_exp, sample = parse

        # print(dataset, shot, model, beam, id_exp)
        res = json.load(open(name,"r"))

        if "B4" in res['score']:
            tables_B4[dataset][shot][model].append(res['score']["B4"])
            tables_F1[dataset][shot][model].append(res['score']["F1"])
            tables_RL[dataset][shot][model].append(res['score']["RL"])
            tables_ppl[dataset][shot][model].append(res['score']["ppl"])
            if "wit_" in name:
                tables_KF1[dataset][shot][model].append(res['score']["kf1"])
        if "mwoz-parse-dialogue" in name or "mwoz-DST" in name:
            tables_JGA[dataset][shot][model].append(res['score']["JGA"])
            tables_SACC[dataset][shot][model].append(res['score']["SLOT_ACC"])        
        if "dialKG-parse" in name:
            tables_path1[dataset][shot][model].append(res['score']["path_1"])
            tables_path5[dataset][shot][model].append(res['score']["path_5"])        
            tables_path25[dataset][shot][model].append(res['score']["path_25"])        
            tables_tgt1[dataset][shot][model].append(res['score']["ent_1"])        
            tables_tgt5[dataset][shot][model].append(res['score']["ent_5"])    
            tables_tgt25[dataset][shot][model].append(res['score']["ent_25"])    
            tables_ppl[dataset][shot][model].append(res['score']["ppl"])
        if "dialKG_" in name: 
            tables_FeQA[dataset][shot][model].append(res['score']["feqa"])
            tables_BLEU[dataset][shot][model].append(res['score']["BLEU"])
        if "smd_" in name:
            tables_ppl[dataset][shot][model].append(res['score']["ppl"])
            tables_BLEU_smd[dataset][shot][model].append(res['score']["BLEU"])
            tables_F1_navigate[dataset][shot][model].append(res['score']["F1 navigate"])
            tables_F1_schedule[dataset][shot][model].append(res['score']["F1 schedule"])            
            tables_F1_weather[dataset][shot][model].append(res['score']["F1 weather"])
            tables_F1_smd[dataset][shot][model].append(res['score']["F1"])
        if "wow-parse" in name:
            if "Rprec" in res['score']:
                tables_Rprec[dataset][shot][model].append(res['score']["Rprec"])
        if "skill-selector" in name:
            tables_accSKILL[dataset][shot][model].append(res['score']["acc"])
            tables_F1SKILL[dataset][shot][model].append(res['score']["MacroF1"])
            


model_checkpoints = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'gpt-neo-1.3B', 'gpt-neo-2.7B', 'gpt-j-6B']

sota_results = {"persona":{"ppl":{"AdptBot":11.08,"BST":8.36},
                           "B4": {"AdptBot":0.8767645707034835,"BST":1.1},
                           "F1": {"AdptBot":19.695849729344104,"BST":18.4},
                           "RL": {"AdptBot":21.115480461224518,"BST":22.7},
                           },

                "wow":        {"ppl": {"AdptBot":9.04,"BST":8.61},
                               "B4":  {"AdptBot":9.2,"BST":2.3},
                               "F1":  {"AdptBot":36.1,"BST":18.8}, 
                               "RL":  {"AdptBot":37.6,"BST":17.5},
                               },   
          "wow-parse":        {"ppl": {"DPR":None,"TFIDF":None,"GENRE":None},
                             "Rprec": {"DPR":25.5,"TFIDF":49.0,"GENRE":62.9},
                               "RL":  {"DPR":None,"TFIDF":None,"GENRE":None},
                               },   
                                 
                "ed":        {"ppl": {"AdptBot":12.18,"BST":7.81},
                               "B4":  {"AdptBot":1.2,"BST":1.4},
                               "F1":  {"AdptBot":19.6,"BST":19.1}, 
                               "RL":  {"AdptBot":21.5,"BST":24.2},
                               },

                "DD": {        "ppl": {"DDD":10.4,"NS+MR":11.1},
                               "B4":  {"DDD":None,"NS+MR":None},
                               "F1":  {"DDD":18.2,"NS+MR":None}, 
                               "RL":  {"DDD":None,"NS+MR":None},
                        },

                "smd": {      "BLEU": {"SOTA":14.4,"GPT2":17.03,"AdptBot":17.7},
                               "F1":  {"SOTA":62.7,"GPT2":58.6, "AdptBot":52.56},
                      "F1 navigate":  {"SOTA":57.9,"GPT2":48.37,"AdptBot":43.96}, 
                       "F1 weather":  {"SOTA":57.6,"GPT2":62.87,"AdptBot":54.36}, 
                      "F1 schedule":  {"SOTA":73.1,"GPT2":72.22,"AdptBot":65.7}, 
                               "ppl": {"SOTA":None,"GPT2":None,"AdptBot":None},
                        },    
                "wit": {        "F1": {"BST":22.0,"BART":25.4,"T5":25.7},
                               "KF1": {"BST":22.8,"BART":23.1, "T5":23.5},
                               "ppl": {"BST":9.2,"BART":10.6,"T5":10.1},
                        },
                "msc-dialogue-2": {       "ppl": {"BST":9.0},
                               "B4":  {"BST":None},
                               "F1":  {"BST":None}, 
                               "RL":  {"BST":None},
                        },
                        
                "IC-img": {   "ppl":  {"MBST":12.6,"BST":None},
                               "B4":  {"MBST":0.4,"BST":0.1},
                               "F1":  {"MBST":13.1,"BST":9.2}, 
                               "RL":  {"MBST":18.0,"BST":12.3},
                        },
                "dialKG": {   "FeQA": {"GPT2":26.54,"AdptBot":23.11},
                              "BLEU": {"GPT2":11.10,"AdptBot":10.08},
                               "Rouge-L":  {"GPT2":30.0,"AdptBot":31.0}, 
                               "B4":  {"GPT2":None,"AdptBot":None}, 
                               "ppl": {"GPT2":None,"AdptBot":None}, 
                               "RL":  {"GPT2":None,"AdptBot":None}, 
                               "F1":  {"GPT2":None,"AdptBot":None}, 
                        },
                }


def print_table(table,metric,sk=False):
    if sk:
        model_checkpoints = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'gpt-neo-1.3B', 'gpt-neo-2.7B', 'gpt-j-6B','roberta-base','roberta-large','bert-base-cased']
    else:
        model_checkpoints = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'gpt-neo-1.3B', 'gpt-neo-2.7B', 'gpt-j-6B']
    for dataset_name, results in table.items():
        table = []
        for shot, shots_table in results.items():
            temp = [shot]+[0.0 for _ in model_checkpoints]
            for model, results_values in shots_table.items():
                temp[model_checkpoints.index(model)+1] = "{:.2f}".format(np.mean(results_values))  +"±"+ "{:.4f}".format(np.std(results_values))
            table.append(temp)

        headers = ['Shots','124M', '355M', '774M', '1.5B', '1.3B', '2.7B', '6B', 'RB','RL','BB']

        _, name, _ = dataset_name.split("/")
        name_path = f"results/{name}/"
        with open(f'{name_path}/table_{metric}.md', 'w') as f:
            f.write(tabulate(table,headers,floatfmt=".2f",tablefmt="github"))
        
def make_subplot(results, metric, ax, plt, dataset_name):
    mapper = {'gpt2':r"$\texttt{124M}$", 
              'gpt2-medium':r"$\texttt{355M}$", 
              'gpt2-large':r"$\texttt{774M}$", 
              'gpt2-xl':r"$\texttt{1.5B}$", 
              'gpt-neo-1.3B':r"$\texttt{1.3B}$", 
              'gpt-neo-2.7B':r"$\texttt{2.7B}$", 
              'gpt-j-6B':r"$\texttt{6B}$",
              "DialoGPT": r"$\texttt{DialoGPT}_{\texttt{\hspace{0.5mm}\small{zero}}}$",
              "AdptBot": r"$\texttt{AdptBot}$",
              "DDD": r"$\texttt{DDD}$",
              "BST": r"$\texttt{BST}$",
              "MBST": r"$\texttt{MBST}$",
              "NS+MR": r"$\texttt{NS+MR}$",
              "SOTA":r"$\texttt{SOTA}$",
              "GPT2":r"$\texttt{GPT2}$",
              "BART":r"$\texttt{BART}$",
              "T5":r"$\texttt{T5}$",
              "DPR":r"$\texttt{DPR}$",
              "TFIDF":r"$\texttt{TFIDF}$",
              "GENRE":r"$\texttt{GENRE}$",
              "roberta-base": r"$\texttt{roberta-base}$",
              "roberta-large": r"$\texttt{roberta-large}$",
              "bert-base-cased": r"$\texttt{bert-base-cased}$",
              }

    color_mapper = {'gpt2':'#f5c63c', 
                    'gpt2-medium':'#f47f68', 
                    'gpt2-large':'#bb5098', 
                    'gpt2-xl': '#7a5197', 
                    'gpt-neo-1.3B': '#5344a9', 
                    'gpt-neo-2.7B': '#0b2652', 
                    'gpt-j-6B':'#034ea2',
                    "DialoGPT": '#b53f84',
                    "AdptBot": '#264653',
                    "DDD": '#264653',
                    "BST": '#428af5',
                    "MBST": '#264653',
                    "NS+MR": '#428af5',
                    # "BST": 'black',
                    "SOTA":'#428af5',
                    "GPT2":'#264653',
                    "BART":'#428af5',
                    "T5":'#264653',
                    "DPR": '#264653',
                    "TFIDF": '#428af5',
                    "GENRE": '#b53f84',
                    "roberta-base": '#b53f84',
                    "roberta-large": '#428af5',
                    "bert-base-cased": '#b53f84',
                    }

    data = {
            'gpt2':{"k-shot":[],f"{metric}":[],f"std_{metric}":[]}, 
            'gpt2-medium':{"k-shot":[],f"{metric}":[],f"std_{metric}":[]}, 
            'gpt2-large':{"k-shot":[],f"{metric}":[],f"std_{metric}":[]}, 
            'gpt2-xl':{"k-shot":[],f"{metric}":[],f"std_{metric}":[]}, 
            'gpt-neo-1.3B':{"k-shot":[],f"{metric}":[],f"std_{metric}":[]}, 
            'gpt-neo-2.7B':{"k-shot":[],f"{metric}":[],f"std_{metric}":[]}, 
            'gpt-j-6B':{"k-shot":[],f"{metric}":[],f"std_{metric}":[]}
            }
    for shot, shots_table in results.items():
        for model, results_values in shots_table.items():
            if model in data.keys():
                data[model]["k-shot"].append(int(shot))
                data[model][f"{metric}"].append(np.mean(results_values))
                data[model][f"std_{metric}"].append(np.std(results_values))
    markers =  ['o','v', '^', 's', 'x', 'D',"*"]
    for i_k, key in enumerate(data.keys()):
        df = pd.DataFrame(data[key],columns=['k-shot',f"{metric}",f"std_{metric}"])
        df = df.sort_values(by=['k-shot'])
        ax.plot(df['k-shot'], df[f"{metric}"], color=color_mapper[key], marker=markers[i_k], label=mapper[key])
        
        ax.fill_between(df['k-shot'], df[f"{metric}"] - df[f"std_{metric}"], df[f"{metric}"] + df[f"std_{metric}"],color=color_mapper[key], alpha=0.1)
    
    data = {
            'roberta-base':{"k-shot":[],f"{metric}":[],f"std_{metric}":[]}, 
            'roberta-large':{"k-shot":[],f"{metric}":[],f"std_{metric}":[]}, 
            }
    for shot, shots_table in results.items():
        for model, results_values in shots_table.items():
            if model in data.keys():
                data[model]["k-shot"].append(int(shot))
                data[model][f"{metric}"].append(np.mean(results_values))
                data[model][f"std_{metric}"].append(np.std(results_values))
    linestyle = ["--","dotted"]
    for id_k, key in enumerate(data.keys()):
        df = pd.DataFrame(data[key],columns=['k-shot',f"{metric}",f"std_{metric}"])
        df = df.sort_values(by=['k-shot'])
        ax.plot(df['k-shot'], df[f"{metric}"],linestyle=linestyle[id_k],color=color_mapper[key], label=mapper[key])
        
        ax.fill_between(df['k-shot'], df[f"{metric}"] - df[f"std_{metric}"], df[f"{metric}"] + df[f"std_{metric}"],color=color_mapper[key], alpha=0.1)
    

    if dataset_name in sota_results:
        for sota_model, value in sota_results[dataset_name][metric].items():
            if value!= None:
                ax.axhline(y=value, color=color_mapper[sota_model], linestyle='--', linewidth='1.2')#, label=mapper[sota_model])

                ax.text(0.0, value, mapper[sota_model], fontsize=8, va='top', ha='center')#, backgroundcolor='w')
    ax.set_xlabel('k-shot (Dialogue)', fontsize=14)
    ax.set_ylabel(f"{metric}", fontsize=14)
    ax.grid(True)


def make_subplot_by_size(results, metric, ax, plt, dataset_name):
    mapper = {'gpt2':r"$\texttt{124M}$", 
              'gpt2-medium':r"$\texttt{355M}$", 
              'gpt2-large':r"$\texttt{774M}$", 
              'gpt2-xl':r"$\texttt{1.5B}$", 
              'gpt-neo-1.3B':r"$\texttt{1.3B}$", 
              'gpt-neo-2.7B':r"$\texttt{2.7B}$", 
              'gpt-j-6B':r"$\texttt{6B}$",
              "DialoGPT": r"$\texttt{DialoGPT}_{\texttt{\hspace{0.5mm}\small{zero}}}$",
              "AdptBot": r"$\texttt{AdptBot}$",
              "DDD": r"$\texttt{DDD}$",
              "MBST": r"$\texttt{MBST}$",
              "BST": r"$\texttt{BST}$",
              "NS+MR": r"$\texttt{NS+MR}$",
              "SOTA":r"$\texttt{SOTA}$",
              "GPT2":r"$\texttt{GPT2}$",
              "BART":r"$\texttt{BART}$",
              "T5":r"$\texttt{T5}$",
              "DPR":r"$\texttt{DPR}$",
              "TFIDF":r"$\texttt{TFIDF}$",
              "GENRE":r"$\texttt{GENRE}$",
              }

    color_mapper = {'gpt2':'#f5c63c', 
                    'gpt2-medium':'#f47f68', 
                    'gpt2-large':'#bb5098', 
                    'gpt2-xl': '#7a5197', 
                    'gpt-neo-1.3B': '#5344a9', 
                    'gpt-neo-2.7B': '#0b2652', 
                    'gpt-j-6B':'#034ea2',
                    "DialoGPT": '#b53f84',
                    "AdptBot": '#264653',
                    "DDD": '#264653',
                    "BST": '#428af5',
                    "MBST": '#264653',
                    "NS+MR": '#428af5',
                    "SOTA":'#428af5',
                    "GPT2":'#428af5',
                    "BART":'#428af5',
                    "T5":'#264653',
                    "DPR": '#264653',
                    "TFIDF": '#428af5',
                    "GENRE": '#b53f84',
                    # "BST": 'black',
                    }

    size_models = ['0.1B', '0.3B', '0.8B','1.3B', '1.5B', '2.7B', '6B']
    mapper_model_to_index = {'gpt2':0,'gpt2-medium':1,
                            'gpt2-large':2,'gpt2-xl':4,
                            'gpt-neo-1.3B':3,'gpt-neo-2.7B':5,'gpt-j-6B':6}
    max_shot = list(results.keys())
    data = {int(m):[0,0,0,0,0,0,0] for m in max_shot}
    data_STD = {int(m):[0,0,0,0,0,0,0] for m in max_shot}
    
    for shot, shots_table in results.items():
        for model, results_values in shots_table.items():
            if model in mapper_model_to_index.keys():
                data[int(shot)][mapper_model_to_index[model]] = np.mean(results_values)
                data_STD[int(shot)][mapper_model_to_index[model]] = np.std(results_values)

    colors =  ['#f5c63c','#bb5098', '#7a5197', '#5344a9', '#0b2652', '#034ea2']
    markers =  ['o','v', '^', 's', 'x', 'D']
    data = {k: v for k, v in sorted(data.items(), key=lambda item: item[0])}
    data_STD = {k: v for k, v in sorted(data_STD.items(), key=lambda item: item[0])}
    for id_s, (shot, values) in enumerate(data.items()):
        list_with_zero = [i for i, e in enumerate(values) if e == 0]
        x = [i for i in range(7) if i not in list_with_zero]
        y = [v for v in values if v != 0]
        y_std = [v for v in data_STD[shot]if v != 0]
        ax.plot(x, y, color=colors[id_s], marker=markers[id_s], markersize=4, label=f"{shot}-shot")
        if len(x) == len(y_std):
            ax.fill_between(x, np.array(y) - np.array(y_std), np.array(y) + np.array(y_std),color=colors[id_s], alpha=0.1)
        
    ax.set_xticklabels(size_models)
    ax.set_xticks(np.arange(0, 8))

    if dataset_name in sota_results:
        for sota_model, value in sota_results[dataset_name][metric].items():
            if value!= None:
                ax.axhline(y=value, color=color_mapper[sota_model], linestyle='--', linewidth='1.2')#, label=mapper[sota_model])
                ax.text(0.0, value, mapper[sota_model], fontsize=8, va='top', ha='center')#, backgroundcolor='w')

    # ax.set_title(f"{metric}")
    if metric not in ["B4", "F1"]:
        ax.set_xlabel('Model Size', fontsize=14)
    if metric == "ppl":
        metric = "Perplexity (P)"
    ax.set_ylabel(f"{metric}", fontsize=14)
    ax.grid(True)


def make_plot_tables(t1,t2,t3,t4, by_size): 
    datasets = list(t1.keys())
    for d in datasets:
        fig, axs = plt.subplots(ncols=2, nrows=2, constrained_layout=True, gridspec_kw={'hspace' : 0.15})

        if by_size:
            make_subplot_by_size(results=t1[d],metric="B4", ax=axs[0, 0],plt=plt,dataset_name=d.replace("generations/",""))
            make_subplot_by_size(results=t2[d],metric="F1", ax=axs[0, 1],plt=plt,dataset_name=d.replace("generations/",""))
            make_subplot_by_size(results=t3[d],metric="RL", ax=axs[1, 0],plt=plt,dataset_name=d.replace("generations/",""))
            make_subplot_by_size(results=t4[d],metric="ppl", ax=axs[1, 1],plt=plt,dataset_name=d.replace("generations/",""))
        else:
            make_subplot(results=t1[d],metric="B4", ax=axs[0, 0],plt=plt,dataset_name=d.replace("generations/",""))
            make_subplot(results=t2[d],metric="F1", ax=axs[0, 1],plt=plt,dataset_name=d.replace("generations/",""))
            make_subplot(results=t3[d],metric="RL", ax=axs[1, 0],plt=plt,dataset_name=d.replace("generations/",""))
            make_subplot(results=t4[d],metric="ppl", ax=axs[1, 1],plt=plt,dataset_name=d.replace("generations/",""))
        d = d.replace("generations/","")
        # plt.title(f'{d}', fontsize=14)
        lines, labels = fig.axes[-1].get_legend_handles_labels()
    
        fig.legend(lines, labels, loc = 'center', ncol=4)

        # plt.legend()
        if by_size:
            plt.savefig(f"results/{d}_bysize.png",dpi=400)
        else:
            plt.savefig(f"results/{d}.png",dpi=400)
        plt.close()


def make_plot_tables_tri(t1,t2,t3, m1,m2,m3, by_size, by_domain): 
    datasets = list(t1.keys())
    for d in datasets:
        fig, axs = plt.subplots(ncols=3, nrows=1, constrained_layout=True, figsize=(8,4))
        if by_size:
            make_subplot_by_size(results=t1[d],metric=m1, ax=axs[0],plt=plt,dataset_name=d.replace("generations/",""))
            make_subplot_by_size(results=t2[d],metric=m2, ax=axs[1],plt=plt,dataset_name=d.replace("generations/",""))
            make_subplot_by_size(results=t3[d],metric=m3, ax=axs[2],plt=plt,dataset_name=d.replace("generations/",""))
        else:
            make_subplot(results=t1[d],metric=m1, ax=axs[0],plt=plt,dataset_name=d.replace("generations/",""))
            make_subplot(results=t2[d],metric=m2, ax=axs[1],plt=plt,dataset_name=d.replace("generations/",""))
            make_subplot(results=t3[d],metric=m3, ax=axs[2],plt=plt,dataset_name=d.replace("generations/",""))
        d = d.replace("generations/","")

        plt.legend()
        if by_size:
            plt.savefig(f"results/{d}_bydomain_{by_domain}_bysize.png",dpi=400)
        else:
            plt.savefig(f"results/{d}_bydomain_{by_domain}.png",dpi=400)
        plt.close()

def make_plot_tables_bi(t1,t2,m1,m2, by_size, by_domain): 
    datasets = list(t1.keys())
    for d in datasets:
        print("PLOTTING DATASET")
        fig, axs = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(8,4))
        if by_size:
            make_subplot_by_size(results=t1[d],metric=m1, ax=axs[0],plt=plt,dataset_name=d.replace("generations/",""))
            make_subplot_by_size(results=t2[d],metric=m2, ax=axs[1],plt=plt,dataset_name=d.replace("generations/",""))
        else:
            make_subplot(results=t1[d],metric=m1, ax=axs[0],plt=plt,dataset_name=d.replace("generations/",""))
            make_subplot(results=t2[d],metric=m2, ax=axs[1],plt=plt,dataset_name=d.replace("generations/",""))
        d = d.replace("generations/","")

        plt.legend()
        plt.savefig(f"results/{d}_bydomain_{by_domain}.png",dpi=400)
        plt.close()


if len(tables_B4.keys())>0:
    make_plot_tables(tables_B4,tables_F1,tables_RL,tables_ppl, True)
    make_plot_tables(tables_B4,tables_F1,tables_RL,tables_ppl, False)
    print_table(tables_B4,"B4")
    print_table(tables_F1,"F1")
    print_table(tables_RL,"RL")
    print_table(tables_ppl,"PPL")


if len(tables_accSKILL.keys())>0:
    make_plot_tables_bi(tables_accSKILL,tables_F1SKILL,"ACC","F1", False, False) 
    print_table(tables_accSKILL, "ACC", True)    
    print_table(tables_F1SKILL, "F1", True)  
    
if len(tables_BLEU_smd.keys())>0:
    make_plot_tables_tri(tables_BLEU_smd,tables_F1_smd,tables_ppl,"BLEU","F1","ppl", True, False)
    make_plot_tables_tri(tables_BLEU_smd,tables_F1_smd,tables_ppl,"BLEU","F1","ppl", False, False)

    make_plot_tables_tri(tables_F1_navigate,tables_F1_schedule,tables_F1_weather,"F1 navigate","F1 schedule","F1 weather", True, True)
    make_plot_tables_tri(tables_F1_navigate,tables_F1_schedule,tables_F1_weather,"F1 navigate","F1 schedule","F1 weather", False, True)
    
    print_table(tables_BLEU_smd,"BLEU")
    print_table(tables_F1_smd,"F1")
    print_table(tables_F1_navigate,"F1_nav")
    print_table(tables_F1_schedule,"F1_wea")
    print_table(tables_F1_weather,"F1_sch")   


if len(tables_JGA.keys())> 0: 
    make_plot_tables_tri(tables_JGA,tables_SACC,tables_ppl,"JGA","SLOTACC","ppl", True, False)
    make_plot_tables_tri(tables_JGA,tables_SACC,tables_ppl,"JGA","SLOTACC","ppl", False, False)

    print_table(tables_JGA,"JGA")    
    print_table(tables_SACC,"SACC")  
  

if len(tables_path1.keys())> 0: 
    make_plot_tables_tri(tables_path1,tables_tgt1,tables_ppl,"Path@1","Tgt@1","ppl", True, False)
    make_plot_tables_tri(tables_path1,tables_tgt1,tables_ppl,"Path@1","Tgt@1","ppl", False, False)
    print_table(tables_path1,"PATH1")
    print_table(tables_path5,"PATH5")
    print_table(tables_path25,"PATH25")
    print_table(tables_tgt1,"TGT1")
    print_table(tables_tgt5,"TGT5")
    print_table(tables_tgt25,"TGT25")
  
if len(tables_KF1.keys())>0:
    make_plot_tables_tri(tables_F1,tables_KF1,tables_ppl,"F1","KF1","ppl", True, False)
    make_plot_tables_tri(tables_F1,tables_KF1,tables_ppl,"F1","KF1","ppl", False, False)
    print_table(tables_KF1,"KF1")


if len(tables_Rprec.keys())>0:
    make_plot_tables_tri(tables_Rprec,tables_RL,tables_ppl,"Rprec","RL","ppl", True, False)
    make_plot_tables_tri(tables_Rprec,tables_RL,tables_ppl,"Rprec","RL","ppl", False, False)
    print_table(tables_Rprec,"Rprec")


if len(tables_FeQA.keys())>0:
    make_plot_tables_tri(tables_FeQA,tables_RL,tables_BLEU,"FeQA","Rouge-L","BLEU", True, False)
    make_plot_tables_tri(tables_FeQA,tables_RL,tables_BLEU,"FeQA","Rouge-L","BLEU", False, False)
    print_table(tables_FeQA, "feqa")  
    print_table(tables_BLEU, "BLEU")  



