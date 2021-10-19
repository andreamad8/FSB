# Retrievers
In this bot we use three knowledge retrievers. In this folder, we implement the knowledge based not the retriver function (e.g., tfidf). 

## Wikipedia
We use the Wiki dump and mongo DB provided by KILT. Please follow the instruction in [https://github.com/facebookresearch/KILT#kilt-knowledge-source](https://github.com/facebookresearch/KILT#kilt-knowledge-source). Once it is setup, then you can run: 
```
sudo mongod --bind_ip_all
```
For the interactive script, we use the wikipedia title retriever from [https://github.com/goldsmith/Wikipedia](https://github.com/goldsmith/Wikipedia).

## Internet Engine
We use the code in [https://github.com/JulesGM/ParlAI_SearchEngine](https://github.com/JulesGM/ParlAI_SearchEngine). Follow the readme of the page to set up the search engine. Then you can run: 
```
python search_server.py serve --host 0.0.0.0:8081 --search_engine="Bing" --use_description_only --subscription_key "YOUR_KEY"
```
or, if you don't have a bing key: 
```
python search_server.py serve --host 0.0.0.0:8081 
```
which uses google search. 


## Knowledge Graph
As in AdapterBot, we use Neo4J to setup the KG. Please follow the instruction in ```https://github.com/HLTCHKUST/adapterbot/tree/main/retriever/graphdb``` to set up the KG and run: 
```
sudo service neo4j start 
```