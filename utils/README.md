# Utils
The ```utils.py``` implements general function used in the main.

## WiT retirever
In Wizard of Internet (WiT), we implement a simple searching engine using [https://github.com/JulesGM/ParlAI_SearchEngine](https://github.com/JulesGM/ParlAI_SearchEngine). The file ```wit_parlai_retriever.py``` implement this class that query the API, clean the output and return the pages. This taken from ParlAI. 

## Tfidf
The ```tiidf.py``` implements a simple TFIDF algorithm, which could be useful in the interactive model (MAYBE :) )

## Merging function
The ```merge_file_smd.py``` and ```merge_MSC.py``` implements function to join different generation files. This is needed beacuse our evaluation is done by domain (in SMD) and by session in MSC. 