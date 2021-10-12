# Metrics
In this forder, we implement the evaluation metrics used in the paper.

## SMD 
The SMD scorer is custum to the task of SMD, so ```smd_score.py`` implement the scorer from Wu et al. 2020. 

## Feqa
The FeQA scorer is used for evaluating DialKG as in Dziri et al 2021. Notice to run this scorer, separated checkpoints for BART and QA model need to downloaded. Please refer to [https://github.com/esdurmus/feqa](https://github.com/esdurmus/feqa) for more information.

## WoW-parse 
We use the scorer from KILT for the Rprec. This is implemented in ```score_retrieval_wow.py```. 