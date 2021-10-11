# Finetuning Classifier for Skill-Selection
In this folder, we implement a simple classifier for the skill selection task. The code is based on the HuggingFace code ```run_glue.py```. 

Firstly, we generate the few-shot datasets by running:
```
python generate_data.py
```
This creates a folder with the required ```*.json``` files for training the model. 
After this step, we training RoBERTa (based/large) by running:
```
python runner.py
```
This is a meta-script for automatically running scripts in multiple GPUs.
Finally, we post-process the runs with the ```post_process_results.py``` script, which convert the output into the right format for tables and plots. 