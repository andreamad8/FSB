from joblib import Parallel, delayed
import queue
import os
import time

# Define number of GPUs available
GPU_available = [0,1,2,3,4,5,6,7,8,9]
N_GPU = len(GPU_available)


## model, training_file, bsz, gradientc_acc, training_file
template = "python run_glue.py --model_name_or_path {} --train_file data/{}.json --validation_file data/valid.json --test_file data/test.json --do_train --do_eval --do_predict --max_seq_length 512 --per_device_train_batch_size {} --gradient_accumulation_steps {} --save_total_limit 1 --fp16 --learning_rate 2e-5 --num_train_epochs 10 --output_dir tmp/{}/ --overwrite_output_dir"

experiments = []
for model, bsz, gradientc_acc in [["roberta-base","8","1"], 
                                 ["roberta-large","1","8"] ]:
    for shot in [7,8,9,10]:
        for rep in [0,1,2]:
            training_file_data = f"train_{shot}_{rep}"
            training_file = f"train_{shot}_{rep}_{model}"
            experiments.append(template.format(model, training_file_data, bsz, gradientc_acc, training_file))

# Put indices in queue
q = queue.Queue(maxsize=N_GPU)
mapper = {}
invert_mapper = {}
for i in range(N_GPU):
    mapper[i] = GPU_available[i]
    invert_mapper[GPU_available[i]] = i
    q.put(i)

def runner(cmd):
    gpu = mapper[q.get()]
    print("RUNNING: ",f"CUDA_VISIBLE_DEVICES={gpu} "+str(cmd))
    os.system(f"CUDA_VISIBLE_DEVICES={gpu} "+str(cmd))
    # os.system("CUDA_VISIBLE_DEVICES=%d %s" % (gpu, cmd))
    q.put(invert_mapper[gpu])

# Change loop
Parallel(n_jobs=N_GPU, backend="threading")( delayed(runner)(e) for e in experiments)

