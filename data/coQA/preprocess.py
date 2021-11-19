import json
import requests

# reader for CoQA dataset
def read_coQA(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# "source": "mctest",
# "id": "3dr23u6we5exclen4th8uq9rb42tel",
# "filename": "mc160.test.41",
def preprocess_coQA(data, save_path):
    preproc_data = []
    for item in data["data"]:
        temp = {}
        temp['id'] = item['id']
        temp['source'] = item['source']
        temp['filename'] = item['filename']
        temp['meta'] = item['story']
        temp['dialogue'] = []
        temp['turn_id'] = []
        assert len(item['questions']) == len(item['answers'])
        for q,a in zip(item['questions'], item['answers']):
            assert q['turn_id'] == a['turn_id']
            temp['dialogue'].append([q['input_text'],a['input_text']])
            temp['turn_id'].append(q['turn_id'])
        preproc_data.append(temp)

    # save preprocessed data in json
    with open(save_path, 'w') as f:
        json.dump(preproc_data, f, indent=4)    

def download_from_link(url, file_path):
    
    r = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(r.content)
        

if __name__ == "__main__":
    # download CoQA dataset
    download_from_link('https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json', 'coqa-train-v1.0.json')
    download_from_link('https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json', 'coqa-dev-v1.0.json')
    
    # # CoQA dataset
    data_path = 'coqa-dev-v1.0.json'
    save_path = 'valid.json'
    data = read_coQA(data_path)
    preprocess_coQA(data, save_path)

    data_path = 'coqa-train-v1.0.json'
    save_path = 'train.json'
    data = read_coQA(data_path)
    preprocess_coQA(data, save_path)