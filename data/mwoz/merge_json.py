import json


data = []
for domain_name in ["attraction","hotel","taxi","restaurant","train"]:
    data += json.load(open(f'{domain_name}-test.json'))
with open(f'test.json', 'w') as fp:
    json.dump(data, fp, indent=4)
