import csv
import json


data = []
with open('train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    line_count = 0
    for row in csv_reader:
        data.append({"dialogue":row[0],"query":row[2]})


with open(f'train.json', 'w') as fp:
    json.dump(data, fp, indent=4)

data = []
with open('valid.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    line_count = 0
    for row in csv_reader:
        data.append({"dialogue":row[0],"query":row[2]})


with open(f'valid.json', 'w') as fp:
    json.dump(data, fp, indent=4)

data = []
with open('test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    line_count = 0
    for row in csv_reader:
        data.append({"dialogue":row[0],"query":row[2]})


with open(f'test.json', 'w') as fp:
    json.dump(data, fp, indent=4)
