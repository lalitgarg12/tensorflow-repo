import json

with open("sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection
# https://rishabhmisra.github.io/publications/