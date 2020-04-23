#!pip install -q tensorflow-datasets

import tensorflow_datasets as tfds 
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

import numpy as np
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# str(s.tonumpy()) is needed in Python 3 instead of just s.numpy()
for s, l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(str(l.numpy()))

# The values for S and I are tensors, so by calling their NumPy method

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length,truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

# Hyperparameters are defined above for managing changes later
# testing_sequences might be seeing lot of oov tokens because it might not have seen
# the word_index in training data.

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length = max_length),
    Flatten(),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])