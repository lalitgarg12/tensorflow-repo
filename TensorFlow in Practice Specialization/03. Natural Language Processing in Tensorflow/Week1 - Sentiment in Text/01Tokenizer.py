import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

# Tokenizer generates the dictionary of word encodings
# Create vectors out of the sentences

# Pass a Tokenizer object and num_words parameter
# In this case, I'm using 100 which is way too big, as there are only five distinct words in this data. 
# If you're creating a training set based on lots of text, you usually don't know how many unique distinct words there are in that text. 
# So by setting this hyperparameter, what the tokenizer will do is take the top 100 words by volume and just encode those.
# The fit on texts method of the tokenizer then takes in the data and encodes it. 
# The tokenizer provides a word index property which returns a dictionary containing key value pairs, where the key is the word, 
# and the value is the token for that word, which you can inspect by simply printing it out.

# Result: {'i' : 1, 'my' : 3, 'dog' : 4, 'cat' : 5, 'love' : 2}

# Notice: Lower case | Upper case and Punctuations are automatically handled