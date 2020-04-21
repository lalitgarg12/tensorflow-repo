import tensorflow as tf 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100, oov_token="<oov>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences)

#padded = pad_sequences(sequences, padding = 'post', maxlen=5, truncating = 'post')

print(word_index)
print(sequences)
print(padded)

# {'<oov>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}
# [[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]
# [[ 0  0  0  5  3  2  4]
#  [ 0  0  0  5  3  2  7]
#  [ 0  0  0  6  3  2  4]
#  [ 8  6  9  2  4 10 11]]

# These sequences can be passed to pad sequences in order to have them padded like above
# Each row in the matrix has same length. By adding zeros before the sentence.
