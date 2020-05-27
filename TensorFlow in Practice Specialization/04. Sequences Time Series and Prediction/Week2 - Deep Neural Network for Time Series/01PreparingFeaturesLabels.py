import tensorflow as tf

dataset = tf.data.Dataset.range(10)
# for val in dataset:
#     print(val.numpy())

dataset = dataset.window(5, shift=1, drop_remainder=True)
# for window_dataset in dataset:
#     for val in window_dataset:
#         print(val.numpy(), end=" ")
#     print()

dataset = dataset.flat_map(lambda window: window.batch(5))
# for window in dataset:
#     print(window.numpy())

dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
    print(x.numpy(), y.numpy())

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift = 1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset