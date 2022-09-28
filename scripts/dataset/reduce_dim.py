import numpy as np
import scipy.stats as sp

# Input: data array with shape (32, 8064) for DEAP or (32, 8064*2) for MAHNOB
# (i.e. 32 eeg-channels with 8064 recordings each)
# Output: data array with shape (32, 99)
# (the 8064 recordings get chunked to 99 statistical values)
def reduce_dim(data):
  assert (data.shape == (32, 8064) or data.shape == (32, 8064*2))

  processed_data = np.zeros((32, 99))

  for channel_n in range(32):
    # Divide the 8064 recordings in 10 batches of 807 (10th batch: 801)
    if data.shape == (32, 8064):
      batch_size = 807
      n_samples = 8064
    elif data.shape == (32, 8064*2):
      batch_size = 807 * 2
      n_samples = 8064 * 2

    batch_n = 0

    for batch_n in range(10):
      if batch_n != 9:
        batch = data[channel_n,(batch_n*batch_size):(batch_n*batch_size+batch_size)]
      else:
        batch = data[channel_n,(batch_n*batch_size):n_samples]

      processed_data[channel_n,(batch_n * 9):(batch_n * 9 + 9)] = np.array([
        np.mean(batch),
        np.median(batch),
        np.max(batch),
        np.min(batch),
        np.std(batch),
        np.var(batch),
        np.max(batch) - np.min(batch),
        sp.skew(batch),
        sp.kurtosis(batch),
      ])
      
    processed_data[channel_n,90:99] = np.array([
        np.mean(data[channel_n,:]),
        np.median(data[channel_n,:]),
        np.max(data[channel_n,:]),
        np.min(data[channel_n,:]),
        np.std(data[channel_n,:]),
        np.var(data[channel_n,:]),
        np.max(data[channel_n,:]) - np.min(data[channel_n,:]),
        sp.skew(data[channel_n,:]),
        sp.kurtosis(data[channel_n,:]),
    ])

  assert processed_data.shape == (32, 99)
  return processed_data


# data = np.random.rand(32, 8064) * 100
# print(data[2,:])
# processed_data = reduce_dim(data)

# print('=' * 40)
# print(processed_data[2,:])