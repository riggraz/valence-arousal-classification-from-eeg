import os, pickle, math
import numpy as np
import reduce_dim

src_dir = '/Users/riccardo/Documents/datasets/DEAP/data_preprocessed_python/'
dst_dir = '/Users/riccardo/Documents/datasets/DEAP/deap_preprocessed_standardized_global/'

n_experiments = 40    # experiment per participant
n_channels = 40       # channel per experiment
n_recordings = 8064   # recordings per channel

participants = os.listdir(src_dir)
participants.sort()
assert len(participants) == 32 # DEAP has 32 participants

tot_experiments = 40 * 32
exported_experiments = 0

# Global mean and std calculation
# total_sum = 0.0
# for participant in participants:
#   participant_data_path = os.path.join(src_dir, participant)
#   with open(participant_data_path, mode='r+b') as f:
#     data = pickle.load(f, encoding='latin1')
#     data, labels = data['data'], data['labels']

#     assert data[:,0:32,:].shape == (n_experiments, 32, n_recordings)

#     total_sum += np.sum(data[:,0:32,:])

# global_mean = total_sum / (len(participants) * n_experiments * 32 * n_recordings)

# sum_of_squared_error = 0.0
# for participant in participants:
#   participant_data_path = os.path.join(src_dir, participant)
#   with open(participant_data_path, mode='r+b') as f:
#     data = pickle.load(f, encoding='latin1')
#     data, labels = data['data'], data['labels']

#     assert data[:,0:32,:].shape == (n_experiments, 32, n_recordings)

#     sum_of_squared_error += np.sum(np.power((data[:,0:32,:] - global_mean), np.array([2])))

# global_std = np.sqrt(sum_of_squared_error / (len(participants) * n_experiments * 32 * n_recordings))

# print(f'Global mean = {global_mean}, global std = {global_std}')

# Preprocess data
for i, participant in enumerate(participants):
  print(f'Participant {i+1}/{len(participants)} ({participant})')

  participant_data_path = os.path.join(src_dir, participant)

  with open(participant_data_path, mode='r+b') as f:
    # encoding needed because DEAP data was pickled with Python2
    data = pickle.load(f, encoding='latin1')

    data, labels = data['data'], data['labels']
    assert data.shape == (n_experiments, n_channels, n_recordings)
    assert labels.shape == (n_experiments, 4)

    for j in range(n_experiments):
      # Removes non-EEG channels from data
      data_tmp = data[j,0:32,:]
      assert data_tmp.shape == (32, n_recordings)

      # Standardize data (globally, and before dimensionality reduction)
      # data_tmp = (data_tmp - global_mean) / global_std

      data_tmp = reduce_dim.reduce_dim(data_tmp)
      assert data_tmp.shape == (32, 99)

      # Standardize data (indipendently for each channel, after dim reduction)
      # for m in range(32):
      #   data_tmp[m] = (data_tmp[m] - np.mean(data_tmp[m])) / np.std(data_tmp[m])

      # Standardize data (for all channels, after dim reduction)
      data_tmp = (data_tmp - np.mean(data_tmp)) / np.std(data_tmp)

      # Removes all annotations except for valence and arousal
      label_tmp = labels[j,0:2]
      assert label_tmp.shape == (2,)

      dat = {
        'data': data_tmp,
        'labels': label_tmp,
      }

      dat_file_path = os.path.join(dst_dir, f'{i+1}_{j+1}.dat')
      with open(dat_file_path, mode='w+b') as dat_file:
        pickle.dump(dat, dat_file)

      print(f'{dat_file_path} exported successfully.')
      exported_experiments += 1

print('Done.')
print(f'Exported {exported_experiments} experiments out of {tot_experiments}.')