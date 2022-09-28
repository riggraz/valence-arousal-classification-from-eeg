import mne
import os
import xml.etree.ElementTree as ET
import numpy as np
import pickle
import reduce_dim

eeg_channels = [
  'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
  'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
  'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
  'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
]
assert len(eeg_channels) == 32

# dir_path must be a directory containing one subdir for each experiment (session)
# each subdirectory must contain both a .bdf file (eeg recordings)
# and a .xml file (labels)
src_path = '/Users/riccardo/Documents/datasets/MAHNOB/Sessions/'

# dst_path is the directory which will contain n .dat files, one for each experiment (session)
# the folder must already exist
dst_path = '/Users/riccardo/Documents/datasets/MAHNOB/mahnob_preprocessed_standardized2_referencedavg_128hz_bandpass/'

n_sessions_exported = 0
sessions = os.listdir(src_path)

for i, session in enumerate(sessions):
  print(f'Session {i+1}/{len(sessions)} ({session})')

  session_path = os.path.join(src_path, session)
  ls = os.listdir(session_path)

  if len(ls) != 2:
    print(f'Wrong number of files in {session}: expected 2, found {len(ls)}. Skipping this session.')
    continue

  bdf_file = ls[0] if '.bdf' in ls[0] else ls[1]
  xml_file = ls[0] if '.xml' in ls[0] else ls[1]

  # Read the .xml file
  xml = ET.parse(os.path.join(session_path, xml_file))
  root = xml.getroot()
  attributes = root.attrib

  if not 'feltVlnc' in attributes or not 'feltArsl' in attributes or not 'sessionId' in attributes:
    print('No annotations for valence and/or arousal. Skipping this session.')
    continue

  session_id = int(attributes['sessionId'])

  valence = float(attributes['feltVlnc'])
  arousal = float(attributes['feltArsl'])
  #emotion = int(attributes['feltEmo'])
  #subject = root.getiterator('subject')[0].attrib['id']

  labels = np.array([valence, arousal])

  # Read and preprocess the .bdf file
  bdf = mne.io.read_raw_bdf(os.path.join(session_path, bdf_file), verbose=False, preload=True)

  # Set EEG reference
  # mne.set_eeg_reference(bdf, ref_channels=['Cz'], copy=False, verbose=False)
  mne.set_eeg_reference(bdf, ref_channels='average', copy=False, verbose=False)

  # Apply a 4-45Hz bandpass filter
  # bdf = bdf.filter(4.0, 45.0, picks=eeg_channels, verbose=False)

  # Picks only the 32 EEG channels
  data = bdf.get_data(picks=eeg_channels)

  # DEAP has 8064 recordings for each experiment
  # MAHNOB usualy has more (~19'000) and in variable number,
  # and also has 30 seconds of measurements
  # before and after each experiment
  # So, the middle 8064 recordings are extracted
  n_target_samples = 8064 * 2
  n_samples = data.shape[1]

  start_sample = 0
  end_sample = n_samples - 1

  if n_samples > n_target_samples:
    start_sample = (n_samples // 2) - (n_target_samples // 2)
    end_sample = start_sample + n_target_samples
  elif n_samples < n_target_samples:
    assert False

  data = data[:,start_sample:end_sample]
  assert data.shape == (len(eeg_channels), n_target_samples)

  data = reduce_dim.reduce_dim(data)
  assert data.shape == (len(eeg_channels), 99)

  # Standardize data (indipendently for each channel, after dim reduction)
  # for m in range(32):
  #   data[m] = (data[m] - np.mean(data[m])) / np.std(data[m])

  # Standardize data (globally, after dim reduction)
  data = (data - np.mean(data)) / np.std(data)

  # Pack both data and label into a pickled .dat file
  data = {
    'data': data,
    'labels': labels
  }

  dat_file_path = os.path.join(dst_path, f'{session_id}.dat')
  dat_file = open(dat_file_path, mode='w+b')
  pickle.dump(data, dat_file)

  n_sessions_exported += 1
  print(f'{dat_file_path} exported successfully.')

print('Done.')
print(f'{n_sessions_exported} sessions out of {len(sessions)} exported.')
