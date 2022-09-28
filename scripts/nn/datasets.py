import os, pickle, torch
from torch.utils.data import Dataset

n_channels = 32
n_recordings = 99

# High val = 708, low val = 572
# High ar = 737, low ar = 543
class DEAP(Dataset):
  def __init__(self, dataset_path):
    self.dataset_path = dataset_path

    self.sessions = os.listdir(dataset_path)
    
    # remove .DS_Store if present
    if '.DS_Store' in self.sessions:
      self.sessions.remove('.DS_Store')

  def __len__(self):
    return len(self.sessions)

  def __getitem__(self, index):
    file_path = os.path.join(self.dataset_path, self.sessions[index])
    with open(file_path, mode='rb') as file:
      session = pickle.load(file)

      data, labels = session['data'], session['labels']
      data, labels = torch.from_numpy(data), torch.from_numpy(labels)
      data, labels = data.float(), labels.float()

      # 1 = high value, 0 = low value
      labels = (labels >= 5.0).long()

      assert data.shape == (n_channels, n_recordings)
      assert labels.shape == (2,)

      return data, labels


class MAHNOB(DEAP):
  def __init__(self, dataset_path):
    super().__init__(dataset_path)

  def __len__(self):
    return super().__len__()

  def __getitem__(self, index):
    return super().__getitem__(index)