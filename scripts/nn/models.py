from torch import nn

n_channels = 32
n_recordings = 99

class DNN(nn.Module):
  def __init__(self, sizes=(5000, 500, 1000), dropout_probs=(0.25, 0.5)):
    super(DNN, self).__init__()

    self.linear1 = nn.Linear(n_channels * n_recordings, sizes[0])
    self.linear2 = nn.Linear(sizes[0], sizes[1])
    self.linear3 = nn.Linear(sizes[1], sizes[2])
    self.linear4 = nn.Linear(sizes[2], 1) # binary classification: high vs low

    self.dropout1 = nn.Dropout(dropout_probs[0])
    self.dropout2 = nn.Dropout(dropout_probs[1])

    self.relu = nn.ReLU()

    self.flatten = nn.Flatten(start_dim=1)

  def forward(self, x):
    x = self.flatten(x)

    x = self.dropout1(x)
    x = self.linear1(x)
    x = self.relu(x)

    x = self.dropout2(x)
    x = self.linear2(x)
    x = self.relu(x)

    x = self.dropout2(x)
    x = self.linear3(x)
    x = self.relu(x)

    x = self.dropout2(x)
    x = self.linear4(x)

    return x


class CNN(nn.Module):
  def __init__(self, dropout_probs=(0.25, 0.15, 0.5, 0.25)):
      super(CNN, self).__init__()

      self.conv1 = nn.Conv2d(1, 20, (3, 3), padding=(1, 1))
      self.conv2 = nn.Conv2d(20, 40, (3, 3), padding=(1, 1))

      self.maxpool = nn.MaxPool2d((2, 2))

      self.linear1 = nn.Linear(40 * 16 * 49, 128)
      self.linear2 = nn.Linear(128, 1)

      self.tanh = nn.Tanh()
      self.relu = nn.ReLU()

      self.dropout0 = nn.Dropout(dropout_probs[0])
      self.dropout1 = nn.Dropout2d(dropout_probs[1])
      self.dropout2 = nn.Dropout(dropout_probs[2])
      self.dropout3 = nn.Dropout(dropout_probs[3])

      self.flatten = nn.Flatten(start_dim=1)

      self.softplus = nn.Softplus()

  def forward(self, x):
    x = x[:,None,:,:] # add dummy dim for channel

    x = self.dropout0(x)

    x = self.relu(self.conv1(x))
    x = self.dropout1(x)

    x = self.relu(self.conv2(x))
    x = self.dropout1(x)
    x = self.maxpool(x)

    x = self.flatten(x)

    x = self.dropout2(x)

    x = self.relu(self.linear1(x))
    x = self.dropout3(x)

    x = self.linear2(x)

    return x
