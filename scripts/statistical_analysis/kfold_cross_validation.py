import torch, os, sys, yaml
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(1, os.path.abspath('./scripts/nn/'))

from datasets import DEAP, MAHNOB
from models import DNN, CNN
from train_utils import train, check_accuracy
from utils import get_current_timestamp

# Config file
with open('./scripts/nn/configs/deap_dnn_arousal.yml') as yaml_file:
  config = yaml.load(yaml_file, Loader=yaml.FullLoader)

# Hyperparams
num_epochs = 150
batch_size = config['TRAIN']['batch_size']
lr = config['TRAIN']['lr']
momentum = config['TRAIN']['momentum']

# Model
model_type = config['MODEL']['model']

# Train
classification_of = config['TRAIN']['classification_of']

# Dataset
dataset_to_use = config['DATASET']['dataset_to_use']

if dataset_to_use == 'deap':
  dataset_path = config['DATASET']['deap_dataset_path']
  dataset = DEAP(dataset_path)
elif dataset_to_use == 'mahnob':
  dataset_path = config['DATASET']['mahnob_dataset_path']
  dataset = MAHNOB(dataset_path)
else:
  assert False


# K-fold external cross validation
n_splits = 32
kfold = KFold(n_splits=n_splits, shuffle=True)

accuracies = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
  print('=' * 50)
  print(f'Fold #{fold+1}')

  # Build data loader from k-fold indices
  train_sampler = SubsetRandomSampler(train_idx)
  test_sampler = SubsetRandomSampler(test_idx)

  train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
  test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

  # Build model
  if model_type == 'dnn':
    model = DNN(
      sizes=tuple(config['MODEL']['sizes']),
      dropout_probs=tuple(config['MODEL']['dropout_probs'])
    )
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
  elif model_type == 'cnn':
    model = CNN(dropout_probs=tuple(config['MODEL']['dropout_probs']))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  else:
    assert False

  # Train model on training fold
  train(
    model,
    train_loader,
    torch.nn.BCEWithLogitsLoss(),
    optimizer,
    classification_of='valence',
    num_epochs=num_epochs,
    do_check_accuracy=False,
  )

  # Check accuracy of model on test fold
  num_examples = len(test_loader.sampler)
  accuracy = check_accuracy(
    model,
    test_loader,
    num_examples=num_examples,
    classification_of='valence'
  )

  print(f'Test accuracy on fold {fold+1} = {(accuracy*100):.3f}%')

  accuracies.append(accuracy)

avg_accuracy = np.mean(np.array(accuracies))
print(f'Average accuracy = {(avg_accuracy*100):.2f}%')

with open(f'{get_current_timestamp()}-cv-results-{dataset_to_use}-{model_type}.txt', mode='w') as f:
  print(f'Fold accuracies = {accuracies}', file=f)
  print(f'Avg accuracy = {(avg_accuracy*100):.2f}%', file=f)
  print('\n', file=f)
  print(config, file=f)
  print(f'Number of folds = {n_splits}', file=f)
  print(f'Number of epochs = {num_epochs}', file=f)

os.system(f'say Average k-fold accuracy: {(avg_accuracy*100):.2f}%')