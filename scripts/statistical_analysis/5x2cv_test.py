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
with open('./scripts/nn/configs/deap_dnn_valence.yml') as yaml_file:
  config = yaml.load(yaml_file, Loader=yaml.FullLoader)

# Hyperparams
num_epochs = 250
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

# cross validation
def cross_validation(kfold):
  accuracies = []

  for fold, (train_idx, test_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
    print('=' * 50)
    print(f'Fold #{fold+1}')

    # Build data loader from k-fold indices
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    # Build models
    dnn = DNN(
      sizes=(5000, 500, 1000),
      dropout_probs=(0.25, 0.5)
    )
    dnn_optimizer = torch.optim.RMSprop(dnn.parameters(), lr=lr)

    cnn = CNN(dropout_probs=(0.25, 0.15, 0.5, 0.25))
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=lr, momentum=momentum)

    models_optimizers = [(dnn, dnn_optimizer), (cnn, cnn_optimizer)]

    for model_i, (model, optimizer) in enumerate(models_optimizers):
      print(f'Model {model_i+1}')

      # Train on training fold
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

      accuracies.append(accuracy)

      print(f'Test accuracy of model {model_i+1} on fold {fold+1} = {(accuracy*100):.2f}%')

  return accuracies


# 5x2cv
p_vars = []
p1_first_iteration = None

for i in range(5):
  print('/' * 50)
  print(f'CV ITERATION #{i+1}')
  print('/' * 50)

  kfold = KFold(n_splits=2, shuffle=True)

  accuracies = cross_validation(kfold)

  assert len(accuracies) == 4

  p1_dnn = accuracies[0]
  p1_cnn = accuracies[1]
  p2_dnn = accuracies[2]
  p2_cnn = accuracies[3]

  p1 = p1_dnn - p1_cnn
  p2 = p2_dnn - p2_cnn

  p_avg = (p1 + p2) / 2
  p_var = (p1 - p_avg)**2 + (p2 - p_avg)**2

  p_vars.append(p_var)

  # first iteration only
  if i == 0:
    p1_first_iteration = p1

# Compute the 5x2cv test statistic
t = p1_first_iteration / np.sqrt(0.2 * np.sum(p_vars))

print(f'p_vars = {p_vars}')
print(f'p1_first_iteration = {p1_first_iteration}')
print(f't statistic = {t}')

with open(f'{get_current_timestamp()}-5x2cv-results-{dataset_to_use}.txt', mode='w') as f:
  print(config, file=f)
  print(f'Number of epochs = {num_epochs}', file=f)
  print('\n', file=f)
  print(f't statistic = {t}', file=f)

# Alpha value of 0.05
# Under null hypothesis, t has a t-distribution with 5 degrees of freedom
# ===> t > 2.571 with probability < 5%
# If t > 2.571 => we can reject the null hypothesis
# If t < 2.571 => we fail to reject the null hypothesis

if t >= 2.571:
  print('Reject H0')
else:
  print('Fail to reject H0')
