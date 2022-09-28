import torch, yaml, os, sys
import numpy as np

from torch.utils.data import DataLoader, random_split
from statsmodels.stats.contingency_tables import mcnemar

sys.path.append(os.path.abspath('./scripts/nn/'))

from datasets import DEAP, MAHNOB
from models import DNN, CNN

# Load config used for training
# Used to get dataset_path, train/test split, etc.
# but not training hyperparams and alike which aren't needed
with open('./scripts/nn/configs/deap_dnn_valence.yml') as yaml_file:
  config = yaml.load(yaml_file, Loader=yaml.FullLoader)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_to_use = config['DATASET']['dataset_to_use']
batch_size = 1
dnn_model_path = f'./pretrained-models/{dataset_to_use}-dnn-valence.pt'
cnn_model_path = f'./pretrained-models/{dataset_to_use}-cnn-valence.pt'

# Dataset and data loaders
if dataset_to_use == 'deap':
  dataset_path = config['DATASET']['deap_dataset_path']
  dataset = DEAP(dataset_path)
elif dataset_to_use == 'mahnob':
  dataset_path = config['DATASET']['mahnob_dataset_path']
  dataset = MAHNOB(dataset_path)
else:
  assert False

seed = config['TRAIN']['seed']
train_set_size, test_set_size = config['TRAIN']['train_test_split']
train_set, test_set = random_split(
  dataset,
  [train_set_size, test_set_size],
  generator=torch.Generator().manual_seed(seed)
)

print(f'{len(dataset)} examples found ({train_set_size} train, {test_set_size} test)')

train_loader = DataLoader(
  train_set,
  batch_size=batch_size,
  shuffle=True
)

test_loader = DataLoader(
  test_set,
  batch_size=batch_size,
  shuffle=True
)

# Create models and load pretrained weights
dnn_model = DNN(
  sizes=(5000, 500, 1000),
  dropout_probs=(0.25, 0.5)
)
dnn_model.load_state_dict(torch.load(dnn_model_path, map_location=device))
dnn_model.eval()

cnn_model = CNN(
  dropout_probs=(0.25, 0.15, 0.5, 0.25)
)
cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
cnn_model.eval()

# McNemar's test

n00 = 0 # examples misclassified by both models
n10 = 0 # examples misclassified by cnn, but not by dnn
n01 = 0 # examples misclassified by dnn, but not by cnn
n11 = 0 # examples classified correctly by both models

for data, label in test_loader:
  with torch.no_grad():
    dnn_model_pred = dnn_model(data)
    cnn_model_pred = cnn_model(data)

    dnn_model_pred = torch.squeeze((dnn_model_pred >= 0.0).long())
    cnn_model_pred = torch.squeeze((cnn_model_pred >= 0.0).long())

    label = label[0,0]

    if   dnn_model_pred != label and cnn_model_pred != label:
      n00 += 1
    elif dnn_model_pred == label and cnn_model_pred != label:
      n10 += 1
    elif dnn_model_pred != label and cnn_model_pred == label:
      n01 += 1
    elif dnn_model_pred == label and cnn_model_pred == label:
      n11 += 1
    else:
      assert False

print(f'n00={n00}, n01={n01}, n10={n10}, n11={n11}')
print(f'n00 + n01 + n10 + n11 = {n00 + n01 + n10 + n11}')
assert (n00 + n01 + n10 + n11) == len(test_set)

contingency_table = [
  [n11, n10],
  [n01, n00]
]

# Calculate McNemar's statistic
result = mcnemar(contingency_table, exact=True)

print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

alpha = 0.05
if result.pvalue > alpha:
	print('Null hypothesis cannot be rejected. The two models have NO meaningfully different performances.')
else:
	print('Null hypothesis can be rejected. The two models have meaningfully different performances.')
