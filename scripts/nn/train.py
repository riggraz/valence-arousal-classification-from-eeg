# Imports
import torch, yaml, os
from torch.utils.data import DataLoader, random_split

from datasets import DEAP, MAHNOB
from models import DNN, CNN
from utils import check_train_test_split_balanced
from train_utils import train

# Read configs
with open('./scripts/nn/configs/deap_dnn_arousal.yml') as yaml_file:
  config = yaml.load(yaml_file, Loader=yaml.FullLoader)

# Hyperparams
num_epochs = config['TRAIN']['num_epochs']
batch_size = config['TRAIN']['batch_size']
lr = config['TRAIN']['lr']
momentum = config['TRAIN']['momentum']

# Model
model_type = config['MODEL']['model']

# Train
classification_of = config['TRAIN']['classification_of']

# Dataset
dataset_to_use = config['DATASET']['dataset_to_use']

# Export
model_path = config['EXPORT']['model_path']
model_name = f'{dataset_to_use}-{model_type}-{classification_of}'

if dataset_to_use == 'deap':
  dataset_path = config['DATASET']['deap_dataset_path']
  dataset = DEAP(dataset_path)
elif dataset_to_use == 'mahnob':
  dataset_path = config['DATASET']['mahnob_dataset_path']
  dataset = MAHNOB(dataset_path)
else:
  assert False

# check_dataset_balanced(dataset_path)

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

check_train_test_split_balanced(train_loader, test_loader)

# Model
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

def init_weights(m):
  if isinstance(m, torch.nn.Linear):
    torch.nn.init.xavier_normal_(m.weight)
    m.bias.data.fill_(0.0)

model.apply(init_weights)

# Launch training
best_acc_test_set = train(
  model,
  train_loader,
  torch.nn.BCEWithLogitsLoss(),
  optimizer,
  classification_of=classification_of,
  num_epochs=num_epochs,
  do_check_accuracy=True,
  check_accuracy_every=1,
  test_loader=test_loader,
  model_path=model_path,
  model_name=model_name,
)

os.system(f'say Best test accuracy {(best_acc_test_set*100):.2f}%')