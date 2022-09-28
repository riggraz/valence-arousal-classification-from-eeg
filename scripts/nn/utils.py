import datetime, math, os, torch, pickle
import matplotlib.pyplot as plt

def get_current_timestamp():
  return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def get_train_test_sizes(train_set_perc, dataset_size):
  train_set_size = math.floor(dataset_size / 100 * train_set_perc)
  test_set_size = dataset_size - train_set_size

  return train_set_size, test_set_size

def plot_loss_and_accuracy_per_epoch(loss_per_epoch, accuracy_per_epoch, model_path):
  fig = plt.figure()
  fig.add_subplot(1, 2, 1)
  plt.title('Average loss per epoch')
  plt.xlabel('epoch')
  plt.ylabel('avg loss')
  plt.plot(range(1, len(loss_per_epoch) + 1), loss_per_epoch)

  fig.add_subplot(1, 2, 2)
  plt.title('Accuracy per epoch')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.plot(range(1, len(accuracy_per_epoch) + 1), accuracy_per_epoch)

  plt.savefig(os.path.join(model_path, f'{get_current_timestamp()}-accuracy-plot.png'))
  # plt.show()

def save_model(model, model_path='/', model_name='model'):
  model_final_path = f'{os.path.join(model_path, model_name)}.pt'
  torch.save(model.state_dict(), model_final_path)

  print(f'Model saved: {model_final_path}')

def check_dataset_balanced(dataset_path):
  high_valence_count = 0
  low_valence_count = 0
  high_arousal_count = 0
  low_arousal_count = 0

  ls = os.listdir(dataset_path)

  if '.DS_Store' in ls:
    ls.remove('.DS_Store')

  for filepath in ls:
    with open(os.path.join(dataset_path, filepath), mode='rb') as f:
      data = pickle.load(f)
      _, labels = data['data'], data['labels']

      if labels[0] > 5.0:
        high_valence_count += 1
      else:
        low_valence_count += 1
      
      if labels[1] > 5.0:
        high_arousal_count += 1
      else:
        low_arousal_count += 1

  print(f'Total examples = {len(ls)}')
  print(f'High valence = {high_valence_count} ({(high_valence_count/len(ls)*100):.2f}%),', end=' ')
  print(f'Low valence = {low_valence_count} ({(low_valence_count/len(ls)*100):.2f}%)')
  print(f'High arousal = {high_arousal_count} ({(high_arousal_count/len(ls)*100):.2f}%),', end=' ')
  print(f'Low arousal = {low_arousal_count} ({(low_arousal_count/len(ls)*100):.2f}%)')

def check_train_test_split_balanced(train_loader, test_loader):
  train_set = train_loader.dataset
  test_set = test_loader.dataset
  
  for loader in [train_loader, test_loader]:
    num_high = 0
    for data, labels in loader:
      labels = labels[:,0]
      num_high += torch.sum(labels)

    if loader == train_loader:
      print(f'Train set: L // H = {len(train_set)-num_high} // {num_high} = {((len(train_set)-num_high)/len(train_set)*100):.2f}% // {(num_high/len(train_set)*100):.2f}%')
    else:
      print(f'Test  set: L // H = {len(test_set)-num_high} // {num_high} = {((len(test_set)-num_high)/len(test_set)*100):.2f} // {(num_high/len(test_set)*100):.2f}%')

def count_model_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)