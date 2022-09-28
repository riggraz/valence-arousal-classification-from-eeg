import torch
import numpy as np
from utils import save_model

def check_accuracy(model, data_loader, classification_of='valence', num_examples=-1):
  num_corrects = 0

  if num_examples == -1:
    num_examples = len(data_loader.dataset)

  with torch.no_grad():
    model.eval()

    for data, labels in data_loader:
      preds = model(data)
      preds = torch.squeeze((preds >= 0.0).long())
      labels = labels[:,0] if classification_of == 'valence' else labels[:,1]

      assert preds.shape == labels.shape

      num_corrects += torch.sum((preds == labels).long())

  model.train()

  return (num_corrects / num_examples).item()

def train(
  model,
  train_loader,
  criterion,
  optimizer,
  classification_of='valence',
  num_epochs=100,
  do_check_accuracy=True,
  test_loader=None,
  model_path=None,
  model_name=None,
  check_accuracy_every=50
):
  model.train()

  best_acc_test_set = 0.0

  avg_loss_per_epoch = []
  accuracy_per_epoch = []

  for epoch_n in range(1, num_epochs+1):
    epoch_losses = []

    for batch_i, (data, labels) in enumerate(train_loader, start=1):
      preds = torch.squeeze(model(data))
      labels = labels[:,0].float() if classification_of == 'valence' else labels[:,1].float()

      loss = criterion(preds, labels)
      
      if len(data) == train_loader.batch_size:
        epoch_losses.append(loss.item())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f'\rEPOCH {epoch_n}/{num_epochs}: batch {batch_i}: {loss:.3f}', end='')

    avg_epoch_loss = np.mean(epoch_losses)
    avg_loss_per_epoch.append(avg_epoch_loss)
    print(f' (Avg epoch loss = {avg_epoch_loss:.3f})', end='')

    if do_check_accuracy and epoch_n % check_accuracy_every == 0:
      if test_loader == None or model_path == None or model_name == None:
        assert False

      print('\nChecking accuracy on training set... ', end=' ')
      acc_train_set = check_accuracy(model, train_loader, classification_of=classification_of)
      print(f'{(acc_train_set*100):.2f}%')

      print('Testing accuracy on test set...', end=' ')
      acc_test_set = check_accuracy(model, test_loader, classification_of=classification_of)
      print(f'{(acc_test_set * 100):.2f}%')
      accuracy_per_epoch.append(acc_test_set)

      if acc_test_set > best_acc_test_set:
        save_model(model, model_path=model_path, model_name=model_name)
        best_acc_test_set = acc_test_set

  print('\n')
  
  if best_acc_test_set != 0.0:
    print(f'Best accuracy on test set: {(best_acc_test_set*100):.2f}%')
    return best_acc_test_set