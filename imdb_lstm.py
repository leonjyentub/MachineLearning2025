import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
device = torch.device('cpu')
# -----------------------------------------------------------
class LSTM_Net(nn.Module):
  def __init__(self):
    # vocab_size = 129892
    super(LSTM_Net, self).__init__()
    self.embed = nn.Embedding(129892, 32)
    self.lstm = nn.LSTM(32, 100)
    self.fc1 = nn.Linear(100, 2)  # 0=neg, 1=pos
 
  def forward(self, x):
    # x = review/sentence. length = fixed w/ padding
    z = self.embed(x)  # x can be arbitrary shape - not
    z = z.reshape(50, -1, 32)  # seq bat embed
    lstm_oupt, (h_n, c_n) = self.lstm(z)
    z = lstm_oupt[-1]  # or use h_n. [1,100]
    z = torch.log_softmax(self.fc1(z), dim=1)  # NLLLoss()
    return z 

# -----------------------------------------------------------
class IMDB_Dataset(Dataset):
  # 50 token IDs then 0 or 1 label, space delimited
  def __init__(self, src_file):
    all_xy = np.loadtxt(src_file, usecols=range(0,51),
      delimiter=" ", comments="#", dtype=np.int64)
    tmp_x = all_xy[:,0:50]   # cols [0,50) = [0,49]
    tmp_y = all_xy[:,50]     # all rows, just col 50
    self.x_data = torch.tensor(tmp_x, dtype=torch.int64) 
    self.y_data = torch.tensor(tmp_y, dtype=torch.int64) 

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    tokens = self.x_data[idx]
    trgts = self.y_data[idx] 
    return (tokens, trgts)

# -----------------------------------------------------------

def accuracy(model, dataset):
  # data_x and data_y are lists of tensors
  # assumes model.eval()
  num_correct = 0; num_wrong = 0
  ldr = torch.utils.data.DataLoader(dataset,
    batch_size=1, shuffle=False)
  for (batch_idx, batch) in enumerate(ldr):
    X = batch[0]  # inputs
    Y = batch[1]  # target sentiment label
    with torch.no_grad():
      oupt = model(X)  # log-probs
   
    idx = torch.argmax(oupt.data)
    if idx == Y:  # predicted == target
      num_correct += 1
    else:
      num_wrong += 1
  acc = (num_correct * 100.0) / (num_correct + num_wrong)
  return acc

# -----------------------------------------------------------

def main():
  # 0. get started
  print("\nBegin PyTorch IMDB LSTM demo ")
  print("Using only reviews with 50 or less words ")
  torch.manual_seed(1)
  np.random.seed(1)

  # 1. load data 
  print("\nLoading preprocessed train and test data ")
  train_file = ".\\Data\\imdb_train_50w.txt"
  train_ds = IMDB_Dataset(train_file) 

  test_file = ".\\Data\\imdb_test_50w.txt"
  test_ds = IMDB_Dataset(test_file) 

  bat_size = 8
  train_ldr = DataLoader(train_ds,
    batch_size=bat_size, shuffle=True, drop_last=True)
  n_train = len(train_ds)
  n_test = len(test_ds)
  print("Num train = %d Num test = %d " % (n_train, n_test))

# -----------------------------------------------------------

  # 2. create network
  net = LSTM_Net().to(device)

  # 3. train model
  loss_func = torch.nn.NLLLoss()  # log-softmax() activation
  optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)
  max_epochs = 20
  log_interval = 2  # display progress

  print("\nbatch size = " + str(bat_size))
  print("loss func = " + str(loss_func))
  print("optimizer = Adam ")
  print("learn rate = 0.001 ")
  print("max_epochs = %d " % max_epochs)

  print("\nStarting training ")
  net.train()  # set training mode
  for epoch in range(0, max_epochs):
    tot_err = 0.0  # for one epoch
    for (batch_idx, batch) in enumerate(train_ldr):
      X = T.transpose(batch[0], 0, 1)
      Y = batch[1]
      optimizer.zero_grad()
      oupt = net(X)

      loss_val = loss_func(oupt, Y) 
      tot_err += loss_val.item()
      loss_val.backward()  # compute gradients
      optimizer.step()     # update weights
  
    if epoch % log_interval == 0:
      print("epoch = %4d  |" % epoch, end="")
      print("   loss = %10.4f  |" % tot_err, end="")

      net.eval()
      train_acc = accuracy(net, train_ds)
      print("  accuracy = %8.2f%%" % train_acc)
      net.train()
  print("Training complete")

# -----------------------------------------------------------

  # 4. evaluate model
  net.eval()
  test_acc = accuracy(net, test_ds)
  print("\nAccuracy on test data = %8.2f%%" % test_acc)

  # 5. save model
  print("\nSaving trained model state")
  fn = ".\\Models\\imdb_model.pt"
  torch.save(net.state_dict(), fn)


  # 6. use model
  print("\nSentiment for \"the movie was a great waste of my time\"")
  print("0 = negative, 1 = positive ")
  review = np.array([4, 20, 16, 6, 86, 425, 7, 58, 64],
    dtype=np.int64)
  padding = np.zeros(41, dtype=np.int64)
  review = np.concatenate([padding, review])
  review = torch.tensor(review, dtype=torch.int64).to(device)
  
  net.eval()
  with torch.no_grad():
    prediction = net(review)  # log-probs
  print("raw output : ", end=""); print(prediction)
  print("pseud-probs: ", end=""); print(torch.exp(prediction))

  print("\nEnd PyTorch IMDB LSTM sentiment demo")

if __name__ == "__main__":
  main()
