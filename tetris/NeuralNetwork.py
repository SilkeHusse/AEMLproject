from pathlib import Path
import requests
import pickle
import gzip

### MNIST data setup ###

#DATA_PATH = Path("data")
#PATH = DATA_PATH / "mnist"

#PATH.mkdir(parents=True, exist_ok=True)

#URL = "http://deeplearning.net/data/mnist/"
#FILENAME = "mnist.pkl.gz"

#if not (PATH / FILENAME).exists():
#        content = requests.get(URL + FILENAME).content
#        (PATH / FILENAME).open("wb").write(content)

#with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
#        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

########################################################################################

import torch

# convert data to tensors

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
#x_train, x_train.shape, y_train.min(), y_train.max()
#print(x_train, y_train)
#print(x_train.shape)
#print(y_train.min(), y_train.max())

########################################################################################

#import math
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

bs = 64  # batch size
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

loss_func = F.cross_entropy # LogLikelihood loss + LogSoftmax activation function

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
