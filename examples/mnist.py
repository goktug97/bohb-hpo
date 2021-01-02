import requests
import os
import hashlib
import tempfile
import gzip

import torch
import numpy as np

from bohb import BOHB
import bohb.configspace as cs

SEED = 123


# https://github.com/geohot/tinygrad/blob/master/extra/utils.py
def fetch(url):
    fp = os.path.join(tempfile.gettempdir(),
                      hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp) and os.stat(fp).st_size > 0:
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        print("fetching %s" % url)
        dat = requests.get(url).content
        with open(fp+".tmp", "wb") as f:
            f.write(dat)
        os.rename(fp+".tmp", fp)
    return dat


def parse_mnist(data):
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


def fetch_mnist():
    X = parse_mnist(
        fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
    )[0x10:].reshape((-1, 28, 28))
    Y = parse_mnist(fetch(
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))[8:]
    idx = np.arange(0, 2048+128)
    np.random.shuffle(idx)
    X_train = X[idx[:2048]].reshape(-1, 28*28)
    Y_train = Y[idx[:2048]]
    X_test = X[idx[2048:]].reshape(-1, 28*28)
    Y_test = Y[idx[2048:]]
    return X_train, Y_train, X_test, Y_test


class MNIST(torch.nn.Module):
    def __init__(self, hidden_size, activation):
        super().__init__()
        self.l1 = torch.nn.Linear(784, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        if activation == 'relu':
            self.a = torch.nn.functional.relu
        elif activation == 'tanh':
            self.a = torch.tanh
        elif activation == 'lrelu':
            self.a = torch.nn.functional.leaky_relu
        else:
            raise NotImplementedError
        self.output = torch.nn.Linear(hidden_size, 10)

    def forward(self, x):
        return self.output(self.a(self.l2(self.a(self.l1(x)))))


def train_mnist(batch_size, n_epochs, optimizer, hidden_size, scheduler_p,
                activation, lr, weight_decay, momentum,
                x_train, y_train, x_test, y_test):
    torch.manual_seed(SEED)
    x_train = torch.from_numpy(x_train.copy()).cpu().float()
    y_train = torch.from_numpy(y_train.copy()).cpu().long()
    x_test = torch.from_numpy(x_test.copy()).cpu().float()
    y_test = torch.from_numpy(y_test.copy()).cpu().long()
    model = MNIST(hidden_size, activation)

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=weight_decay, lr=lr)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    weight_decay=weight_decay, lr=lr,
                                    momentum=momentum)
    elif optimizer == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        weight_decay=weight_decay, lr=lr,
                                        momentum=momentum)
    else:
        raise NotImplementedError

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs)

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(int(n_epochs)):
        for batch_idx in range(len(x_train) // batch_size):
            batch = x_train[
                batch_idx * batch_size:(batch_idx + 1) * batch_size]
            output = model(batch)
            loss = loss_fn(output, y_train[
                batch_idx * batch_size:(batch_idx + 1) * batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch}, train loss:{loss:.5f}")
        output = model(x_test)
        loss = loss_fn(output, y_test)
        acc = (output.argmax(axis=1) == y_test).float().sum()/y_test.shape[0]
        print(f"epoch: {epoch}, validation loss:{loss:.5f}, accuracy:{acc}")
        if scheduler_p:
            scheduler.step()
    return loss.item()


if __name__ == '__main__':
    np.random.seed(SEED)
    x_train, y_train, x_test, y_test = fetch_mnist()


    def evaluate(params, budget):
        loss = train_mnist(**params, x_train=x_train, y_train=y_train,
                           x_test=x_test, y_test=y_test, n_epochs=budget)
        return loss


    batch_size = cs.CategoricalHyperparameter('batch_size', [8, 16, 32])
    optimizer = cs.CategoricalHyperparameter('optimizer', ['adam', 'sgd', 'rms'])
    momentum = cs.UniformHyperparameter(
        'momentum', 0, 1, (optimizer=='sgd') | (optimizer=='rms'))
    not_momentum = cs.UniformHyperparameter('momentum', 0, 0, ~momentum.cond)
    hidden_size = cs.CategoricalHyperparameter('hidden_size', [16, 32, 64])
    scheduler_p = cs.CategoricalHyperparameter('scheduler_p', [False, True])
    learning_rate = cs.UniformHyperparameter('lr', 1e-4, 1e-1, log=True)
    activation = cs.CategoricalHyperparameter(
        'activation', ['relu', 'lrelu', 'tanh'])
    regularization_p = cs.CategoricalHyperparameter(
        'regularization_p', [False, True], dont_pass=True)
    weight_decay = cs.UniformHyperparameter(
        'weight_decay', 0, 1e-3, regularization_p == True)
    not_weight_decay = cs.UniformHyperparameter(
        'weight_decay', 0, 0, ~weight_decay.cond)

    configspace = cs.ConfigurationSpace([batch_size, optimizer, hidden_size,
                                         scheduler_p, activation, weight_decay,
                                         not_momentum, not_weight_decay,
                                         regularization_p, momentum, learning_rate],
                                        seed=SEED)

    opt = BOHB(configspace, evaluate, max_budget=81, min_budget=1, n_proc=2)
    logs = opt.optimize()
    print(logs)
