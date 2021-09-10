import os
import copy
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch


def PlotScheduler(scheduler, n_iter=1000, title="LR Scheduler"):
    scheduler = copy.deepcopy(scheduler)
    scheduler.verbose = False
    for i in range(n_iter):
        lr = scheduler.get_last_lr()
        plt.plot(i, lr, 'bo')
        scheduler.step()
    
    plt.title(title)
    plt.show()
    scheduler.last_epoch = -1


def SeedEverything(seed=42, env=None):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if env is not None:
        env.seed(seed)


def GetTrainIteration(n, batchSize):
    left, start = n, 0
    while left > 0:
        thisBatchSize = min(batchSize, left)
        end = start + thisBatchSize
        yield start, end
        left -= thisBatchSize
        start = end


def RandomTrainIteration(n, batchSize, batchLength):
    iteration = GetTrainIteration(n, batchSize)
    choices = random.sample(list(iteration), k=batchLength)
    for choice in choices:
        yield choice


def GetLR(optimizer):
    for p in optimizer.param_groups:
        return p['lr']


def SavePickle(obj, path):
    while os.path.exists(path):
        path, ext = os.path.splitext(path)
        path += ('_new' + ext)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    a = [1,2,3]
    print(a[:None])
