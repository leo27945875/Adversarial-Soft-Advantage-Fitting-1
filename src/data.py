import numpy as np
import pickle
import random

import torch
from torch.utils.data import Dataset


class Buffer(Dataset):
    def __init__(self, nTransition, device="cpu"):
        self.device      = device
        self.transitions = [None for _ in range(nTransition)]
        self.pointer     = 0
    
    def __getitem__(self, i):
        if isinstance(i, slice):
            return [torch.tensor(data).to(self.device).float() for data in zip(*self.transitions[i])]

        state, action = self.transitions[i]
        return (
            torch.from_numpy(state ).to(self.device).float(), 
            torch.from_numpy(action).to(self.device).float()
        )
    
    def __len__(self):
        return len(self.transitions)

    def Sample(self, num):
        transitions = random.sample(self.transitions, num)
        return [torch.tensor(data).to(self.device).float() for data in zip(*transitions)]
    
    def Push(self, state, action):
        self.transitions[self.pointer] = (state, action)
        self.pointer += 1
        if self.pointer >= len(self.transitions):
            self.pointer = 0
    
    def Clear(self):
        for i in range(len(self.transitions)):
            self.transitions[i] = None
    
    def IsFull(self):
        return self.transitions[-1] is not None


class ExpertBuffer(Buffer):
    def __init__(self, path, numData=None, device="cpu"):
        super().__init__(0, device)
        self.transitions = self.LoadData(path, numData)
    
    def LoadData(self, path, numData=None):
        with open(path, "rb") as f:
            data = pickle.load(f)
    
        if numData is None:
            return data
        else:
            return data[-numData:]


class AgentBuffer(Buffer):
    pass


def Preprocess():
    pass


if __name__ == "__main__":
    trans = [
        [np.array([2.1, 3.5, 6.7]), np.array([2.8, 4.1, 3.5, 4.3])],
        [np.array([2.4, 4.3, 5.5]), np.array([2.4, 5.4, 4.5, 4.6])],
        [np.array([3.3, 3.8, 1.5]), np.array([2.9, 5.0, 4.0, 7.6])],
        [np.array([5.6, 6.2, 9.9]), np.array([2.7, 1.3, 5.4, 3.4])]
    ]
    device = torch.device("cuda:0")
    
    aBuffer = AgentBuffer(len(trans), device)
    for tran in trans:
        aBuffer.Push(*tran)
        print(aBuffer.IsFull())

    print(aBuffer.Sample(3))














