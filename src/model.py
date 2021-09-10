import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.utils import clip_grad_norm_
from warmup_scheduler import GradualWarmupScheduler

from utils import GetTrainIteration, RandomTrainIteration, GetLR


INF  = float("inf")
EPS  = 1e-8
EPS2 = 1e-4


def InitializeLinear(module, mul=1.):
    module.weight.data.mul_(mul)
    module.bias.data.zero_()


def GetScheduler(name=None, optimizer=None, schedulerWarmup=0, **kwargs):
    if not name or not optimizer:
        return None
    else:
        if schedulerWarmup > 0:
            return GradualWarmupScheduler(optimizer, 1, schedulerWarmup, torch.optim.lr_scheduler.__getattribute__(name)(optimizer, **kwargs))
        else:
            return torch.optim.lr_scheduler.__getattribute__(name)(optimizer, **kwargs)


def GetOptimizer(name=None, params={}, lr=1e-3, **kwargs):
    if not name:
        return None
    else: 
        try:
            return torch.optim.__getattribute__(name)(params, lr, **kwargs)
        except:
            import torch_optimizer
            return torch_optimizer.__getattribute__(name)(params, lr, **kwargs)


class GaussianPolicy(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenDim=256, squashing=None):
        super().__init__()
        self.fc0   = nn.Linear(stateDim, hiddenDim)
        self.fc1   = nn.Linear(hiddenDim, hiddenDim)
        self.mu    = nn.Linear(hiddenDim, actionDim)
        self.sigma = nn.Linear(hiddenDim, actionDim)
        self.Initialize()

        self.squashing = squashing
    
    def forward(self, state):
        h = F.relu(self.fc0(state))
        h = F.relu(self.fc1(h))
        return self.mu(h), torch.clamp(self.sigma(h), -20, 2).exp()
    
    def Initialize(self):
        InitializeLinear(self.fc0, 1)
        InitializeLinear(self.fc1, 1)
        InitializeLinear(self.mu, 0.1)
        InitializeLinear(self.sigma, 0.1)
        
    def GetLogProb(self, state, action):
        if self.squashing is None:
            mean, std = self(state)
            logProb   = Normal(mean, std).log_prob(action)
            logProb   = torch.sum(logProb, dim=-1, keepdim=True)
            return logProb
        elif self.squashing == "tanh":
            mean, std  = self(state)
            actionTemp = torch.atanh(torch.clamp(action, -1 + EPS2, 1 - EPS2))
            logProb    = Normal(mean, std).log_prob(actionTemp)
            logProb   -= torch.log(1 - action ** 2 + EPS)
            logProb    = torch.sum(logProb, dim=-1, keepdim=True)
            return logProb
        else:
            raise ValueError(f"Invalid squashing function {self.squashing} !")
    
    # state: np.array (shape=(stateDim, ))
    def OneStepAction(self, state):
        if self.squashing is None:
            state   = torch.tensor(state).to(self.fc0.weight.device).float().unsqueeze(0)
            mean, _ = self(state)
            return mean.squeeze(0).detach().cpu().numpy()
        elif self.squashing == "tanh":
            state   = torch.tensor(state).to(self.fc0.weight.device).float().unsqueeze(0)
            mean, _ = self(state)
            return torch.tanh(mean).squeeze(0).detach().cpu().numpy()
        else:
            raise ValueError(f"Invalid squashing function {self.squashing} !")
        

class ASAF1:
    def __init__(self, stateDim, actionDim, hiddenDim, minActVal, maxActVal, optimizer="Adam", scheduler=None, 
                 lr=1e-3, gradClip=INF, squashing=None, schedulerWarmup=0, optimizerParams={}, schedulerParams={}):
        self.minActVal = minActVal
        self.maxActVal = maxActVal
        self.gradClip  = gradClip
        self.policy    = GaussianPolicy(stateDim, actionDim, hiddenDim, squashing)
        self.optimizer = GetOptimizer(optimizer, self.policy.parameters(), lr, **optimizerParams)
        self.scheduler = GetScheduler(scheduler, self.optimizer, schedulerWarmup, **schedulerParams)
    
    # One step update of the policy:
    def UpdatePolicy(self, expertState, expertAction, expertOldProb, agentState, agentAction, agentOldProb):
        expertLogProb = self.policy.GetLogProb(expertState, expertAction)
        agentLogProb  = self.policy.GetLogProb(agentState , agentAction )

        expertLoss = -(expertLogProb - torch.log(expertLogProb.exp() + expertOldProb.exp() + EPS)).mean()
        agentLoss  = -(agentOldProb  - torch.log(agentLogProb .exp() + agentOldProb .exp() + EPS)).mean()
        
        loss = expertLoss + agentLoss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy.parameters(), self.gradClip)
        self.optimizer.step()

        return expertLoss.item(), agentLoss.item()
    
    # Update learning rate scheduler:
    def UpdateScheduler(self, objective=None):
        if self.scheduler:
            if objective is None:
                self.scheduler.step()
            else:
                self.scheduler.step(objective)
            
        return GetLR(self.optimizer)

    # Get initial probability of the action vector (πG):
    def GetInitProb(self, expertState, expertAction, agentState, agentAction, batchSize):
        device, nExpertTransition, nAgentTransition = expertState.device, expertState.size(0), agentState.size(0)
        with torch.no_grad():
            expertOldProb = torch.zeros([nExpertTransition, 1], device=device)
            for s, e in GetTrainIteration(nExpertTransition, batchSize):
                expertOldProb[s: e] = self.policy.GetLogProb(expertState[s: e], expertAction[s: e])
                
            agentOldProb  = torch.zeros([nAgentTransition, 1], device=device)
            for s, e in GetTrainIteration(nAgentTransition, batchSize):
                agentOldProb [s: e] = self.policy.GetLogProb(agentState [s: e], agentAction [s: e])
        
        return expertOldProb, agentOldProb
    
    # action: [min, max] -> [-1, 1]
    def MapAction(self, action):
        maxVal , minVal   = self.maxActVal, self.minActVal
        toMinus, toDivide = (maxVal + minVal) * 0.5, (maxVal - minVal) * 0.5
        return (action -  toMinus) / toDivide
    
    # action: [-1, 1] -> [min, max]
    def RecoverAction(self, action):
        maxVal, minVal     = self.maxActVal, self.minActVal
        toAdd , toMultiply = (maxVal + minVal) * 0.5, (maxVal - minVal) * 0.5
        return action * toMultiply + toAdd

    # Update the policy for many times:
    def Fit(self, expertState, expertAction, agentState, agentAction, epochs, batchSize):
        nExpertTransition, nAgentTransition = expertState.size(0), agentState.size(0)
        batchLength = nAgentTransition // batchSize + int(nAgentTransition % batchSize != 0)
        if self.optimizer:
            expertAction  , agentAction   = self.MapAction(expertAction), self.MapAction(agentAction)
            expertOldProb , agentOldProb  = self.GetInitProb(expertState, expertAction, agentState, agentAction, batchSize)
            expertLossList, agentLossList = [], []
            for _ in range(epochs):
                agentBatchWindows  = GetTrainIteration(nAgentTransition, batchSize)
                expertBatchWindows = RandomTrainIteration(nExpertTransition, batchSize, batchLength)
                for (sE, eE), (sA, eA) in zip(expertBatchWindows, agentBatchWindows):
                    expertLoss, agentLoss = self.UpdatePolicy(
                        expertState[sE: eE], expertAction[sE: eE], expertOldProb[sE: eE],
                        agentState [sA: eA], agentAction [sA: eA], agentOldProb [sA: eA]
                    )
                    expertLossList.append(expertLoss)
                    agentLossList .append(agentLoss )

            return sum(expertLossList) / len(expertLossList), sum(agentLossList) / len(agentLossList)
        else:
            raise Exception("There is no optimizer, so we cannot update policy !")
    
    # state: np.array (shape=(stateDim, ))
    def Act(self, state):
        action = self.policy.OneStepAction(state)
        return self.RecoverAction(action)
    
    def SetDevice(self, device):
        self.policy.to(device)
    
    def ToTrainMode(self):
        self.policy.train()
    
    def ToEvalMode(self):
        self.policy.eval()
    
    def Save(self, path):
        while os.path.exists(path):
            path, ext = os.path.splitext(path)
            path += ('_new' + ext)
        
        checkpoint = {
            "policy"   : self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else {},
            "scheduler": self.scheduler.state_dict() if self.scheduler else {}
        }
        torch.save(checkpoint, path)
    
    def Load(self, path, isLoadOptimizer=False):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        if isLoadOptimizer:
            if self.optimizer: self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler: self.scheduler.load_state_dict(checkpoint["scheduler"])


if __name__ == '__main__':
    import gym
    import pybullet_envs

    env = gym.make("InvertedDoublePendulumBulletEnv-v0")
    print(env.observation_space.shape)
    print(env.action_space.shape)
    print(env.action_space.low)
    print(env.action_space.high)
    print(env.reward_range)






