import gym
import pybullet_envs
import glob
from tqdm import trange

from model import ASAF1
from utils import SeedEverything

import torch


def Test(agent, env, nEpisode, isRender=False, isEnvClose=False, minReward=-float("inf")):
    agent.ToEvalMode()
    anyTooSmall = False
    with torch.no_grad():
        totalReward = 0
        for e in range(nEpisode):
            episodeReward = 0
            state, done = env.reset(), False
            while not done:
                if isRender: env.render()
                action = agent.Act(state)
                state, reward, done, _ = env.step(action)
                episodeReward += reward
            
            totalReward += episodeReward
            anyTooSmall  = episodeReward < minReward
    
    agent.ToTrainMode()
    if isEnvClose: env.close()
    return totalReward / nEpisode, anyTooSmall


def TestGenerally(modelPath, envName, testNumSeed=50, hiddenDim=256):
    env       = gym.make(envName)
    stateDim  = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    maxAction = env.action_space.high[0]
    minAction = env.action_space.low[0]

    agent = ASAF1(stateDim, actionDim, hiddenDim, minAction, maxAction)
    agent.Load(modelPath)

    rewardList = []
    for seed in trange(testNumSeed):
        SeedEverything(seed, env)
        rewardList.append(Test(agent, env, 1, isEnvClose=False)[0])

    avgReward, minReward, maxReward = sum(rewardList) / testNumSeed, min(rewardList), max(rewardList)
    print("=" * 100)
    print(f"Average Test Reward = {avgReward :.2f}")
    print(f"Minimum Test Reward = {maxReward :.2f} (index = {rewardList.index(maxReward)})")
    print(f"Maximum Test Reward = {minReward :.2f} (index = {rewardList.index(minReward)})")
    env.close()
    return avgReward, rewardList


def TestAll(envName, hiddenDim, testNumSeed=50):
    modelPaths = glob.glob(f"../model/ASAF1_{hiddenDim}_{envName}_*.pth")
    bestReward, bestModelPath = -float("inf"), None
    for modelPath in modelPaths:
        print("-" * 50 + f"\n[{modelPath}]")
        avgReward, _ = TestGenerally(modelPath, envName, testNumSeed, hiddenDim)
        if avgReward > bestReward:
            bestReward = avgReward
            bestModelPath = modelPath
    
    print("\n" + "=" * 70)
    print(f"Best Model: [{bestModelPath}]")
    print(f"Best Reward = {bestReward :.2f}")
    print("")


if __name__ == '__main__':

    TestAll(
        envName    = "Ant-v2",
        hiddenDim  = 256
    )







