import os
import gym
import pybullet_envs

import torch

from model import ASAF1
from data import ExpertBuffer, AgentBuffer
from utils import SeedEverything, SavePickle
from test import Test


ENVIRONMENT_NAME      = "Ant-v2"
RANDOM_SEED           = 87

EXPERT_DEMO_PATH      = "../data/Ant-v2_trajectory_R=5000.pkl"
MODEL_SAVE_FOLDER     = "../model"
HISTORY_SAVE_FOLDER   = "../history"

MAX_NUM_EXPERT_DEMO   = 25000
MAX_NUM_TRANSITION    = 2000000
NUM_TRANSITION_UPDATE = 4000
EPOCHS_PER_UPDATE     = 10
UPDATE_BATCH_SIZE     = 256

GRADIENT_CLIPPING     = 1.
LEARNING_RATE         = 1e-3
OPTIMIZER_NAME        = "Adam"
OPTIMIZER_PARAMETERS  = {}
SCHEDULER_NAME        = "StepLR"
SCHEDULER_PARAMETERS  = {"step_size": float("inf"), "gamma": 1}
SCHEDULER_WARMUP      = 30

NETWORK_HIDDEN_DIM    = 128
SQUASHING_FUNCTION    = None

IS_TEST_AGENT         = False
IS_EARLY_STOPPING     = False
TEST_NUM_EPISODE      = 5
CAN_TEST_REWARD       = 0
END_TRAIN_REWARD      = 2000


def Train(expertDemoPath, maxExpertDemo, envName, endTrainReward, maxTrans, numTransUpdate, epochsPerUpdate, batchSize, 
          gradClip, lr, optimizer, optimizerParams, scheduler, schedulerParams, schedulerWarmup, hiddenDim, squashing, 
          canTestReward, isTest, numTestEpisode, isEarlyStop, modelSaveFolder, historySaveFolder, seed):
    # Environment:
    env       = gym.make(envName)
    stateDim  = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    maxAction = env.action_space.high[0]
    minAction = env.action_space.low[0]

    print("=" * 40 + f"  {envName}  " + "=" * 40)
    print(f"State  dim   = {stateDim} ")
    print(f"Action dim   = {actionDim}")
    print(f"Max action   = {maxAction :.5f}")
    print(f"Min action   = {minAction :.5f}")
    print(f"Reward range = {env.reward_range}")
    print("=" * 100)

    # Random seed:
    SeedEverything(seed, env)

    # Device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transition buffers:
    expertBuffer = ExpertBuffer(expertDemoPath, maxExpertDemo, device)
    agentBuffer  = AgentBuffer (numTransUpdate, device)


    # Agent:
    agent = ASAF1(stateDim, actionDim, hiddenDim, minAction, maxAction, optimizer, scheduler, lr, 
                  gradClip, squashing, schedulerWarmup, optimizerParams, schedulerParams)
    agent.SetDevice(device)
    agent.ToTrainMode()

    # History:
    history = {
        "Transition": [],
        "Episode"   : [],
        "Reward"    : [],
        "TestReward": [],
        "LR"        : []
    }

    # Training process:
    state, nowLR = env.reset(), 0.
    expertStates   , expertActions = expertBuffer[:]
    totalEpisode   , totalNumTrans, totalReachGoalTimes = 0, 0, 0
    episodeNumTrans, episodeReward, ewmaReward, testReward = 0, 0, 0, 0
    while totalNumTrans < maxTrans:
        # Get one transition of the agent:
        action = agent.Act(state)
        nextState, reward, done, _ = env.step(action)
        agentBuffer.Push(state, action)

        # Record something:
        episodeReward   += reward
        episodeNumTrans += 1
        totalNumTrans   += 1

        # When the episode is done:
        if done:
            # Print training message:
            totalEpisode += 1
            ewmaReward    = ewmaReward * 0.95 + episodeReward * 0.05
            print(f"| Epi: {totalEpisode} | Total trans: {totalNumTrans :7d} | Epi trans: {episodeNumTrans :5d} | Epi Reward: {episodeReward :.2f} | EWMA Reward: {ewmaReward :.2f} | LR: {nowLR :.6f} |", end="")
            
            # Test agent:
            if isTest and episodeReward >= canTestReward and totalNumTrans >= maxTrans // 2:
                testReward, anyTooSmall = Test(agent, env, numTestEpisode, False, False, endTrainReward * 0.9)
                print(f" => Test Reward = {testReward :.2f}")
                if testReward >= endTrainReward and not anyTooSmall:
                    agent.Save(os.path.join(modelSaveFolder  , f"ASAF1_{hiddenDim}_{envName}_E={totalEpisode}_R={round(testReward)}.pth"))
                    totalReachGoalTimes += 1
                    if isEarlyStop and totalReachGoalTimes >= 10: break

            else:
                print("")

            # Record history:
            history["Transition"].append(totalNumTrans)
            history["Episode"   ].append(totalEpisode)
            history["Reward"    ].append(episodeReward)
            history["TestReward"].append(testReward)
            history["LR"        ].append(nowLR)

            # Reset environment:
            episodeNumTrans, episodeReward = 0, 0
            state = env.reset()
        else:
            state = nextState
        
        # Update policy:
        if agentBuffer.IsFull():
            agentStates , agentActions  = agentBuffer[:]
            agent.Fit(expertStates, expertActions, agentStates, agentActions, epochsPerUpdate, batchSize)
            agentBuffer.Clear()
            nowLR = agent.UpdateScheduler(objective=ewmaReward if scheduler == "ReduceLROnPlateau" else None)
    
    # Save model and history:
    modelSavePath   = os.path.join(modelSaveFolder  , f"ASAF1_{hiddenDim}_{envName}_R={round(testReward)}.pth")
    historySavePath = os.path.join(historySaveFolder, f"ASAF1_{hiddenDim}_{envName}_R={round(testReward)}.pkl")
    agent.Save(modelSavePath)
    SavePickle(history, historySavePath)

    # Close environment:
    env.close()
        
        
if __name__ == '__main__':
    Train(
        EXPERT_DEMO_PATH,
        MAX_NUM_EXPERT_DEMO,
        ENVIRONMENT_NAME,
        END_TRAIN_REWARD,
        MAX_NUM_TRANSITION,
        NUM_TRANSITION_UPDATE,
        EPOCHS_PER_UPDATE, 
        UPDATE_BATCH_SIZE,
        GRADIENT_CLIPPING,
        LEARNING_RATE,
        OPTIMIZER_NAME,
        OPTIMIZER_PARAMETERS,
        SCHEDULER_NAME,
        SCHEDULER_PARAMETERS,
        SCHEDULER_WARMUP,
        NETWORK_HIDDEN_DIM,
        SQUASHING_FUNCTION,
        CAN_TEST_REWARD,
        IS_TEST_AGENT,
        TEST_NUM_EPISODE,
        IS_EARLY_STOPPING,
        MODEL_SAVE_FOLDER,
        HISTORY_SAVE_FOLDER,
        RANDOM_SEED
    )