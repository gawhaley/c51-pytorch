import argparse
import gym
import numpy as np
import random
import torch
import torch.optim
import torch.functional as Fnn
import torch.nn as nn
from baselines.common.atari_wrappers import WarpFrame, MaxAndSkipEnv
import model

num_atoms = 51

parser = argparse.ArgumentParser(description='C51-DQN Implementation Using Pytorch')
parser.add_argument('env_name', type=str, help='gym id')
parser.add_argument('--no-cuda', action='store_true', help='use to disable available CUDA')
parser.add_argument('--minibatch-size', type=int, default=32, help='size of minibatch')
parser.add_argument('--total-steps', type=int, default=int(4e7), help='Total steps taken during training')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--render', action='store_true', help='render training environments')
parser.add_argument('--gamma', type=int, default=4, help='number of steps between environment renders')
parser.add_argument('--initial_epsilon', type=float, default=1.0, help='probability of selecting random action')
parser.add_argument('--final_epsilon', type=int, default=0.0001, help='eventual decision randomness')

args = parser.parse_args()

def preprocessImage(img):
    img = np.rollaxis(img, 2, 0) #set to 3 x 210 x 160



#set up environment, initialize model
env = gym.make(args.env_name)
env = MaxAndSkipEnv(env)
env = WarpFrame(env) #84x84 observation space from Mnih et al.
env.reset()
model = model.C51(env.action_space.n, num_atoms)

total_steps = 0
r_t = 0
a_t = np.zeros(env.action_space.shape)
