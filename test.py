import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import os
from Mario import *
from model import AtariNet
from agent import Agent
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = AtariNet(num_actions=7)
model.load_model()
agent = Agent(model=model,
              device=device,
              epsilon=0,
              min_epsilon=0,
              number_warmup=50, # originally 10000
              number_actions=7,
              memory_capacity=100000,
              batch_size=64)
test_environment = DQNBreakoutTesting(device=device, render_mode='human')
agent.test(env=test_environment)