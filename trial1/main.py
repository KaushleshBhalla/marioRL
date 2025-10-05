import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import os
from Mario import *
from model import AtariNet
from agent import Agent
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
device='cuda'
environment=DQNBreakout(device=device)
model=AtariNet(num_actions=7)
model.load_model()
model.to(device)
agent=Agent(model=model,
            device=device,
            epsilon=0.1,
            number_warmup=1,
            number_actions=7,
            learning_rate=0.00001,
            memory_capacity=100000,
            batch_size=64)

agent.train(env=environment,epochs=20000,initial_bias=27371)