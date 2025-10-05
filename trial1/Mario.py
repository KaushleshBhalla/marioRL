import collections
import cv2
import gym
import numpy as np
import gymnasium as gym
from PIL import Image
import torch
import torchvision.transforms as transforms
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
class DQNBreakout(gym.Env):
    def __init__(self,render_mode='rgb_array',repeat=4,device='cpu'):
        base_env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode=render_mode, apply_api_compatibility=True)
        super().__init__()
        self.env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
        state, info = self.env.reset()
        self.lives = 2
        self.repeat = repeat
        self.frame_buffer = []
        self.device = device
        self.image_shape = (84, 84)
        self.stage=1

    def step(self,action):
        total_reward=0
        done=False

        for i in range(self.repeat):
            # print(action)
            observation,reward,done,truncated,info=self.env.step(int(action[0][0]))
            total_reward+=reward
            total_reward-=0.05 #constant 
            current_lives=info.get("life", 2)
            current_stage=info.get("stage",2)
            if current_lives<self.lives:
                total_reward-=200
                self.lives=current_lives
            if current_stage>self.stage:
                total_reward+=500
                self.stage=current_stage
            # print(f"lives:{self.lives}, total_reward:{total_reward}")
            self.frame_buffer.append(observation)
            if done:
                break
        max_frame=np.max(self.frame_buffer[-2:],axis=0)
        # max_frame=max_frame.to(self.device)
        max_frame=self.process_observation((max_frame))
        max_frame=max_frame.to(self.device)

        total_reward=torch.tensor(total_reward).view(1,-1).float()
        total_reward=total_reward.to(self.device)

        done=torch.tensor(done).view(1,-1)
        done=done.to(self.device)
        return max_frame,total_reward,done,info
    def process_observation(self,observation):
        img=Image.fromarray(observation) #for resizing and grayscaling
        img=img.resize(self.image_shape)
        img=img.convert("L") #grayscaling
        transform = transforms.ToTensor() # normalization+ PIL image -> torch tensor
        observation = transform(img)
        # print(observation.shape)
        observation=observation.unsqueeze(0)
        return observation
    def reset(self):
        self.frame_buffer=[]
        observation,_=self.env.reset()
        observation=self.process_observation(observation)
        observation=observation.to(self.device)
        return observation
    
class DQNBreakoutTesting(gym.Env):
    def __init__(self,render_mode='rgb_array',repeat=4,device='cpu'):
        base_env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode=render_mode, apply_api_compatibility=True)
        super().__init__()
        self.env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
        state, info = self.env.reset()
        self.lives = 2
        self.repeat = repeat
        self.frame_buffer = []
        self.device = device
        self.image_shape = (84, 84)
        self.stage=1

    def step(self,action):
        total_reward=0
        done=False

        for i in range(self.repeat):
            # print(action)
            observation,reward,done,truncated,info=self.env.step(int(action[0][0]))
            total_reward+=reward
            total_reward-=0. #constant 
            current_lives=info.get("life", 2)
            current_stage=info.get("stage",2)
            if current_lives<self.lives:
                total_reward-=200
                self.lives=current_lives
            if current_stage>self.stage:
                total_reward+=500
                break
            # print(f"lives:{self.lives}, total_reward:{total_reward}")
            self.frame_buffer.append(observation)
            if done:
                break
        max_frame=np.max(self.frame_buffer[-2:],axis=0)
        # max_frame=max_frame.to(self.device)
        max_frame=self.process_observation((max_frame))
        max_frame=max_frame.to(self.device)

        total_reward=torch.tensor(total_reward).view(1,-1).float()
        total_reward=total_reward.to(self.device)

        done=torch.tensor(done).view(1,-1)
        done=done.to(self.device)
        return max_frame,total_reward,done,info
    def process_observation(self,observation):
        img=Image.fromarray(observation) #for resizing and grayscaling
        img=img.resize(self.image_shape)
        img=img.convert("L") #grayscaling
        transform = transforms.ToTensor() # normalization+ PIL image -> torch tensor
        observation = transform(img)
        # print(observation.shape)
        observation=observation.unsqueeze(0)
        return observation
    def reset(self):
        self.frame_buffer=[]
        observation,_=self.env.reset()
        observation=self.process_observation(observation)
        observation=observation.to(self.device)
        return observation