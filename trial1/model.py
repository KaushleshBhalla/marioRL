import torch
import torch.nn as nn
import os

class AtariNet(nn.Module):
    def __init__(self,num_actions=2):
        super().__init__()
        self.relu=nn.ReLU()

        self.conv1=nn.Conv2d(1,32,kernel_size=(8,8),stride=(4,4))
        self.conv2=nn.Conv2d(32,64,kernel_size=(4,4),stride=(2,2))
        self.conv3=nn.Conv2d(64,64,kernel_size=(3,3),stride=(1,1))
        self.flatten=nn.Flatten()
        self.dropout=nn.Dropout(p=0.15)

        self.action_value1=nn.Linear(3136,1024)
        self.action_value2=nn.Linear(1024,1024)
        self.action_value3=nn.Linear(1024,num_actions)

        self.state_value1=nn.Linear(3136,1024)
        self.state_value2=nn.Linear(1024,1024)
        self.state_value3=nn.Linear(1024,1)
    def forward(self,X):
        X=self.relu(self.conv1(X))
        X=self.relu(self.conv2(X))
        X=self.relu(self.conv3(X))
        X=self.flatten(X)

        state_value=self.relu(self.state_value1(X))
        state_value=self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)
        state_value = self.state_value3(state_value)

        action_value=self.relu(self.action_value1(X))
        action_value=self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.action_value3(action_value)

        output=state_value+(action_value-action_value.mean())

        return output
    def save_model(self, weights_filename='weights/latest.pt'):
        # Take the default weights filename(latest.pt) and save it
        torch.save(self.state_dict(), weights_filename)


    def load_model(self, weights_filename='weights/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")
        except:
            print(f"No weights file available at {weights_filename}")