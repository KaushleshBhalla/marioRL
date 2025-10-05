
import copy
import random
import torch
import torch.optim as optim
import torch.nn.functional as f
from plot import LivePlot
import numpy as np
import time
class ReplayMemory:
    def __init__(self,capacity,device='cpu'):
        self.capacity=capacity
        self.memory=[]
        self.device=device
        self.position=0
        self.memory_max_report=0

    def insert(self,transition):
        transition=[item.to('cpu') for item in transition]
        if(len(self.memory)<self.capacity):
            self.memory.append(transition)
        else:
            self.memory.pop(0)
            self.memory.append(transition)
    def sample(self,batch_size=32):
        assert self.can_sample(batch_size)

        batch=random.sample(self.memory,batch_size)
        batch=zip(*batch)
        return [torch.cat(items).to(self.device) for items in batch] # [[states],[actions],[rewards],[next state]], length of each list=batch_size
    def can_sample(self,batch_size=32):
        if(len(self.memory)>=batch_size*5):
            return True
        else:
            return False
    def __len__(self):
        return len(self.memory)
class Agent:
    def __init__(self,model,device='cpu',epsilon=1.0,min_epsilon=0.08,number_warmup=10000,number_actions=None,memory_capacity=100000,batch_size=32
                 ,learning_rate=0.00025):
        self.memory=ReplayMemory(device=device,capacity=memory_capacity)
        self.model=model
        self.target_model=copy.deepcopy(model).eval()
        self.epsilon=epsilon
        self.min_epsilon=min_epsilon
        self.epsilon_decay=1-((epsilon-min_epsilon)*2/number_warmup)
        self.batch_size=batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma=0.99
        self.number_actions=number_actions
        self.optimizer=optim.Adam(model.parameters(),lr=learning_rate)
        print(f"starting epsilon is {self.epsilon}")
        print(f"epsilon decay is {self.epsilon_decay}")
    def get_action(self,state):
        if torch.rand(1)<self.epsilon:
            return torch.randint(0,self.number_actions,(1,1))
        else:
            av=self.model(state).detach()
            return torch.argmax(av,dim=1,keepdim=True)
    def train(self,env,epochs,initial_bias):
        stats={'Returns':[],"AvgReturns":[],"EpsilonCheckpoint":[]}
        plotter=LivePlot()

        for epoch in range(epochs):
            state=env.reset()
            env.lives=2
            env.stage=1
            done=False
            ep_return=0
            # i=0e
            while not done:
                action=self.get_action(state)
                next_state,reward,done,info=env.step(action)
                self.memory.insert([state,action,reward,done,next_state])
                if self.memory.can_sample(self.batch_size):
                    state_b,action_b,reward_b,done_b,next_state_b=self.memory.sample(self.batch_size)
                    predicted_q_value_b=self.model(state_b).gather(1,action_b)
                    next_q_value_b=self.target_model(next_state_b)
                    next_q_value_b=torch.max(next_q_value_b,dim=-1,keepdim=True)[0]
                    target_q_value_b=reward_b+ ~done_b*self.gamma*next_q_value_b
                    loss=f.mse_loss(predicted_q_value_b,target_q_value_b)
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                state=next_state
                ep_return+=reward.item()
                if info.get("stage",2)==2:
                    break
                # if i%100==0:
                #     print(ep_return)
                # i+=1
            stats['Returns'].append(ep_return)
            if self.epsilon>self.min_epsilon:
                self.epsilon*=self.epsilon_decay
            if epoch%10==1:
                self.model.save_model()
            if epoch%2==0:
                average_returns=np.mean(stats["Returns"][-100:])
                stats['AvgReturns'].append(average_returns)
                stats['EpsilonCheckpoint'].append(self.epsilon)
                if len(stats['Returns'])>100:
                    print(f"epoch:{epoch},Average Returns:{stats["AvgReturns"][-1]},Return:{stats["Returns"][-1]},epsilon:{self.epsilon}")
                else:
                    print(
                        f"epoch:{epoch},Average Returns:{np.mean(stats["Returns"][:len(stats["Returns"])])},Return:{stats["Returns"][-1]},epsilon:{self.epsilon}")
            if epoch%50==0:
                self.target_model.load_state_dict(self.model.state_dict())
                plotter.update_plot(stats,50000)
            if epoch%100==1:
                self.model.save_model(f'model_itter/{epoch+initial_bias}.pt')

    def test(self,env):
        for epoch in range(10):
            state=env.reset()
            done=False
            i=0
            total_reward=0
            while not done:
                time.sleep(0.01)
                action=self.get_action(state)
                state,reward,done,info=env.step(action)
                if i%10==0:
                    print(info,total_reward)
                    total_reward=0
                else:
                    total_reward+=reward
                i+=1
                if done:
                    break
