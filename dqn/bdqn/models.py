import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from replay_buffer import PrioritizedReplayBuffer,ReplayBuffer

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Device:", device)

class BDQ(nn.Module):
    def __init__(self ,
                 input_shape ,
                 num_action_branches ,
                 num_actions_per_branch ,
                 hidden_state_size = 512 ,
                 hidden_value_size = 256 ,
                 shared_representation_size = 256 ,
                 hidden_advantage_size = 128 
                 ):
        
        super().__init__()
        self.input_shape = input_shape
        self.num_action_branches = num_action_branches
        self.num_actions_per_branch = num_actions_per_branch
        self.hidden_state_size = hidden_state_size
        self.hidden_value_size = hidden_value_size
        self.shared_representation_size = shared_representation_size
        self.hidden_advantage_size = hidden_advantage_size

        self.state_net = nn.Sequential(
            nn.Linear(self.input_shape, self.hidden_state_size),
            nn.ReLU(),
            nn.Linear(self.hidden_state_size, self.shared_representation_size)
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.shared_representation_size, self.hidden_value_size),
            nn.ReLU(),
            nn.Linear(self.hidden_value_size , 1)
        )

        self.advantage_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.shared_representation_size, self.hidden_advantage_size),
                nn.ReLU(),
                nn.Linear(self.hidden_advantage_size, self.num_actions_per_branch)
            )
            for _ in range(self.num_action_branches)
        ])
    
    def forward(self,x):
        """This function takes in the state as input and gives out the q values \
            in the shape [B, num_action_branches, num_actions_per_branch]"""
        shared_rep = self.state_net(x)            # [B, shared]
        v = self.value_net(shared_rep)            # [B, 1]

        advs = []
        for head in self.advantage_nets:
            a = head(shared_rep)                  # [B, actions_per_branch]
            a = a - a.mean(dim=1, keepdim=True)   # dueling mean takes mean of each row and subtracts with it
            advs.append(a)

        #Right now the shape of adv is [num_action_branches, B, num_actions_per_branch] we need to convert
        adv = torch.stack(advs, dim=1)      # [B, num_action_branches, num_actions_per_branch]
        
        #Add a dimension to v to make it [B,1,1]
        q = v.unsqueeze(1) + adv            # [B, num_action_branches, num_actions_per_branch]

        return q


class DQNAgent:
    def __init__(self, num_taps_per_filter,model_path, delta=0.01, gamma=0.99, lr=1e-3,replay_buffer_size=int(1e6),prioritised_replay_alpha=0.6,prioritised_replay_beta=0.4,prioritised_replay_epsilon=1e-6):
        
        self.num_taps_per_filter = num_taps_per_filter
        self.num_filters=4
        #We set our state dimension as twice the total number_coefficients as we have real and imaginary parts
        self.num_coeffs=num_taps_per_filter*self.num_filters
        self.state_dim = 2*self.num_coeffs

        self.num_action_branches=self.state_dim
        #The actions are increase,decrease and remain same
        self.num_actions=3
        
        self.gamma = gamma
        self.prioritised_replay_beta=prioritised_replay_beta
        self.prioritised_replay_eps=prioritised_replay_epsilon

        #Number of action branches is same as state_dim
        self.q_net = BDQ(self.state_dim,self.num_action_branches,self.num_actions).to(device)
        self.target_net = BDQ(self.state_dim,self.num_action_branches,self.num_actions).to(device)
        
        try:
            state= torch.load(model_path,map_location=device)
            self.q_net.load_state_dict(state)
        except FileNotFoundError:
            pass

        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.loss_fn = nn.MSELoss(reduction='none')



    def select_action(self, state, epsilon):
        """This function selects an action based on the epsilon-greedy policy
        and returns an tensor of shape num_branches"""
        if random.random() < epsilon:
            actions = torch.randint(
                low=0,
                high=self.num_actions,
                size=(self.num_action_branches,),
                device=device
            )
            return actions
        with torch.no_grad():
            #The recieved state has dimension (24,) we need to make it (1,24) as number of batches is 1
            cur_state=torch.as_tensor(state,dtype=torch.float32,device=device).unsqueeze(0)

            q_vals = self.q_net(cur_state)
            # q_vals: [B, num_action_branches, num_actions_per_branch]
            actions = q_vals.argmax(dim=2, keepdim=True)  # [B, num_action_branches, 1]
            
            #Returns shape [num_action_branches]
            return actions.squeeze()

    def update(self, batch_size=512,GRAD_NORM_CLIP=10):
        if len(self.replay_buffer) < batch_size:
            return None
        #The shapes of states is [B,state_dim],actions is [B, num_action_branches, 1],rewards is [B,1]
        replay_buffer_samples=self.replay_buffer.sample(batch_size)

        states, actions, rewards, next_states= replay_buffer_samples

        states=torch.as_tensor(states,dtype=torch.float32,device=device)
        actions=torch.as_tensor(actions,dtype=torch.int64,device=device)
        rewards=torch.as_tensor(rewards,dtype=torch.float32,device=device)
        next_states=torch.as_tensor(next_states,dtype=torch.float32,device=device)
        #weights=torch.as_tensor(weights,dtype=torch.float32,device=device)
        #batch_indexs=torch.as_tensor(batch_indexs,dtype=torch.int64,device=device)
        
        #This will give shape [B,num_action_branches,1]
        # print(self.q_net(states).shape)
        # print(actions.shape)


        q_vals = self.q_net(states).gather(dim=2, index=actions.unsqueeze(2)) #I need to make sure actions has dimension [B,num_action_branches,1]
        with torch.no_grad():
            #So I need to take argmax along num_actions_per_branch this will give shape [B,num_action_branches,1]
            next_actions = self.q_net(next_states).argmax(dim=2,keepdims=True)        # Online SELECTS

            #This will give shape [B,num_action_branches,1]
            next_q_vals = self.target_net(next_states).gather(dim=2,index=next_actions)  # Target EVALUATES


            #Reshaping reward so that It matches the right shape
            rewards=rewards.view(batch_size,1,1)
            #rewards is of shape [B,1] and gets broadcast to [B,num_action_branches,1]
            targets = rewards + self.gamma * next_q_vals
        
        #Has dimension [B,num_action_branches,1]
        mse_all = self.loss_fn(q_vals, targets)
        
        #this has dimension [B]
        td_errors_sq= mse_all.mean(dim=(1,2))
        td_errors=torch.abs(q_vals-targets).mean(dim=(1,2))

        #This is a scalar
        loss= td_errors_sq.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        #Clip gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(),GRAD_NORM_CLIP)
        self.optimizer.step()

        new_priorities=td_errors.detach().cpu().numpy() + self.prioritised_replay_eps
        #self.replay_buffer.update_priorities(batch_indexs,new_priorities)
        return loss.item()

    def soft_update(self, tau=0.01):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)