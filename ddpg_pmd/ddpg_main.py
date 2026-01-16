# Download required libraries
import gymnasium as gym
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import math
import os
import time
import logging
logging.basicConfig(filename="run.log", level=logging.INFO)
import cma_utils

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Device:", device)


# Set up the environment
#env = gym.make('Hopper-v5')
state_dim = 24
action_dim = 24
#print(env.action_space.high)
max_action = float(1)
ACTOR_MODEL_PATH = 'actor_model.pth'
CRITIC_MODEL_PATH = 'critic_model.pth'

# Hyperparameters
ACTOR_LR = 1e-3
CRITIC_LR = 1e-4
HIDDEN_LAYERS_ACTOR = 256
HIDDEN_LAYERS_CRITIC = 256

N_SYMBOLS=1000
LR_CMA=0.01
NUM_TAPS=3
NUM_FILTERS=4
MATLAB_FUNCTION_PATH="C:\\optical\\Coherent-Optical-Communication\\Coherent-Optical-Communication\\functions\\DSP"
NUM_ACTION_BRANCHES=24 #Two for each filter tap for real and imag
NUM_ACTIONS_PER_BRANCH=3 #Increase,Decrease,No change
DELTA=1e-3 #How much to change increase or decrease each tap by for an action
GAMMA=0.99 #Discount factor
LR_Q_NET=1e-3
BUFFER_SIZE=int(1e5)
NUM_EPISODES=10000
PRIORITISED_REPLAY_ALPHA=0.7
PRIORITISED_REPLAY_BETA=0.4
PRIORITISED_REPLAY_EPSILON=1e-3
GREEDY_EPS_INITIAL=1
GREEDY_EPS_FINAL=0.1
TAU=0.01
MODEL_PATH="bdq_qnet.pt"
BATCH_SIZE=256
GRAD_NORM_CLIP=10
REWARD_CLIP=-10
MAX_DISTANCE=1

class Actor(nn.Module):
    """
    Actor network for the DDPG algorithm.
    """
    def __init__(self, state_dim, action_dim, max_action,use_batch_norm):
        """
        Initialise the Actor's Policy network.

        :param state_dim: Dimension of the state space
        :param action_dim: Dimension of the action space
        :param max_action: Maximum value of the action
        """
        super(Actor, self).__init__()
        self.bn1 = nn.LayerNorm(HIDDEN_LAYERS_ACTOR,device=device) if use_batch_norm else nn.Identity()
        self.bn2 = nn.LayerNorm(HIDDEN_LAYERS_ACTOR,device=device) if use_batch_norm else nn.Identity()

        self.l1 = nn.Linear(state_dim, HIDDEN_LAYERS_ACTOR,device=device)
        self.l2 = nn.Linear(HIDDEN_LAYERS_ACTOR, HIDDEN_LAYERS_ACTOR,device=device)
        self.l3 = nn.Linear(HIDDEN_LAYERS_ACTOR, action_dim,device=device)
        self.max_action = max_action

    def forward(self, state):
        """
        Forward propagation through the network.

        :param state: Input state
        :return: Action
        """

        a = torch.relu(self.bn1(self.l1(state)))
        a = torch.relu(self.bn2(self.l2(a)))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    """
    Critic network for the DDPG algorithm.
    """
    def __init__(self, state_dim, action_dim,use_batch_norm):
        """
        Initialise the Critic's Value network.

        :param state_dim: Dimension of the state space
        :param action_dim: Dimension of the action space
        """
        super(Critic, self).__init__()
        self.bn1 = nn.BatchNorm1d(HIDDEN_LAYERS_CRITIC,device=device) if use_batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(HIDDEN_LAYERS_CRITIC,device=device) if use_batch_norm else nn.Identity()
        self.l1 = nn.Linear(state_dim + action_dim, HIDDEN_LAYERS_CRITIC,device=device)

        self.l2 = nn.Linear(HIDDEN_LAYERS_CRITIC, HIDDEN_LAYERS_CRITIC,device=device)
        self.l3 = nn.Linear(HIDDEN_LAYERS_CRITIC, 1,device=device)

    def forward(self, state, action):
        """
        Forward propagation through the network.

        :param state: Input state
        :param action: Input action
        :return: Q-value of state-action pair
        """
        q = torch.relu(self.bn1(self.l1(torch.cat([state, action], 1))))
        q = torch.relu(self.bn2(self.l2(q)))
        return self.l3(q)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""
class OUNoise(object):
    def __init__(self, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.low          = 0
        self.high         = max_action
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

class DDPG():
    """
    Deep Deterministic Policy Gradient (DDPG) agent.
    """
    def __init__(self, state_dim, action_dim, max_action,use_batch_norm):
        """
        [STEP 0]
        Initialise the DDPG agent.

        :param state_dim: Dimension of the state space
        :param action_dim: Dimension of the action space
        :param max_action: Maximum value of the action
        """
        # [STEP 0]
        # Initialise Actor's Policy network
        self.actor = Actor(state_dim, action_dim, max_action,use_batch_norm)
        try:
            actor_state= torch.load(ACTOR_MODEL_PATH,map_location=device)
            self.actor.load_state_dict(actor_state)
        except FileNotFoundError:
            pass
        # Initialise Actor target network with same weights as Actor's Policy network
        self.actor_target = Actor(state_dim, action_dim, max_action,use_batch_norm)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        # Initialise Critic's Value network
        self.critic = Critic(state_dim, action_dim,use_batch_norm)
        try:
            critic_state= torch.load(CRITIC_MODEL_PATH,map_location=device)
            self.critic.load_state_dict(critic_state)
        except FileNotFoundError:
            pass
        
        
        # Initialise Crtic's target network with same weights as Critic's Value network
        self.critic_target = Critic(state_dim, action_dim,use_batch_norm)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        # Initialise the Replay Buffer
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)


    def select_action(self, state):
        """
        [STEP 1]
        Select an action given the current state.

        :param state: Current state
        :return: Selected action
        """
        #print(state)
        state = torch.tensor(state.reshape(1, -1),device=device,dtype=torch.float32)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, use_target_network,use_batch_norm):
        """
        Train the DDPG agent.

        :param use_target_network: Whether to use target networks or not
        :param use_batch_norm: Whether to use batch normalisation or not
        """
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        # [STEP 4]. Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(BATCH_SIZE)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        state = torch.tensor(state,device=device,dtype=torch.float32)
        action = torch.tensor(action,device=device,dtype=torch.float32)
        next_state = torch.tensor(next_state,device=device,dtype=torch.float32)
        reward = torch.tensor(reward.reshape(-1, 1),device=device,dtype=torch.float32)
        done = torch.tensor(done.reshape(-1, 1),device=device,dtype=torch.int64)

        # Critic Network update #
        if use_target_network:
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
        else:
            target_Q = self.critic(next_state, self.actor(next_state))

        # [STEP 5]. Calculate target Q-value (y_i)
        target_Q = reward + (1 - done) * GAMMA * target_Q
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())

        # [STEP 6]. Use gradient descent to update weights of the Critic network
        # to minimise loss function
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Network update #
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # [STEP 7]. Use gradient descent to update weights of the Actor network
        # to minimise loss function and maximise the Q-value => choose the action that yields the highest cumulative reward
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # [STEP 8]. Update target networks
        if use_target_network:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
def train_ddpg(use_target_network, use_batch_norm, num_episodes=NUM_EPISODES):
    """
    Train the DDPG agent.

    :param use_target_network: Whether to use target networks
    :param use_batch_norm: Whether to use batch normalization
    :param num_episodes: Number of episodes to train
    :return: List of episode rewards
    """
    agent = DDPG(state_dim, action_dim, 1,use_batch_norm)

    episode_rewards = []
    noise = OUNoise()
    E_created=cma_utils.gen_I_Q_qpsk(N_SYMBOLS)
    E_after_pmd=cma_utils.apply_pmd(E_created)
    E_in=cma_utils.normalise(E_after_pmd)

    for episode in range(num_episodes):
        noise.reset()
        episode_reward = 0
        done = False
        step=0
        initial_filters=cma_utils.initialise_filters(NUM_TAPS)
        initial_state=cma_utils.convert_filter_to_state(initial_filters)
        state=cma_utils.convert_filter_to_state(initial_filters)
        cur_ind=NUM_TAPS
        while not done:
            action_actor = agent.select_action(state)
            action = noise.get_action(action_actor,step) # Add noise for exploration
            next_state = state + action
            x_out,y_out=cma_utils.apply_filters(E_in,cur_ind,NUM_TAPS,cma_utils.state_to_filter(next_state,NUM_TAPS))
            reward=cma_utils.compute_reward(x_out,y_out)
            reward+=step
            #next_state, reward,terminated,truncated,info_= env.step(action)
            #print(action)
            #print(state)
            cur_ind+=1
            done = (cur_ind==N_SYMBOLS) or (cma_utils.calculate_state_distance(next_state,initial_state)>MAX_DISTANCE)

            #done_replay_buffer = terminated
            #done = float(done) if isinstance(done, (bool, int)) else float(done[0])
            agent.replay_buffer.push(state, action, reward, next_state, done)

            if len(agent.replay_buffer) > BATCH_SIZE:
                agent.train(use_target_network,use_batch_norm)

            state = next_state
            episode_reward += reward
            step+=1

        episode_rewards.append(episode_reward)

        logging.info("Episode %d,Reward = %f", episode + 1, episode_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
        
        if episode%100==0:
            torch.save(agent.actor.state_dict(),ACTOR_MODEL_PATH)
            torch.save(agent.critic.state_dict(),CRITIC_MODEL_PATH)
    return agent, episode_rewards

def test():
    agent=DDPG(state_dim,action_dim,1,use_batch_norm=False)
    E_created=cma_utils.gen_I_Q_qpsk(N_SYMBOLS)
    E_after_pmd=cma_utils.apply_pmd(E_created)
    E_in=cma_utils.normalise(E_after_pmd)

    initial_filters=cma_utils.initialise_filters(NUM_TAPS)
    state=cma_utils.convert_filter_to_state(initial_filters)

    eps=0
    cur_ind=NUM_TAPS-1
    x_out_arr=[]
    y_out_arr=[]
    total_reward=0

    for cur_ind in range(NUM_TAPS,N_SYMBOLS):

            #cur_action has shape [num_action_branches]
            action_actor = agent.select_action(state)
            #Actions have values 0,1,2 we subtract 1 to get -1,0,1 for the direction
            next_state=state+action_actor
            x_out,y_out=cma_utils.apply_filters(E_in,cur_ind,NUM_TAPS,cma_utils.state_to_filter(next_state,NUM_TAPS))

            #compute reward
            #Move in the direction to get the new state
            #next_state=state+directions*DELTA
            reward=cma_utils.compute_reward(x_out,y_out)
            total_reward+=reward
            x_out_arr.append(x_out)
            y_out_arr.append(y_out)
            #update the state
            state=next_state

    x_out_arr=np.array(x_out_arr)
    y_out_arr=np.array(y_out_arr)
    print(total_reward)
    E_out=np.column_stack((x_out_arr,y_out_arr))
    
    return  E_out

if __name__ == '__main__':
    train_ddpg(use_target_network=True, use_batch_norm=False)
    #E_out=test()
    #cma_utils.plot_constellation(E_out)
    #test_agent(render=True)