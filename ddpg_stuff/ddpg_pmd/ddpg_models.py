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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Device:", device)


# Set up the environment
env = gym.make('Hopper-v5')
state_dim = 11
action_dim = 3
#print(env.action_space.high)
max_action = float(env.action_space.high[0])
ACTOR_MODEL_PATH = 'actor_model.pth'
CRITIC_MODEL_PATH = 'critic_model.pth'

# Hyperparameters
NUM_EPISODES = 1000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.01
ACTOR_LR = 1e-3
CRITIC_LR = 1e-4
BUFFER_SIZE = 100000
HIDDEN_LAYERS_ACTOR = 256
HIDDEN_LAYERS_CRITIC = 256

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
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
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
    noise = OUNoise(env.action_space)

    for episode in range(num_episodes):
        state,info= env.reset()
        noise.reset()
        episode_reward = 0
        done = False
        step=0
        while not done:
            action_actor = agent.select_action(state)
            action = noise.get_action(action_actor,step) # Add noise for exploration
            next_state, reward,terminated,truncated,info_= env.step(action)
            done = truncated or terminated

            done_replay_buffer = terminated
            #done = float(done) if isinstance(done, (bool, int)) else float(done[0])
            agent.replay_buffer.push(state, action, reward, next_state, done_replay_buffer)

            if len(agent.replay_buffer) > BATCH_SIZE:
                agent.train(use_target_network,use_batch_norm)

            state = next_state
            episode_reward += reward
            step+=1

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
        
        if episode%100==0:
            torch.save(agent.actor.state_dict(),ACTOR_MODEL_PATH)
            torch.save(agent.critic.state_dict(),CRITIC_MODEL_PATH)
    return agent, episode_rewards

def test_agent(num_episodes=10, render=False, save_video=False,use_batch_norm=False):
    """
    Test the trained agent on Hopper environment

    Args:
        num_episodes: Number of test episodes to run
        render: Whether to render the environment (slower but visual)
        save_video: Whether to save video of the episodes

    Returns:
        episode_rewards: List of total rewards per episode
        episode_lengths: List of episode lengths
    """
    # Create environment
    if save_video:
        env = gym.make("Hopper-v5", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda x: True)
        print("📹 Recording videos to 'videos/' directory")
    elif render:
        env = gym.make("Hopper-v5", render_mode="human")
    else:
        env = gym.make("Hopper-v5")

    frame_delay = 1.0 / 30 if render else 0

    # Initialize agent and load trained model
    agent=DDPG(state_dim, action_dim, 1,use_batch_norm)

    #print(f"✅ Loaded trained model from {MODEL_PATH}")
    print(f"\nRunning {num_episodes} test episodes...\n")

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Select greedy action (no exploration)
            action_actor = agent.select_action(state)
            print(action_actor)
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action_actor)
            done = terminated or truncated

            # Add delay for visible rendering
            if render and frame_delay > 0:
                time.sleep(frame_delay)
            
            total_reward += reward
            state = next_state
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        print(f"Episode {episode+1}/{num_episodes}: "
              f"Reward = {total_reward:.2f}, Steps = {steps}")

    env.close()

    # Print statistics
    print(f"\n{'='*60}")
    print("TEST RESULTS:")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"\nReward Statistics:")
    print(f"  Mean: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min: {np.min(episode_rewards):.2f}")
    print(f"  Max: {np.max(episode_rewards):.2f}")
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Min: {np.min(episode_lengths)}")
    print(f"  Max: {np.max(episode_lengths)}")

    return episode_rewards, episode_lengths

if __name__ == '__main__':
    #train_ddpg(use_target_network=True, use_batch_norm=False)
    test_agent(render=True)