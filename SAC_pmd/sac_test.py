# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard.writer import SummaryWriter

from buffers import ReplayBuffer

import cma_utils

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v5"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = int(5e3)
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""




# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim,action_dim):
        super().__init__()
        self.fc1 = nn.Linear(
            state_dim + action_dim,
            256
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, state_dim,action_dim, max_action,min_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (max_action - min_action) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (max_action + min_action) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        #print("mean is",mean)
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #envs =  gym.make("Hopper-v5",render_mode="human")
    #frame_delay=1/30
    
    #max_action = float(envs.action_space.high[0])
    state_dim=24
    action_dim=24
    max_action = float(0.5)
    min_action = -max_action

    ckpt=torch.load("./checkpoints/Hopper-v5__sac_continuous_action__1__1768748860_step100000.pt",map_location=device)
    actor = Actor(state_dim,action_dim,max_action,min_action).to(device)
    #qf1 = SoftQNetwork(state_dim,action_dim).to(device)
    #qf2 = SoftQNetwork(state_dim,action_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    # Automatic entropy tuning
    
    episode_rewards=[]
    episode_lengths=[]

    N_symbols=1000
    NUM_TAPS=3

    E_in=cma_utils.gen_I_Q_qpsk(N_symbols)
    E_after_pmd=cma_utils.apply_pmd(E_in)
    E_after_pmd_normalised=cma_utils.normalise(E_after_pmd)
    for episode in range(1):
        initial_filters=cma_utils.initialise_filters(NUM_TAPS=3)
        intial_state=cma_utils.convert_filter_to_state(initial_filters)
        state=cma_utils.convert_filter_to_state(initial_filters)
        E_cma,cma_converged_filters=cma_utils.cma_python(E_after_pmd_normalised,NUM_TAPS)
        cur_ind=3
        done=False
        episode_reward=0
        episode_length=0
        cma_total_reward=0
        while not done:
            _,_,mean_actions= actor.get_action(torch.Tensor(state).to(device).unsqueeze(0))
            actions = mean_actions.detach().cpu().numpy().flatten()

            next_state=state+actions
            
            done=(cur_ind==N_symbols-1)

            x_out,y_out=cma_utils.apply_filters(E_after_pmd_normalised,cur_ind,NUM_TAPS,cma_utils.state_to_filter(next_state,NUM_TAPS))
            x_out_ideal,y_out_ideal=cma_utils.apply_filters(E_after_pmd_normalised,cur_ind,NUM_TAPS,cma_converged_filters)
            
            reward=cma_utils.compute_reward(x_out,y_out)
            cma_reward=cma_utils.compute_reward(x_out_ideal,y_out_ideal)
            cma_total_reward+=cma_reward
            episode_reward+=reward

            state=next_state
            cur_ind+=1
        
        episode_rewards.append(episode_reward)
        
        #episode_lengths.append(steps)

        print(f"Episode {episode+1}: "
              f"Reward = {episode_reward:.2f}"
              f" CMA Reward = {cma_total_reward:.2f}")

    converged_filters=cma_utils.state_to_filter(state,NUM_TAPS)
    print(converged_filters)
    E_out=cma_utils.apply_entire_filters(E_after_pmd_normalised,converged_filters)
    cma_utils.plot_constellation(E_after_pmd_normalised)
    #cma_utils.plot_constellation(E_out)
    #cma_utils.plot_constellation(E_cma)