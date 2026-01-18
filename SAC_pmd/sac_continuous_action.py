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

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    state_dim=24
    action_dim=24
    max_action = float(0.5)
    min_action = -max_action

    actor = Actor(state_dim,action_dim,max_action,min_action).to(device)
    qf1 = SoftQNetwork(state_dim,action_dim).to(device)
    qf2 = SoftQNetwork(state_dim,action_dim).to(device)
    qf1_target = SoftQNetwork(state_dim,action_dim).to(device)
    qf2_target = SoftQNetwork(state_dim,action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    try:
        ckpt=torch.load("./checkpoints/Hopper-v5__sac_continuous_action__1__1768649355_step10000.pt",map_location=device)
        actor.load_state_dict(ckpt["actor"])
        qf1.load_state_dict(ckpt["qf1"])
        qf2.load_state_dict(ckpt["qf2"])
        qf1_target.load_state_dict(ckpt["qf1_target"])
        qf2_target.load_state_dict(ckpt["qf2_target"])
        q_optimizer.load_state_dict(ckpt["q_optimizer"])
        actor_optimizer.load_state_dict(ckpt["actor_optimizer"])

        print("✅ Loaded trained model from ./checkpoints/Hopper-v5__sac_continuous_action__1__1768649355_step10000.pt")
    except FileNotFoundError:
        pass
    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -action_dim  # heuristic value from original paper
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)

        try:
            ckpt=torch.load("./checkpoints/Hopper-v5__sac_continuous_action__1__1768649355_step10000.pt",map_location=device)
            log_alpha.data.copy_(ckpt["log_alpha"])
            a_optimizer.load_state_dict(ckpt["a_optimizer"])

            print("✅ Loaded autotune model from ./checkpoints/Hopper-v5__sac_continuous_action__1__1768649355_step10000.pt")
        except FileNotFoundError:
            pass
    else:
        alpha = args.alpha

    rb = ReplayBuffer(args.buffer_size)
    start_time = time.time()
    global_step=0
    num_episodes=100000
    N_symbols=1000
    
    E_in=cma_utils.gen_I_Q_qpsk(N_symbols)
    E_after_pmd=cma_utils.apply_pmd(E_in)
    E_after_pmd_normalised=cma_utils.normalise(E_after_pmd)
    
    

    for episode in range(num_episodes):
        initial_filters=cma_utils.initialise_filters(NUM_TAPS=3)
        intial_state=cma_utils.convert_filter_to_state(initial_filters)
        state=cma_utils.convert_filter_to_state(initial_filters)
        cur_ind=3
        done=False
        episode_reward=0
        episode_length=0
        while not done:
        # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                #Make sure actions are bw -0.5 to 0.5
                actions = np.clip(np.random.randn(action_dim), -0.001, 0.001) #np.random.randn(action_dim)
            else:
                #print(torch.Tensor(state).to(device).unsqueeze(0))
                actions, _, _ = actor.get_action(torch.Tensor(state).to(device).unsqueeze(0))
                actions = actions.detach().cpu().numpy().squeeze()
                #print("actions is",actions)
            # TRY NOT TO MODIFY: execute the game and log data.
            #next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            next_state=state+actions
            #print("next state is",next_state)
            x_out,y_out=cma_utils.apply_filters(E_after_pmd_normalised,cur_ind,3,cma_utils.state_to_filter(next_state,3))
            reward=cma_utils.compute_reward(x_out,y_out)
            
            episode_reward+=reward
            
            episode_length+=1
            done= (cur_ind==N_symbols-1) or (cma_utils.calculate_state_distance(next_state,intial_state)>0.7)
            
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                #print("data is",data)
                #print("this does",map(np.stack, zip(*data)))
                states_sampled, actions_sampled, rewards_sampled, next_states_sampled, dones_sampled = map(np.stack, zip(*data))
                
                states_sampled = torch.tensor(states_sampled,device=device,dtype=torch.float32)
                actions_sampled = torch.tensor(actions_sampled,device=device,dtype=torch.float32)
                next_states_sampled = torch.tensor(next_states_sampled,device=device,dtype=torch.float32)
                rewards_sampled = torch.tensor(rewards_sampled.reshape(-1, 1),device=device,dtype=torch.float32)
                dones_sampled = torch.tensor(dones_sampled.reshape(-1, 1),device=device,dtype=torch.int64)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(next_states_sampled)
                    qf1_next_target = qf1_target(next_states_sampled, next_state_actions)
                    qf2_next_target = qf2_target(next_states_sampled, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    #print("rewards are",rewards_sampled.flatten(),"dones are",dones_sampled.flatten())
                    next_q_value = rewards_sampled.flatten() + (1 - dones_sampled.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(states_sampled, actions_sampled).view(-1)
                qf2_a_values = qf2(states_sampled, actions_sampled).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = actor.get_action(states_sampled)
                        qf1_pi = qf1(states_sampled, pi)
                        qf2_pi = qf2(states_sampled, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(states_sampled)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

            # update the target networks
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                
                if global_step > 0 and (global_step % 10_000 == 0):
                    os.makedirs("checkpoints", exist_ok=True)
                    ckpt_path = f"checkpoints/{run_name}_step{global_step}.pt"
                    torch.save(
                        {
                            "global_step": global_step,
                            "actor": actor.state_dict(),
                            "qf1": qf1.state_dict(),
                            "qf2": qf2.state_dict(),
                            "qf1_target": qf1_target.state_dict(),
                            "qf2_target": qf2_target.state_dict(),
                            "actor_optimizer": actor_optimizer.state_dict(),
                            "q_optimizer": q_optimizer.state_dict(),
                            # optional: alpha state if autotune
                            "log_alpha": log_alpha.detach().cpu() if args.autotune else None,
                            "a_optimizer": a_optimizer.state_dict() if args.autotune else None,
                            "args": vars(args),
                        },
                        ckpt_path
                    )

            rb.push(state,actions,reward,next_state,done)
            #print("state is",state,"next state is",next_state,"action is",actions)
            state=next_state
            cur_ind+=1
            global_step+=1
        # TRY NOT TO MODIFY: record rewards_sampled for plotting purposes
        #print(infos["_episode"])
        
        print(f"global_step={global_step}, episodic_return={episode_reward} , episode_length={episode_length} , final_state_distance={cma_utils.calculate_state_distance(state,intial_state)}")
        writer.add_scalar("charts/episodic_return", episode_reward, global_step)
        writer.add_scalar("charts/episodic_length", episode_length, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        
        # ALGO LOGIC: training.
        

    writer.close()
