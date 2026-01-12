import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models import DQNAgent
import cma_utils
import logging
logging.basicConfig(filename="run.log", level=logging.INFO)
import gymnasium as gym
import time


STATE_DIM=11
NUM_ACTION_BRANCHES=3 #Two for each filter tap for real and imag
NUM_ACTIONS_PER_BRANCH=33 #Increase,Decrease,No change
GAMMA=0.99 #Discount factor
LR_Q_NET=1e-3
REPLAY_BUFFER_SIZE=int(1e6)
NUM_EPISODES=1000
PRIORITISED_REPLAY_ALPHA=0.6
PRIORITISED_REPLAY_BETA=0.4
PRIORITISED_REPLAY_EPSILON=1e-3
GREEDY_EPS_INITIAL=1
GREEDY_EPS_FINAL=0.01
TARGET_NETWORK_TAU=1e-3
MODEL_PATH="bdq_qnet.pt"
BATCH_SIZE=128
MAX_REWARD=-0.02


def convert_cont_action_to_discrete_action(actions, actions_min, actions_max, num_actions_per_branch):
    """
    Convert continuous actions to discrete action indices.
    
    Args:
        actions: continuous action values, shape (3,)
        actions_min: minimum values for each action dimension, shape (3,)
        actions_max: maximum values for each action dimension, shape (3,)
        num_actions_per_branch: number of discrete bins per action dimension
    
    Returns:
        discrete_actions: discrete action indices, shape (3,)
    """
    # Normalize continuous actions to [0, 1]
    normalized = (actions - actions_min) / (actions_max - actions_min)
    
    # Clip to ensure values are within [0, 1]
    normalized = np.clip(normalized, 0, 1)
    
    # Convert to discrete indices [0, num_actions_per_branch-1]
    discrete_actions = np.floor(normalized * num_actions_per_branch).astype(int)
    
    # Clip to ensure indices don't exceed num_actions_per_branch-1
    discrete_actions = np.clip(discrete_actions, 0, num_actions_per_branch - 1)
    
    return discrete_actions

def convert_discrete_action_to_cont_action(actions_discrete,actions_min,actions_max,num_actions_per_branch):
    actions=actions_discrete/num_actions_per_branch
    actions=actions*(actions_max-actions_min)+actions_min
    return actions

def train(env):
    state_dim=11
    num_action_branches=3
    num_actions_per_branch=33
    agent=DQNAgent(state_dim,num_action_branches,num_actions_per_branch,
                   MODEL_PATH, 
                   gamma=GAMMA, 
                   lr=LR_Q_NET, 
                   prioritised_replay_alpha=PRIORITISED_REPLAY_ALPHA,
                   prioritised_replay_beta=PRIORITISED_REPLAY_BETA,
                   prioritised_replay_epsilon=PRIORITISED_REPLAY_EPSILON,
                   replay_buffer_size=REPLAY_BUFFER_SIZE)
    
    eps=GREEDY_EPS_INITIAL
    eps_decay=(GREEDY_EPS_INITIAL-GREEDY_EPS_FINAL)/NUM_EPISODES
    
    for episode in range(NUM_EPISODES):

        initial_state,info=env.reset()
        state=initial_state

        cum_reward=0
        avg_loss=0
        done=False
        num_steps=0
        while not done:
            
            if np.random.rand() < eps:
                actions=env.action_space.sample()
                #actions=convert_cont_action_to_discrete_action(action_cont,env.action_space.low,env.action_space.high,num_actions_per_branch)
            else:
                actions_ind=agent.select_greedy_action(state)
                actions=convert_discrete_action_to_cont_action(actions_ind,env.action_space.low,env.action_space.high,num_actions_per_branch)            
            
            next_state,reward,terminated,truncated,info=env.step(actions)

            actions_ind=convert_cont_action_to_discrete_action(actions,env.action_space.low,env.action_space.high,num_actions_per_branch)
            #print(actions_ind)
            #actions_ind=np.asarray(actions_ind,dtype=np.int64)
            done=terminated or truncated
            
            agent.replay_buffer.add(state,actions_ind,reward,next_state,done)


            cum_reward+=reward
            #update the state
            state=next_state

            loss=agent.update(batch_size=BATCH_SIZE)
            avg_loss+=loss

            num_steps+=1

        eps=GREEDY_EPS_INITIAL-eps_decay*episode
        
        agent.soft_update(tau=TARGET_NETWORK_TAU)
        avg_loss/=num_steps

        print(f"Episode {episode+1}, epsilon={eps:.3f}, last_loss={avg_loss:.6f}, reward={cum_reward:.6f}")
        #print("Q values are",agent.q_net(torch.tensor(state,dtype=torch.float32,device="cuda").unsqueeze(0)))
        logging.info("episode=%d avg_loss=%f reward=%f", episode+1, avg_loss,cum_reward)
        if episode%100==0:
            torch.save(agent.q_net.state_dict(),MODEL_PATH)

def test_agent(num_episodes=10, render=False, save_video=False):
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
    agent = DQNAgent(
        STATE_DIM,
        NUM_ACTION_BRANCHES,
        NUM_ACTIONS_PER_BRANCH,
        MODEL_PATH,
        gamma=0.99,
        lr=1e-3  # Learning rate doesn't matter for testing
    )

    print(f"✅ Loaded trained model from {MODEL_PATH}")
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
            actions_ind = agent.select_greedy_action(state)
            actions = convert_discrete_action_to_cont_action(
                actions_ind,
                env.action_space.low,
                env.action_space.high,
                NUM_ACTIONS_PER_BRANCH
            )

            # Take action
            next_state, reward, terminated, truncated, info = env.step(actions)
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


def main():
    env=gym.make("Hopper-v5")
    
    #print(convert_cont_action_to_discrete_action(env.action_space.sample(),env.action_space.low,env.action_space.high,33))
    #print(env.action_space.low,env.action_space.high)
    # observation,info=env.reset()
    # print(observation)
    # print(env.step(env.action_space.sample()))
    #print(info)
    #train(env)
    #E_out=test()
    #cma_utils.plot_constellation(E_out)

    #train(env)
    rewards,lengths=test_agent(num_episodes=5,render=True,save_video=False)
if __name__ == '__main__':
    main()