import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models import DQNAgent
import cma_utils
import logging
logging.basicConfig(filename="run.log", level=logging.INFO)




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
REPLAY_BUFFER_SIZE=int(1e6)
NUM_EPISODES=400
PRIORITISED_REPLAY_ALPHA=0.7
PRIORITISED_REPLAY_BETA=0.4
PRIORITISED_REPLAY_EPSILON=1e-3
GREEDY_EPS_INITIAL=1
GREEDY_EPS_FINAL=0.1
TARGET_NETWORK_TAU=0.01
MODEL_PATH="bdq_qnet.pt"
BATCH_SIZE=256
GRAD_NORM_CLIP=10
REWARD_CLIP=-10

def initialise_filters(NUM_TAPS):
    filters={}
    filters['pxx'] = np.zeros(NUM_TAPS, dtype=complex)
    filters['pxx'][NUM_TAPS//2] = 1

    filters['pxy'] = np.zeros(NUM_TAPS,dtype=complex)
    
    filters['pyx'] = np.zeros(NUM_TAPS,dtype=complex)
    
    filters['pyy'] = np.zeros(NUM_TAPS)
    filters['pyy'][NUM_TAPS//2] = 1
    
    
    return filters


def _interleave_real_imag(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    # shape (N,) -> (N,2) as [real, imag] per element, then flatten -> (2N,)
    return np.stack([x.real, x.imag], axis=1).ravel().astype(np.float32)

def convert_filter_to_state(filters: dict) -> np.ndarray:
    keys = ["pxx", "pxy", "pyx", "pyy"]
    parts = [_interleave_real_imag(filters[k]) for k in keys]
    return np.concatenate(parts, axis=0)

def _deinterleave_to_complex(v: np.ndarray, num_taps: int, complex_dtype=np.complex64) -> np.ndarray:
    v = np.asarray(v)
    assert v.size == 2 * num_taps, f"Expected {2*num_taps} values, got {v.size}"
    ri = v.reshape(num_taps, 2)                # [[r0,i0],[r1,i1],...]
    return (ri[:, 0] + 1j * ri[:, 1]).astype(complex_dtype) 

def state_to_filter(state: np.ndarray, num_taps: int) -> dict:
    """
    Inverse of convert_filter_to_state (the interleaved version):
    state layout:
      pxx: r0,i0,r1,i1,...,r_{N-1},i_{N-1},
      pxy: ...,
      pyx: ...,
      pyy: ...
    Returns dict with complex tap arrays.
    """
    state = np.asarray(state, dtype=np.float32)
    n_per_filter = 2 * num_taps
    expected = 4 * n_per_filter
    assert state.size == expected, f"Expected state length {expected}, got {state.size}"

    keys = ["pxx", "pxy", "pyx", "pyy"]
    filters = {}

    offset = 0
    for k in keys:
        chunk = state[offset: offset + n_per_filter]
        filters[k] = _deinterleave_to_complex(chunk, num_taps)
        offset += n_per_filter

    return filters

def round_filters_to_step(filters: dict, step: float = 1e-3) -> dict:
    out = {}
    for k, v in filters.items():
        v = np.asarray(v)
        # handle complex and real arrays uniformly
        vr = np.round(v.real / step) * step  # [web:441]
        vi = np.round(v.imag / step) * step  # [web:441]
        out[k] = (vr + 1j * vi).astype(v.dtype)
    return out

def calculate_state_distance(cur_state,intial_state):
    return np.linalg.norm(cur_state-intial_state)

def train(E_in):
    """Takes as argument E_in which is normalised and has polarisation mixing"""
    agent=DQNAgent(NUM_TAPS,
                   MODEL_PATH,
                   DELTA,
                   GAMMA,
                   LR_Q_NET,
                   REPLAY_BUFFER_SIZE,
                   PRIORITISED_REPLAY_ALPHA,
                   PRIORITISED_REPLAY_BETA,
                   PRIORITISED_REPLAY_EPSILON)
    
    eps=GREEDY_EPS_INITIAL
    eps_decay=(GREEDY_EPS_INITIAL-GREEDY_EPS_FINAL)/NUM_EPISODES
    for episode in range(NUM_EPISODES):
        #E_created=cma_utils.gen_I_Q_qpsk(N_SYMBOLS)
        #E_after_pmd=cma_utils.apply_pmd(E_created)
        #E_in=cma_utils.normalise(E_after_pmd)

        initial_filters=initialise_filters(NUM_TAPS)
        intial_state=convert_filter_to_state(initial_filters)
        state=convert_filter_to_state(initial_filters)
        cur_ind=NUM_TAPS-1
        episodic_reward=0

        #distance_arr=[]
        for cur_ind in range(NUM_TAPS,N_SYMBOLS):

            #cur_action has shape [num_action_branches]
            cur_actions=agent.select_action(state,eps).cpu().numpy()
            
            #Actions have values 0,1,2 we subtract 1 to get -1,0,1 for the direction
            directions=cur_actions-1


            #Move in the direction to get the new state
            next_state=state+directions*DELTA

            #Add the distance
            #distance_arr.append(calculate_state_distance(next_state,intial_state))
            #We want to first play the action then recieve the reward for it
            x_out,y_out=cma_utils.apply_filters(E_in,cur_ind,NUM_TAPS,state_to_filter(next_state,NUM_TAPS))
            #compute reward
            reward=cma_utils.compute_reward(x_out,y_out,REWARD_CLIP)
            #print("state is",state)
            #print("next_state is",next_state)

            agent.replay_buffer.add(state,cur_actions,reward,next_state)

            episodic_reward+=reward
            #update the state
            state=next_state

            loss=agent.update(batch_size=BATCH_SIZE,GRAD_NORM_CLIP=GRAD_NORM_CLIP)

        eps=GREEDY_EPS_INITIAL-eps_decay*episode
        #loss=agent.update(batch_size=BATCH_SIZE,GRAD_NORM_CLIP=GRAD_NORM_CLIP)
        agent.soft_update(tau=TARGET_NETWORK_TAU)

        print(f"Episode {episode+1}, epsilon={eps:.3f}, last_loss={loss:.6f}, reward={episodic_reward:.6f}")
        logging.info("episode=%d loss=%f reward=%f \n", episode+1, loss,episodic_reward)
        torch.save(agent.q_net.state_dict(),MODEL_PATH)

def test():
    agent=DQNAgent(NUM_TAPS,
                   MODEL_PATH,
                   DELTA,
                   GAMMA,
                   LR_Q_NET,
                   REPLAY_BUFFER_SIZE,
                   PRIORITISED_REPLAY_ALPHA,
                   PRIORITISED_REPLAY_BETA,
                   PRIORITISED_REPLAY_EPSILON)
    E_created=cma_utils.gen_I_Q_qpsk(N_SYMBOLS)
    E_after_pmd=cma_utils.apply_pmd(E_created)
    E_in=cma_utils.normalise(E_after_pmd)

    initial_filters=initialise_filters(NUM_TAPS)
    state=convert_filter_to_state(initial_filters)

    eps=0
    cur_ind=NUM_TAPS-1
    x_out_arr=[]
    y_out_arr=[]

    for cur_ind in range(NUM_TAPS,N_SYMBOLS):

            #cur_action has shape [num_action_branches]
            cur_actions=agent.select_action(state,eps).cpu().numpy()
            
            #Actions have values 0,1,2 we subtract 1 to get -1,0,1 for the direction
            directions=cur_actions-1

            x_out,y_out=cma_utils.apply_filters(E_in,cur_ind,NUM_TAPS,state_to_filter(state,NUM_TAPS))

            #compute reward
            #Move in the direction to get the new state
            next_state=state+directions*DELTA

            x_out_arr.append(x_out)
            y_out_arr.append(y_out)
            #update the state
            state=next_state

    x_out_arr=np.array(x_out_arr)
    y_out_arr=np.array(y_out_arr)

    E_out=np.column_stack((x_out_arr,y_out_arr))
    
    return  E_out


def main():

    E_created=cma_utils.gen_I_Q_qpsk(N_SYMBOLS)
    print("e created is",E_created)
    E_after_pmd=cma_utils.apply_pmd(E_created)
    
    E_out,converged_filters=cma_utils.cma_python(E_after_pmd,NUM_TAPS,LR_CMA)

    print("Initial converged filters are",converged_filters)

    converged_filters_rounded=round_filters_to_step(converged_filters,DELTA)
    
    print("Rounded filters are",converged_filters_rounded)
    E_normalised=cma_utils.normalise(E_after_pmd)
    
    print("Distance bw converged and intial for cma is",calculate_state_distance(convert_filter_to_state(converged_filters),convert_filter_to_state(initialise_filters(NUM_TAPS))))
    #cma_utils.plot_constellation(E_out)
    
    E_out_rounded=cma_utils.apply_entire_filters(E_normalised,converged_filters_rounded)
    #cma_utils.plot_constellation(E_out_rounded)
    train(E_normalised)
    #E_out=test()
    #cma_utils.plot_constellation(E_out)



if __name__ == '__main__':
    main()