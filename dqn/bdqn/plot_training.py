import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file_path):
    """
    Parse log file and extract episode, avg_loss, and reward

    Args:
        log_file_path: Path to the log file (e.g., 'run.log')

    Returns:
        episodes, avg_losses, rewards: Lists of parsed values
    """
    episodes = []
    avg_losses = []
    rewards = []

    # Regular expression to match the log format
    pattern = r'episode=(\d+)\s+avg_loss=([\d.]+)\s+reward=([\d.]+)'

    with open(log_file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                episode = int(match.group(1))
                avg_loss = float(match.group(2))
                reward = float(match.group(3))

                episodes.append(episode)
                avg_losses.append(avg_loss)
                rewards.append(reward)

    return episodes, avg_losses, rewards

def plot_training_metrics(episodes, avg_losses, rewards, save_path='training_plots.png'):
    """
    Plot training metrics: reward and loss vs episodes

    Args:
        episodes: List of episode numbers
        avg_losses: List of average losses
        rewards: List of rewards
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Reward vs Episodes
    ax1.plot(episodes, rewards, 'b-', linewidth=1.5, alpha=0.7, label='Reward')
    ax1.scatter(episodes, rewards, c='blue', s=20, alpha=0.5)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Training Reward vs Episodes', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add moving average for reward (if enough data points)
    if len(rewards) >= 5:
        window_size = min(5, len(rewards))
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        moving_avg_episodes = episodes[window_size-1:]
        ax1.plot(moving_avg_episodes, moving_avg, 'r--', linewidth=2, 
                label=f'Moving Avg (window={window_size})', alpha=0.8)
        ax1.legend()

    # Plot 2: Average Loss vs Episodes
    # Filter out the initial large losses (1000.0) for better visualization
    filtered_losses = []
    filtered_episodes = []
    for ep, loss in zip(episodes, avg_losses):
        if loss < 100:  # Filter out placeholder losses
            filtered_losses.append(loss)
            filtered_episodes.append(ep)

    if filtered_losses:
        ax2.plot(filtered_episodes, filtered_losses, 'g-', linewidth=1.5, 
                alpha=0.7, label='Avg Loss (filtered)')
        ax2.scatter(filtered_episodes, filtered_losses, c='green', s=20, alpha=0.5)
    else:
        ax2.plot(episodes, avg_losses, 'g-', linewidth=1.5, alpha=0.7, label='Avg Loss')
        ax2.scatter(episodes, avg_losses, c='green', s=20, alpha=0.5)

    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Average Loss', fontsize=12)
    ax2.set_title('Training Loss vs Episodes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')  # Log scale for better visualization of loss

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

    # Print statistics
    print(f"\n{'='*60}")
    print("TRAINING STATISTICS:")
    print(f"{'='*60}")
    print(f"Total Episodes: {len(episodes)}")
    print(f"\nReward Statistics:")
    print(f"  Mean Reward: {np.mean(rewards):.4f}")
    print(f"  Max Reward: {np.max(rewards):.4f} (Episode {episodes[np.argmax(rewards)]})")
    print(f"  Min Reward: {np.min(rewards):.4f} (Episode {episodes[np.argmin(rewards)]})")
    print(f"  Final Reward: {rewards[-1]:.4f}")

    if filtered_losses:
        print(f"\nLoss Statistics (filtered):")
        print(f"  Mean Loss: {np.mean(filtered_losses):.6f}")
        print(f"  Max Loss: {np.max(filtered_losses):.6f}")
        print(f"  Min Loss: {np.min(filtered_losses):.6f}")
        print(f"  Final Loss: {filtered_losses[-1]:.6f}")

if __name__ == "__main__":
    # Parse the log file
    log_file = "run.log"  # Change this to your log file path

    try:
        episodes, avg_losses, rewards = parse_log_file(log_file)

        if not episodes:
            print(f"No data found in {log_file}. Check the file format.")
        else:
            print(f"Successfully parsed {len(episodes)} episodes from {log_file}")
            plot_training_metrics(episodes, avg_losses, rewards)

    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found.")
        print("Make sure the file exists in the current directory.")
    except Exception as e:
        print(f"Error: {e}")
