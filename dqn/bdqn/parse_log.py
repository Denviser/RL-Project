import re
import numpy as np
import matplotlib.pyplot as plt

def plot_distance_histogram(filename, n_bins=50, save_name='distance_histogram.png'):
    """
    Parse distance data from file and plot frequency histogram
    
    Parameters:
    -----------
    filename : str
        Path to the file containing distance data
    n_bins : int
        Number of bins for the histogram (default: 50)
    save_name : str
        Output filename for the plot (default: 'distance_histogram.png')
    """
    
    # Read the file
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract all distance values using regex
    # Matches np.float64(...) patterns
    pattern = r'np\.float64\(([\d.e+-]+)\)'
    matches = re.findall(pattern, content)
    
    # Convert to numpy array
    distances = np.array([float(x) for x in matches])
    
    # Print statistics
    print(f"Total number of distances: {len(distances)}")
    print(f"Min distance: {distances.min():.6f}")
    print(f"Max distance: {distances.max():.6f}")
    print(f"Mean distance: {distances.mean():.6f}")
    print(f"Std distance: {distances.std():.6f}")
    
    # Create histogram with N bins from 0 to max distance
    distance_max = distances.max()
    
    plt.figure(figsize=(12, 6))
    counts, bins, patches = plt.hist(distances, bins=n_bins, 
                                      range=(0, distance_max), 
                                      edgecolor='black', alpha=0.7, color='steelblue')
    
    plt.xlabel('Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distance Distribution Histogram (N={n_bins} bins)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Total: {len(distances)}\nMean: {distances.mean():.4f}\nStd: {distances.std():.4f}'
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"\nHistogram saved as '{save_name}'")
    
    # Print bin statistics
    bin_width = distance_max / n_bins
    print(f"\nBin width: {bin_width:.6f}")
    print(f"Most frequent bin: [{bins[counts.argmax()]:.6f}, {bins[counts.argmax()+1]:.6f}]")
    print(f"Frequency in most frequent bin: {int(counts.max())} samples")
    
    return distances, counts, bins


if __name__ == "__main__":
    # Example usage
    distances, counts, bins = plot_distance_histogram('run.log', n_bins=50)
    
    # Optional: Show the plot
    plt.show()
