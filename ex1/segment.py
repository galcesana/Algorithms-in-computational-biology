import numpy as np
import argparse
import matplotlib.pyplot as plt


def read_signal_file(filepath: str) -> np.ndarray:
    """
    Read signal data from a tab-separated text file.
    Each line contains one or more columns of values separated by tabs.

    Args:
        filepath: Path to the input text file

    Returns:
        np.ndarray: Array of signal values from all columns
    """
    return np.loadtxt(filepath, delimiter='\t', dtype=np.float32)


def print_segments(s: np.ndarray, c: float) -> None:
    """
    Print segments array with 3 decimal places.
    Each row contains [start, end, average].

    Args:
        s: numpy array of shape (n_segments, 3)
        c: Total cost value of optimal segmentation
    """
    for row in s:
        print(f"{int(row[0])} {int(row[1])} {np.round(row[2], 3)}")
    print(np.round(c, 3))
    return



def segment(x: np.ndarray, p: float, q: int):
    """
    Segment a signal using dynamic programming.

    Args:
        x: Input signal (numpy 1d array)
        p: Penalty parameter
        q: Maximum segment length

    Returns:
        s: numpy Array of segments. Shaped (n, 3), each row in format [start, end, average]
        c: Total cost value of optimal segmentation
    """
    # TODO: implement
    n = len(x)
    # array where cost[i] holds the minimum cost of segmenting (best score) the signal up to index 𝑖
    c = np.full(n + 1, np.inf, dtype=np.float32)
    c[0] = 0  # Base case: zero cost to segment an empty prefix
    # traceback -  array to store the optimal split points for reconstruction of segments.
    t = np.zeros(n + 1, dtype=int)

    # Precompute cumulative sums for efficient mean calculation
    cumulative_sum = np.cumsum(x, dtype=np.float32)
    cumulative_sum_squared = np.cumsum(x ** 2, dtype=np.float32)


    # Dynamic programming to compute minimum cost
    #For each possible end point 𝑗, evaluate possible starting points 𝑖, ensuring that j−i≤q to enforce the maximum
    # segment length.
    for j in range(1, n + 1):
        for i in range(max(0, j - q), j):
            seg_cost = segment_cost(i, j, cumulative_sum, cumulative_sum_squared) + p
            #update the cost array if a new minimum is found, and store the split point.
            if c[i] + seg_cost < c[j]:
                c[j] = c[i] + seg_cost
                t[j] = i

    # Backtrack to retrieve the segments with traceback array
    s = [] #segment array
    j = n
    while j > 0:
        i = t[j]
        segment_len = j - i
        sum_s = calc_sum_s(j,i,cumulative_sum)
        mean_s = sum_s / segment_len
        s.append([i+1, j, mean_s])
        j = i

    s.reverse()  # Order segments from start to end
    return np.array(s), c[n]-p


def segment_multi_channel(x: np.ndarray, p: float, q: int):
    """
    Similar to segment but with multiple channels.

    Args:
        x: Input signal (numpy 2d array, where rows are different channels)
        p: Penalty parameter
        q: Maximum segment length

    Returns:
        s: numpy Array of segments. Shaped (n, 2), each row in format [start, end]
        c: Total cost value of optimal segmentation
    """

    # TODO: implement
    return None


### helper func ###

def calc_sum_s(j, i, cumulative_sum):
    return cumulative_sum[j - 1] - (cumulative_sum[i - 1] if i > 0 else 0)

def segment_cost(i, j, cumulative_sum=None, cumulative_sum_squared=None):
  """Calculate cost of segment x[i:j] based on sse sum of squared errors."""
  segment_len = j - i
  sum_s = calc_sum_s(j,i,cumulative_sum)
  sum_s2 = calc_sum_s(j,i,cumulative_sum_squared)
  #mean_segment
  mean_s = sum_s / segment_len
  #calc sse - sum of squared errors - Opening the equation by abbreviated multiplication
  sse = sum_s2 - 2 * mean_s * sum_s + segment_len * mean_s ** 2
  return sse

def plot_one_signal(segments):
    # Plotting the original signal
    plt.figure(figsize=(10, 3))
    plt.plot(seq, 'bo', markersize=3, label='Original Signal')  # Blue dots for the original signal

    # Colors for segments
    segment_colors = ['purple', 'orange', 'green', 'blue', 'red']

    # Plot each segment with its respective color
    for i, segment in enumerate(segments):
        start, end, mean_value = int(segment[0]), int(segment[1]), segment[2]
        plt.plot(range(start, end), [mean_value] * (end - start),
                 color=segment_colors[i % len(segment_colors)], linewidth=4, label=f'Segment {i + 1}')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Signal Segmentation')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process signal data from a text file.')
    parser.add_argument('--filepath', type=str, required=True, help='Path to the input text file')
    parser.add_argument('--penalty', type=float, required=True, help='Penalty parameter')
    parser.add_argument('--max_len', type=int, required=True, help='Maximum segment length')
    parser.add_argument('--is_multi_channel', type=bool, required=False, default=False, help='call segment_multi_channel')
    args = parser.parse_args()
    seq = read_signal_file(args.filepath)

    if args.is_multi_channel:
        pass # TODO: implement only if you completed the bonus part
    else:
        segments, total_cost = segment(seq, args.penalty, args.max_len)
        plot_one_signal(segments)

    print_segments(segments,total_cost)


