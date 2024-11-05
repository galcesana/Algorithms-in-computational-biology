import numpy as np
import argparse

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

def print_segments(s: np.ndarray, c: float) -> None:
    """
    Print segments array with 3 decimal places.
    Each row contains [start, end, average] for single-channel or [start, end] for multi-channel.

    Args:
        s: numpy array of shape (n_segments, 2 or 3)
        c: Total cost value of optimal segmentation
    """
    for row in s:
        if len(row) == 3:  # Single-channel
            print(f"{int(row[0])} {int(row[1])} {np.round(row[2], 3)}")
        elif len(row) == 2:  # Multi-channel
            print(f"{int(row[0])} {int(row[1])}")
    print(np.round(c, 3))

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
    n = len(x)
    c = np.zeros(n + 1)
    t = np.zeros(n + 1, dtype=int)
    dp = np.full(n + 1, float('inf'))
    dp[0] = 0

    for i in range(1, n + 1):
        for j in range(max(0, i - q), i):
            segment_mean = np.mean(x[j:i])
            segment_cost = np.sum((x[j:i] - segment_mean) ** 2) + p
            cost = dp[j] + segment_cost
            if cost < dp[i]:
                dp[i] = cost
                t[i] = j

    segments = []
    idx = n
    while idx > 0:
        start = t[idx]
        end = idx
        segment_mean = np.mean(x[start:end])
        segments.append([start, end - 1, segment_mean])
        idx = start

    segments.reverse()
    return np.array(segments), dp[n]

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
    n, d = x.shape
    dp = np.full(n + 1, float('inf'))
    dp[0] = 0
    t = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        for j in range(max(0, i - q), i):
            segment_mean = np.mean(x[j:i], axis=0)
            segment_cost = np.sum((x[j:i] - segment_mean) ** 2) + p
            cost = dp[j] + segment_cost
            if cost < dp[i]:
                dp[i] = cost
                t[i] = j

    segments = []
    idx = n
    while idx > 0:
        start = t[idx]
        end = idx
        segments.append([start, end - 1])
        idx = start

    segments.reverse()
    return np.array(segments), dp[n]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process signal data from a text file.')
    parser.add_argument('--filepath', type=str, required=True, help='Path to the input text file')
    parser.add_argument('--penalty', type=float, required=True, help='Penalty parameter')
    parser.add_argument('--max_len', type=int, required=True, help='Maximum segment length')
    parser.add_argument('--is_multi_channel', type=bool, required=False, default=False, help='call segment_multi_channel')
    args = parser.parse_args()

    seq = read_signal_file(args.filepath)

    if args.is_multi_channel:
        segments, total_cost = segment_multi_channel(seq, args.penalty, args.max_len)
    else:
        segments, total_cost = segment(seq, args.penalty, args.max_len)

    print_segments(segments, total_cost)
