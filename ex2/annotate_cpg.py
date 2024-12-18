import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import gzip
import argparse

def parse_fasta_file(file_path):
    """
    Parses a FASTA file (plain or gzipped) and returns a mapping of sequence identifiers to nucleotide sequences.
    """
    sequences = {}
    # Determine the file opener based on the file extension
    file_opener = gzip.open if file_path.endswith(".gz") else open
    with file_opener(file_path, "rt") as file_handle:
        # Parse the file using Biopython's SeqIO module
        for record in SeqIO.parse(file_handle, "fasta"):
            sequences[record.id] = str(record.seq)
    return sequences

def split_data(data: list[tuple[str, str]], train_ratio: float = 0.8):
    """
    Splits the input data into training and test sets.

    Parameters:
        data (list[tuple[str, str]]): The input data as a list of (sequence, label) pairs.
        train_ratio (float): The ratio of data to include in the training set (default is 0.8).

    Returns:
        tuple: A tuple containing two lists:
            - training_data (list[tuple[str, str]]): 80% of the data for training.
            - test_data (list[tuple[str, str]]): 20% of the data for testing.
    """
    import random

    # Shuffle the data for randomness
    random.shuffle(data)

    # Calculate the split index
    split_index = int(len(data) * train_ratio)

    # Split the data into training and test sets
    training_data = data[:split_index]
    test_data = data[split_index:]

    # Add reverse complement sequences to the training data
    reverse_complements = [
        (str(Seq(sequence).reverse_complement()), label[::-1])
        for sequence, label in training_data
    ]

    # Add reverse complements to training data
    training_data.extend(reverse_complements)

    return training_data, test_data

def prepare_training_data(sequence_file: str, label_file: str):
    """
    Aligns nucleotide sequences with corresponding labels to create a training dataset.

    Parameters:
        sequence_file (str): Path to the FASTA file containing sequences.
        label_file (str): Path to the FASTA file containing labels.

    Returns:
        list[tuple[str, str]]: A list of tuples where each tuple contains a DNA sequence and its label.
    """
    sequences = parse_fasta_file(sequence_file)  # Parse sequences
    labels = parse_fasta_file(label_file)  # Parse labels

    if sequences.keys() != labels.keys():
        # Ensure sequence IDs match label IDs
        raise ValueError("Mismatch between sequence IDs and label IDs in the provided files.")

    # Create pairs of sequences and labels
    training_data = [(sequences[seq_id], labels[seq_id]) for seq_id in sequences]

    # Generate reverse complements for sequences and labels
    reverse_complements = [
        (str(Seq(sequence).reverse_complement()), label[::-1])
        for sequence, label in training_data
    ]
    # Add reverse complements
    training_data.extend(reverse_complements)

    return training_data

def train_hmm(training_data):
    """
    Trains an HMM using the given labeled training data.

    Parameters:
        training_data (list[tuple[str, str]]): A list of sequences and their corresponding labels.

    Returns:
        tuple: Transition probabilities, emission probabilities.
    """
    states = ["C", "N"]  # Define states: CpG and non-CpG
    nucleotides = ["A", "T", "G", "C", "N"]  # Possible nucleotide values
    transition_counts = {s1: {s2: 0 for s2 in states} for s1 in states}  # Transition counts
    emission_counts = {s: {n: 0 for n in nucleotides} for s in states}  # Emission counts
    state_counts = {s: 0 for s in states}  # State occurrence counts

    for sequence, labels in training_data:
        for i, (nucleotide, state) in enumerate(zip(sequence, labels)):
            if state not in states:
                print(f"Unexpected state found: {state}")
                continue
            state_counts[state] += 1
            if nucleotide not in emission_counts[state]:
                print(f"Unexpected nucleotide found: {nucleotide}")
                continue  # Skip invalid states
            emission_counts[state][nucleotide] += 1  # Count emissions
            if i > 0:  # Count transitions
                prev_state = labels[i - 1]
                if prev_state not in states:
                    continue
                transition_counts[prev_state][state] += 1

    # Normalize counts to get probabilities
    transition_probs = {
        s1: {s2: transition_counts[s1][s2] / sum(transition_counts[s1].values()) if sum(transition_counts[s1].values()) > 0 else 0 for s2 in states}
        for s1 in states
    }
    emission_probs = {
        s: {n: emission_counts[s][n] / state_counts[s] if state_counts[s] > 0 else 0 for n in nucleotides}
        for s in states
    }

    total_states = sum(state_counts.values())
    start_probs = {state: state_counts[state] / total_states for state in states}

    return transition_probs, emission_probs, start_probs

def viterbi(sequence, states, start_probs, trans_probs, emit_probs):
    """
    Performs the Viterbi algorithm for HMM decoding.

    Parameters:
        sequence (str): The observed sequence.
        states (list): List of states.
        start_probs (dict): Initial probabilities for each state.
        trans_probs (dict): Transition probabilities.
        emit_probs (dict): Emission probabilities.

    Returns:
        str: The most likely sequence of states.
    """
    n = len(sequence)
    dp = np.zeros((len(states), n))  # DP table for probabilities
    pointers = np.zeros((len(states), n), dtype=int)  # Pointers for traceback

    # Initialization
    for i, state in enumerate(states):
        dp[i, 0] = start_probs[state] * emit_probs[state][sequence[0]]

    # Recursion
    for t in range(1, n):
        for i, curr_state in enumerate(states):
            max_prob, max_state = max(
                (dp[j, t - 1] * trans_probs[prev_state][curr_state] * emit_probs[curr_state][sequence[t]], j)
                for j, prev_state in enumerate(states)
            )
            dp[i, t] = max_prob
            pointers[i, t] = max_state

    # Traceback
    best_path = []
    best_last_state = np.argmax(dp[:, -1])  # Find the best final state
    for t in range(n - 1, -1, -1):
        best_path.insert(0, states[best_last_state])
        best_last_state = pointers[best_last_state, t]
    return "".join(best_path)

def annotate_sequence(model, sequence):
    """
    Annotates a DNA sequence with CpG island predictions using the Viterbi algorithm.
    """
    states, start_probs, trans_probs, emit_probs = model
    return viterbi(sequence, states, start_probs, trans_probs, emit_probs)

def annotate_fasta_file(model, input_path, output_path):
    """
    Annotates all sequences in a FASTA file with CpG island predictions.
    """
    sequences = parse_fasta_file(input_path)
    with gzip.open(output_path, "wt") as gzipped_file:
        for seq_id, sequence in sequences.items():
            annotation = annotate_sequence(model, sequence)
            gzipped_file.write(f">{seq_id}\n{annotation}\n")

def test_classifier(data):
    from sklearn.metrics import precision_score, recall_score, f1_score
    training_data, test_data = split_data(data)

    transition_probs, emission_probs, start_probs = train_hmm(training_data)

    # Print transition and emission probability matrices
    print_matrix(transition_probs, "Transition Probabilities", row_labels=["C", "N"],
                 column_labels=["C", "N"])
    print_matrix(emission_probs, "Emission Probabilities", row_labels=["C", "N"],
                 column_labels=["A", "T", "G", "C", "N"])

    states = ["C", "N"]
    model = (states, start_probs, transition_probs, emission_probs)

    # Extract sequences and labels for testing
    test_sequences = [seq for seq, label in test_data]
    true_labels = [label for seq, label in test_data]

    # Predict labels using the trained classifier
    predicted_labels = [annotate_sequence(model, seq) for seq in test_sequences]

    # Calculate evaluation metrics
    # Flatten all labels for metric calculation
    true_flat = ''.join(true_labels)
    pred_flat = ''.join(predicted_labels)

    accuracy = sum(1 for t, p in zip(true_flat, pred_flat) if t == p) / len(true_flat)

    precision = precision_score(list(true_flat), list(pred_flat), average='weighted', zero_division=0)
    recall = recall_score(list(true_flat), list(pred_flat), average='weighted', zero_division=0)
    f1 = f1_score(list(true_flat), list(pred_flat), average='weighted', zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    # Print evaluation metrics
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    return metrics

def print_matrix(matrix, title, row_labels=None, column_labels=None):
    """
    Prints a formatted matrix with a title.
    Parameters:
        matrix (dict): The matrix to be printed (dictionary of dictionaries).
        title (str): The title of the matrix to be displayed.
        row_labels (list): Optional list of row labels for the matrix.
        column_labels (list): Optional list of column labels for the matrix.
    """
    print(f"\n{title}:")

    # Use provided column labels or matrix keys as default
    if column_labels is None:
        column_labels = list(next(iter(matrix.values())).keys())
    if row_labels is None:
        row_labels = list(matrix.keys())

    # Print header
    print("   ", "  ".join(f"{col:>8}" for col in column_labels))

    # Print each row
    for row_key in row_labels:
        row = "  ".join(f"{matrix[row_key][col]:8.4f}" for col in column_labels)
        print(f"{row_key:>3} {row}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict CpG islands in DNA sequences.")
    parser.add_argument("--fasta_path", type=str, help="Path to the input FASTA file containing DNA sequences.")
    parser.add_argument("--output_file", type=str, help="Path to the output FASTA file for saving predictions.")

    args = parser.parse_args()

    training_sequences_path = "data/CpG-islands.2K.seq.fa.gz"
    training_labels_path = "data/CpG-islands.2K.lbl.fa.gz"

    # Prepare training data
    training_data = prepare_training_data(training_sequences_path, training_labels_path)

    if args.fasta_path == "test":
        test_classifier(training_data)

    else:
        # Train HMM
        transition_probs, emission_probs, start_probs = train_hmm(training_data)

        # Print transition and emission probability matrices
        print_matrix(transition_probs, "Transition Probabilities", row_labels=["C", "N"],
                     column_labels=["C", "N"])
        print_matrix(emission_probs, "Emission Probabilities", row_labels=["C", "N"],
                     column_labels=["A", "T", "G", "C", "N"])
        states = ["C", "N"]

        model = (states, start_probs, transition_probs, emission_probs)

        # Annotate sequences and save predictions
        annotate_fasta_file(model, args.fasta_path, args.output_file)