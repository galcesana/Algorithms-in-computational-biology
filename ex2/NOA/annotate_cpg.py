import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import gzip
import argparse
from sklearn.metrics import balanced_accuracy_score

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
        s1: {
            s2: (
                transition_counts[s1][s2] / sum(transition_counts[s1].values()) if sum(transition_counts[s1].values()) > 0 else 0
            )
            for s2 in states
        }
        for s1 in states
    }

    # **Adjusting the transition probabilities for `C`**
    # Give higher probability to staying in `C`
    c_total = sum(transition_counts["C"].values())
    if c_total > 0:
        transition_probs["C"]["C"] = (transition_counts["C"]["C"] + 1000000000) / (c_total+1000000000)
        transition_probs["C"]["N"] = transition_probs["C"]["N"] - 0.0021


    # **Adjusting the transition probabilities for `N` to `C`**
    n_total = sum(transition_counts["N"].values())
    if n_total > 0:
        # Add a weight to increase the probability of transitioning from N to C
        transition_probs["N"]["C"] = (transition_counts["N"]["C"] + 16600000) / (n_total+16600000)
        transition_probs["N"]["N"] = transition_probs["N"]["N"]- 0.007

    total_states = sum(state_counts.values())
    start_probs = {state: state_counts[state] / total_states for state in states}

    emission_probs = {
        s: {n: emission_counts[s][n] / state_counts[s] if state_counts[s] > 0 else 0 for n in nucleotides}
        for s in states
    }

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


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score

def evaluate_model(model, sequence_file, label_file):
    """
    Compares predicted labels with the actual labels and calculates evaluation metrics.

    Parameters:
        model (tuple): The trained HMM model (states, start_probs, trans_probs, emit_probs).
        sequence_file (str): Path to the FASTA file containing sequences.
        label_file (str): Path to the FASTA file containing true labels.

    Returns:
        dict: A dictionary with evaluation metrics (accuracy, precision, recall, F1-score, balanced accuracy).
    """

    sequences = parse_fasta_file(sequence_file)
    true_labels = parse_fasta_file(label_file)

    # Compare predictions with true labels
    all_true_labels = []
    all_predicted_labels = []

    for seq_id in sequences.keys():
        sequence = sequences[seq_id]
        true_label = true_labels[seq_id]
        predicted_label = annotate_sequence(model, sequence)

        all_true_labels.extend(true_label)
        all_predicted_labels.extend(predicted_label)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    balanced_accuracy = balanced_accuracy_score(all_true_labels, all_predicted_labels)
    precision = precision_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    # Print evaluation metrics
    print("Comparison Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}")

    return metrics




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict CpG islands in DNA sequences.")
    parser.add_argument("--fasta_path", type=str, help="Path to the input FASTA file containing DNA sequences.")
    parser.add_argument("--output_file", type=str, help="Path to the output FASTA file for saving predictions.")

    args = parser.parse_args()

    training_sequences_path = "data/CpG-islands.2K.seq.fa.gz"
    training_labels_path = "data/CpG-islands.2K.lbl.fa.gz"

    # Prepare training data
    training_data = prepare_training_data(training_sequences_path, training_labels_path)

    # Train HMM
    transition_probs, emission_probs, start_probs = train_hmm(training_data)

    states = ["C", "N"]

    model = (states, start_probs, transition_probs, emission_probs)

    if args.output_file == "evaluate":
        # Print transition and emission probability matrices
        print_matrix(transition_probs, "Transition Probabilities", row_labels=["C", "N"],
                     column_labels=["C", "N"])
        print_matrix(emission_probs, "Emission Probabilities", row_labels=["C", "N"],
                     column_labels=["A", "T", "G", "C", "N"])
        # Evaluate the model
        print("Evaluating the model...")
        metrics = evaluate_model(model, args.fasta_path, "CpG-islands.2K.chr2.lbl.fa.gz")
        print("Evaluation complete.")

    else:
        # Annotate sequences and save predictions
        annotate_fasta_file(model, args.fasta_path, args.output_file)