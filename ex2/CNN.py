from Bio import SeqIO  # pip install biopython
import argparse
import gzip

# from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random


from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, TimeDistributed, Input, Activation
from tensorflow.keras.layers import MaxPooling1D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

def parse_fasta_file(file_path: str)->dict:
    """
    Parses a FASTA file (plain or gzipped) and returns a mapping of sequence identifiers to nucleotide sequences.

    Parameters:
        file_path (str): The path to the FASTA file.

    Returns:
        dict: A dictionary with sequence IDs as keys and DNA sequences as values.
    """
    sequences = {}

    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as file_handle:
            for record in SeqIO.parse(file_handle, "fasta"):
                sequences[record.id] = str(record.seq)
    else:
        with open(file_path, 'r') as file_handle:
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
    sequences = parse_fasta_file(sequence_file)
    labels = parse_fasta_file(label_file)

    if sequences.keys() != labels.keys():
        raise ValueError("Mismatch between sequence IDs and label IDs in the provided files.")

    return [(sequences[seq_id], labels[seq_id]) for seq_id in sequences]

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
    # Shuffle the data for randomness
    random.shuffle(data)

    # Calculate the split index
    split_index = int(len(data) * train_ratio)

    # Split the data into training and test sets
    training_data = data[:split_index]
    test_data = data[split_index:]

    return training_data, test_data


def one_hot_encode_nucleotide(nuc: str):
    """
    One-hot encode a single nucleotide: A, C, G, T.
    The order chosen is A,C,G,T.
    Returns a numpy array of shape (4,).
    """
    mapping = {'A':[1,0,0,0],
               'C':[0,1,0,0],
               'G':[0,0,1,0],
               'T':[0,0,0,1]}
    return np.array(mapping.get(nuc, [0,0,0,0])) # if unknown char, all zeros


def extract_features_and_labels(data):
    """
    Convert the list of (sequence, label) into a feature matrix X and label vector y.
    Here, we assume all sequences are the same length.

    X: shape (num_sequences, seq_length, 4)
    y: shape (num_sequences, seq_length) - binary (0 or 1)
    """
    # Check sequence lengths
    seq_lengths = [len(seq) for seq, lbl in data]
    if len(set(seq_lengths)) > 1:
        raise ValueError(
            "Sequences have different lengths. For this CNN approach, all must have the same length.")
    seq_length = seq_lengths[0]

    num_sequences = len(data)
    X = np.zeros((num_sequences, seq_length, 4), dtype=np.float32)
    y = np.zeros((num_sequences, seq_length), dtype=np.int32)

    label_map = {'C': 1, 'N': 0}

    for i, (seq, lbl) in enumerate(data):
        for j, (nuc, l) in enumerate(zip(seq, lbl)):
            X[i, j, :] = one_hot_encode_nucleotide(nuc)
            y[i, j] = label_map[l]

    return X, y

def build_cnn_model(input_length):
    """
    Builds a CNN model for per-nucleotide classification.
    Input shape: (sequence_length, 4)
    Output shape: (sequence_length, 1) with sigmoid activation for binary classification.
    """
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(input_length, 4)))
    model.add(MaxPooling1D(pool_size=2))
    # After pooling, the sequence length is reduced by a factor of 2
    # We need to ensure output has the same sequence length as input for per-nucleotide prediction.
    # Instead, let's remove pooling to keep the sequence length the same.
    # We'll rely on convolution alone for now.

    # Adjusted model without pooling (to maintain same length output):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(input_length,4)))
    # Another conv layer for better feature extraction
    model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    # Output layer: per-position binary classification
    # Dense layer must be applied to each timestep. We can use a TimeDistributed layer for this.
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    return model

def train_classifier(training_data):
    """
    Trains a classifier to identify CpG islands in DNA sequences.

    Parameters:
        training_data (list[tuple[str, str]]): Training data consisting of sequences and their labels.

    Returns:
        object: Your trained classifier model.
    """
    # TODO: Implement your model training logic here

    # Extract features and labels
    X, y = extract_features_and_labels(training_data)
    seq_length = X.shape[1]

    # Build the model
    model = build_cnn_model(seq_length)

    # Train the model
    # y needs to be expanded to (num_sequences, seq_length, 1) for binary crossentropy
    y_expanded = np.expand_dims(y, axis=-1)
    model.fit(X, y_expanded, epochs=5, batch_size=32, validation_split=0.2)

    return model

def annotate_sequence(model, sequence):
    """
    Annotates a DNA sequence with CpG island predictions.

    Parameters:
        model (object): Your trained classifier model.
        sequence (str): A DNA sequence to be annotated.

    Returns:
        str: A string of annotations, where 'C' marks a CpG island region and 'N' denotes non-CpG regions.
    """
    # TODO: Replace with your (hopefully better) prediction logic
    # For each nucleotide, predict 'C' or 'N'
    seq_length = len(sequence)
    X = np.zeros((1, seq_length, 4), dtype=np.float32)
    for i, nuc in enumerate(sequence):
        X[0, i, :] = one_hot_encode_nucleotide(nuc)

    # Predict probabilities
    predictions = model.predict(X)  # shape: (1, seq_length, 1)
    predictions = predictions[0, :, 0]  # shape: (seq_length,)

    # Threshold at 0.5: >0.5 = C, else N
    pred_labels = np.where(predictions > 0.5, 'C', 'N')
    return ''.join(pred_labels)

    #
    # annotations = ''.join(['C' if nucleotide == 'C' else 'N' for nucleotide in sequence])
    # return annotations

def annotate_fasta_file(model, input_path, output_path):
    """
    Annotates all sequences in a FASTA file with CpG island predictions.

    Parameters:
        model (object): A trained classifier model.
        input_path (str): Path to the input FASTA file.
        output_path (str): Path to the output FASTA file where annotations will be saved.

    Writes:
        A gzipped FASTA file containing predicted annotations for each input sequence.
    """
    sequences = parse_fasta_file(input_path)

    with gzip.open(output_path, 'wt') as gzipped_file:
        for seq_id, sequence in sequences.items():
            annotation = annotate_sequence(model, sequence)
            gzipped_file.write(f">{seq_id}\n{annotation}\n")

def test_classifier(data):
    training_data, test_data = split_data(data)

    classifier = train_classifier(training_data)

    # Extract sequences and labels for testing
    test_sequences = [seq for seq, label in test_data]
    true_labels = [label for seq, label in test_data]

    # Predict labels using the trained classifier
    predicted_labels = [annotate_sequence(classifier, seq) for seq in test_sequences]

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict CpG islands in DNA sequences.")
    parser.add_argument("--fasta_path", type=str, help="Path to the input FASTA file containing DNA sequences.")
    parser.add_argument("--output_file", type=str, help="Path to the output FASTA file for saving predictions.")

    args = parser.parse_args()

    training_sequences_path = r"data/CpG-islands.2K.seq.fa.gz"
    training_labels_path = r"data/CpG-islands.2K.lbl.fa.gz"

    # Prepare training data and train model
    data = prepare_training_data(training_sequences_path, training_labels_path)

    test_classifier(data)  # TODO only for inside testing

    # classifier = train_classifier(data)

    # annotate_fasta_file(classifier, args.fasta_path, args.output_file)
    # Annotate sequences and save predictions
