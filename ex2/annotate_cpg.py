from Bio import SeqIO  # pip install biopython
import argparse
import gzip

from sklearn.ensemble import RandomForestClassifier
import numpy as np


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

def extract_features_and_labels(training_data):
    """
    Convert the list of (sequence, label) into feature matrix X and label vector y.
    We will do a simple per-nucleotide classification:
    X: Each row represents one position in one sequence.
    y: The corresponding label ('C' or 'N').
    """
    X = []
    y = []
    for seq, lbl in training_data:
        if len(seq) != len(lbl):
            # Just a safety check
            continue
        for nuc, l in zip(seq, lbl):
            X.append(one_hot_encode_nucleotide(nuc))
            y.append(l)
    X = np.vstack(X)
    y = np.array(y)
    return X, y

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

    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
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
    X = [one_hot_encode_nucleotide(nuc) for nuc in sequence]
    X = np.vstack(X)
    predictions = model.predict(X)
    return ''.join(predictions)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict CpG islands in DNA sequences.")
    parser.add_argument("--fasta_path", type=str, help="Path to the input FASTA file containing DNA sequences.")
    parser.add_argument("--output_file", type=str, help="Path to the output FASTA file for saving predictions.")

    args = parser.parse_args()

    training_sequences_path = r"data/CpG-islands.2K.seq.fa.gz"
    training_labels_path = r"data/CpG-islands.2K.lbl.fa.gz"

    # Prepare training data and train model
    training_data = prepare_training_data(training_sequences_path, training_labels_path)
    classifier = train_classifier(training_data)

    # Annotate sequences and save predictions
    annotate_fasta_file(classifier, args.fasta_path, args.output_file)
