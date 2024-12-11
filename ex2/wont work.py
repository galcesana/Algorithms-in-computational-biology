import argparse
import gzip
from Bio import SeqIO
import numpy as np


def parse_fasta_file(file_path: str) -> dict:
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


# נקבע את האלפבית
nucs = ['A', 'C', 'G', 'T','N']
n_states = 2  # 0 - CpG, 1 - Non-CpG
n_symbols = len(nucs)


def initialize_hmm():
    # הסתברויות התחלה (pi)
    pi = np.array([0.5, 0.5])
    # מטריצת מעברים A (2x2)
    A = np.array([[0.9, 0.1],
                  [0.1, 0.9]])
    # מטריצת הפליטות B (2x4)
    B = np.array([[0.25, 0.25, 0.25, 0.25],
                  [0.25, 0.25, 0.25, 0.25]])
    return pi, A, B


def forward_pass(seq, pi, A, B):
    # אלגוריתם forward
    T = len(seq)
    alpha = np.zeros((T, n_states))
    # אתחול
    symbol_idx = nucs.index(seq[0])
    alpha[0, :] = pi * B[:, symbol_idx]

    for t in range(1, T):
        symbol_idx = nucs.index(seq[t])
        for j in range(n_states):
            alpha[t, j] = (alpha[t - 1, :] @ A[:, j]) * B[j, symbol_idx]
    return alpha


def backward_pass(seq, pi, A, B):
    # אלגוריתם backward
    T = len(seq)
    beta = np.zeros((T, n_states))
    beta[T - 1, :] = 1
    for t in range(T - 2, -1, -1):
        symbol_idx = nucs.index(seq[t + 1])
        for i in range(n_states):
            beta[t, i] = (A[i, :] * B[:, symbol_idx] * beta[t + 1, :]).sum()
    return beta


def baum_welch(sequences, n_iter=10):
    pi, A, B = initialize_hmm()

    for iteration in range(n_iter):
        # E-step accumulators
        pi_acc = np.zeros(n_states)
        A_acc = np.zeros_like(A)
        B_acc = np.zeros_like(B)
        gamma_sum = np.zeros(n_states)  # sum of gamma_t(i) over t for normalization of B

        for seq in sequences:
            # Forward-backward
            alpha = forward_pass(seq, pi, A, B)
            beta = backward_pass(seq, pi, A, B)
            T = len(seq)

            # gamma_t(i): הסתברות להיות במצב i בזמן t
            gamma = (alpha * beta) / (alpha[-1, :].sum())

            # xi_t(i,j): הסתברות להיות במצב i בזמן t ובמצב j בזמן t+1
            xi = np.zeros((T - 1, n_states, n_states))
            for t in range(T - 1):
                denom = (alpha[t, :] @ A @ (B[:, nucs.index(seq[t + 1])] * beta[t + 1, :]))
                for i in range(n_states):
                    for j in range(n_states):
                        xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, nucs.index(seq[t + 1])] * beta[
                            t + 1, j] / denom

            # הצטברות
            pi_acc += gamma[0, :]
            for i in range(n_states):
                for j in range(n_states):
                    A_acc[i, j] += xi[:, i, j].sum()

            for t in range(T):
                symbol_idx = nucs.index(seq[t])
                for i in range(n_states):
                    B_acc[i, symbol_idx] += gamma[t, i]

            gamma_sum += gamma.sum(axis=0)

        # M-step
        pi = pi_acc / pi_acc.sum()
        for i in range(n_states):
            A[i, :] = A_acc[i, :] / A_acc[i, :].sum() if A_acc[i, :].sum() > 0 else A[i, :]
        for i in range(n_states):
            B[i, :] = B_acc[i, :] / gamma_sum[i] if gamma_sum[i] > 0 else B[i, :]

    return pi, A, B


def viterbi(seq, pi, A, B):
    T = len(seq)
    delta = np.zeros((T, n_states))
    psi = np.zeros((T, n_states), dtype=int)
    delta[0, :] = pi * B[:, nucs.index(seq[0])]
    for t in range(1, T):
        symbol_idx = nucs.index(seq[t])
        for j in range(n_states):
            probs = delta[t - 1, :] * A[:, j]
            psi[t, j] = np.argmax(probs)
            delta[t, j] = np.max(probs) * B[j, symbol_idx]
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1, :])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def annotate_sequence(seq, pi, A, B):
    # שימוש ב-Viterbi להפקת המסלול הסביר ביותר: 0 -> CpG (C), 1-> Non-CpG (N)
    states = viterbi(seq, pi, A, B)
    annotation = ''.join(['C' if s == 0 else 'N' for s in states])
    return annotation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict CpG islands in DNA sequences using HMM + Baum-Welch.")
    parser.add_argument("--fasta_path", type=str,
                        help="Path to the input FASTA file containing DNA sequences.")
    parser.add_argument("--output_file", type=str,
                        help="Path to the output FASTA file for saving predictions.")
    args = parser.parse_args()

    # Read sequences
    seqs_dict = parse_fasta_file(args.fasta_path)
    seqs = list(seqs_dict.values())

    # Train HMM parameters using Baum-Welch (unsupervised)
    pi, A, B = baum_welch(seqs, n_iter=10)

    # Annotate sequences
    with gzip.open(args.output_file, 'wt') as gzipped_file:
        for seq_id, sequence in seqs_dict.items():
            annotation = annotate_sequence(sequence, pi, A, B)
            gzipped_file.write(f">{seq_id}\n{annotation}\n")
