import argparse
from Bio import SeqIO

def parse_fasta_file(file_path: str):
    """
    Parses a FASTA file (plain or gzipped) and returns a mapping of sequence
    identifiers to nucleotide sequences.

    Parameters:
        file_path (str): The path to the FASTA file.

    Returns:
        dict: A dictionary with sequence IDs as keys and DNA sequences as values.
    """
    sequences = {}

    with open(file_path, 'r') as file_handle:
        for record in SeqIO.parse(file_handle, "fasta"):
            sequences[record.id] = str(record.seq)

    return sequences


def compute_pairwise_distance(seq1: str, seq2: str) -> float:
    """
    Computes a simple p-distance (Hamming distance / length) between two sequences.

    If sequences are of differing lengths, it will only compare over the minimum length.
    """
    length = min(len(seq1), len(seq2))
    mismatches = sum(1 for i in range(length) if seq1[i] != seq2[i])
    return mismatches / length if length > 0 else 0.0


def build_distance_matrix(sequences: dict):
    """
    Given a dictionary of {name: sequence}, compute and return a dictionary of dictionaries
    representing the distance matrix. Also returns a list of original labels for convenience.

    Returns:
        distance_matrix: dict of dict, where distance_matrix[a][b] = distance between a and b
        labels: list of sequence IDs
    """
    labels = list(sequences.keys())
    distance_matrix = {a: {} for a in labels}

    for i in range(len(labels)):
        for j in range(i, len(labels)):
            seq_i = sequences[labels[i]]
            seq_j = sequences[labels[j]]
            dist_ij = compute_pairwise_distance(seq_i, seq_j)
            distance_matrix[labels[i]][labels[j]] = dist_ij
            distance_matrix[labels[j]][labels[i]] = dist_ij

    return distance_matrix, labels


def upgma(distance_matrix, labels):
    """
    Construct a UPGMA tree from the given distance matrix and return
    a Newick-formatted string.

    UPGMA steps:
    1) Initialize each sequence as its own cluster.
    2) Find the pair of clusters (i,j) with the smallest distance.
    3) Merge clusters i and j into a new cluster u; compute distances to other clusters by
       size-weighted average.
    4) Repeat until only one cluster remains.

    The Newick string includes branch lengths. UPGMA assumes an ultrametric tree, so
    we simply assign each branch half of the distance used to merge the two clusters.
    """
    # Keep track of cluster sizes and the current Newick representation for each cluster.
    # clusters dict: clusterName -> (newick_string, size_of_cluster)
    clusters = {label: (label, 1) for label in labels}

    # We'll keep distance_matrix as a dict of dict of floats, which we update as we merge.
    current_labels = labels[:]  # working list of cluster labels

    # While more than one cluster remains, merge the closest two
    while len(current_labels) > 1:
        # 1) Find the pair of distinct clusters with the smallest distance
        min_dist = float('inf')
        to_merge = (None, None)
        for i in range(len(current_labels)):
            for j in range(i + 1, len(current_labels)):
                a = current_labels[i]
                b = current_labels[j]
                if distance_matrix[a][b] < min_dist:
                    min_dist = distance_matrix[a][b]
                    to_merge = (a, b)

        c1, c2 = to_merge
        # The two clusters we will merge
        (newick_c1, size_c1) = clusters[c1]
        (newick_c2, size_c2) = clusters[c2]

        # 2) Create a new cluster label (any unique string)
        new_label = f"({newick_c1}:{min_dist/2:.3f},{newick_c2}:{min_dist/2:.3f})"

        # 3) Compute size of new cluster
        new_size = size_c1 + size_c2

        # 4) Update distances from the new cluster to the others by size-weighted average
        clusters[new_label] = (new_label, new_size)
        distance_matrix[new_label] = {}
        for other in current_labels:
            if other not in [c1, c2]:
                dist_to_new = (
                    (distance_matrix[c1][other] * size_c1) +
                    (distance_matrix[c2][other] * size_c2)
                ) / (size_c1 + size_c2)
                distance_matrix[new_label][other] = dist_to_new
                distance_matrix[other][new_label] = dist_to_new

        # 5) Remove old clusters from the distance matrix and current labels
        current_labels.remove(c1)
        current_labels.remove(c2)
        del distance_matrix[c1]
        del distance_matrix[c2]
        for k in distance_matrix:
            if c1 in distance_matrix[k]:
                del distance_matrix[k][c1]
            if c2 in distance_matrix[k]:
                del distance_matrix[k][c2]

        # 6) Add new cluster to the list
        current_labels.append(new_label)

    # Now there's only one cluster label left
    return current_labels[0] + ";"


def neighbor_joining(distance_matrix, labels):
    """
    Construct a tree using the Neighbor Joining (NJ) method and return
    a Newick-formatted string.

    Steps:
    1) Compute r_i for each cluster i (the sum of distances from i to all other clusters).
    2) Construct Q-matrix: Q(i,j) = (N-2)*D(i,j) - r_i - r_j
    3) Find the pair (i,j) with the smallest Q(i,j).
    4) Create a new node 'u' that joins i and j. The branch lengths from u to i and u to j are:
         dist(u,i) = 0.5 * D(i,j) + (r_i - r_j)/(2*(N-2))
         dist(u,j) = 0.5 * D(i,j) + (r_j - r_i)/(2*(N-2))
       (There are variants of the formula, but this is the commonly cited one.)
    5) Compute distances from the new node u to every other node k:
         D(u,k) = 0.5 * [ D(i,k) + D(j,k) - D(i,j) ]
    6) Remove i and j, add u in the distance matrix and repeat.

    We keep track of the partial Newick string as we merge clusters.
    """
    # Initialize each label's "Newick" representation
    newick_dict = {lbl: lbl for lbl in labels}

    active_labels = labels[:]

    while len(active_labels) > 2:
        N = len(active_labels)

        # 1) Compute r_i
        r = {}
        for i in active_labels:
            r[i] = sum(distance_matrix[i][j] for j in active_labels if j != i)

        # 2) Compute Q-matrix
        Q = {}
        min_val = float('inf')
        pair_to_join = (None, None)
        for i in active_labels:
            Q[i] = {}
            for j in active_labels:
                if i == j:
                    Q[i][j] = 0
                else:
                    Q[i][j] = (N - 2) * distance_matrix[i][j] - r[i] - r[j]
                if i != j and Q[i][j] < min_val:
                    min_val = Q[i][j]
                    pair_to_join = (i, j)

        i, j = pair_to_join

        # 3) Distances from new node u to i and j
        dist_ij = distance_matrix[i][j]
        dist_i_u = 0.5 * dist_ij + (r[i] - r[j]) / (2 * (N - 2))
        dist_j_u = 0.5 * dist_ij + (r[j] - r[i]) / (2 * (N - 2))

        # 4) Build new node label in Newick format
        new_label = f"({newick_dict[i]}:{dist_i_u:.3f},{newick_dict[j]}:{dist_j_u:.3f})"

        # 5) Compute distances from the new node to all other nodes k
        distance_matrix[new_label] = {}
        for k in active_labels:
            if k not in (i, j):
                d_u_k = 0.5 * (distance_matrix[i][k] + distance_matrix[j][k] - dist_ij)
                distance_matrix[new_label][k] = d_u_k
                distance_matrix[k][new_label] = d_u_k

        # Store the newick representation for the new node
        newick_dict[new_label] = new_label

        # 6) Remove i and j from the distance matrix
        active_labels.remove(i)
        active_labels.remove(j)
        for label in active_labels:
            del distance_matrix[label][i]
            del distance_matrix[label][j]
        del distance_matrix[i]
        del distance_matrix[j]

        # 7) Add new node to the distance matrix
        active_labels.append(new_label)

    # Finally, we have two labels left; join them
    if len(active_labels) == 2:
        a, b = active_labels
        dist_ab = distance_matrix[a][b]
        final_tree = f"({newick_dict[a]}:{dist_ab/2:.5f},{newick_dict[b]}:{dist_ab/2:.5f});"
    else:
        # If there's only one label, just return it
        final_tree = active_labels[0] + ";"

    return final_tree


def gen_newick_tree(fasta_path: str, algo: str):
    """
    Reads a FASTA file, computes a distance matrix, and generates a Newick-format
    UPGMA or NJ tree.

    Parameters:
        fasta_path (str): Path to the FASTA file.
        algo (str): Which algorithm to use for tree construction. Either "nj" or "upgma".

    Prints:
        str: The Newick-format tree string with original leaf names and
             no internal node names (other than automatically assigned).
    """
    # 1) Parse sequences
    sequences = parse_fasta_file(fasta_path)
    if len(sequences) == 0:
        print(";")
        return

    # 2) Build the initial distance matrix
    distance_matrix, labels = build_distance_matrix(sequences)

    # 3) Build the tree using the selected algorithm
    if algo.lower() == "upgma":
        newick_str = upgma(distance_matrix, labels)
    elif algo.lower() == "nj":
        newick_str = neighbor_joining(distance_matrix, labels)
    else:
        raise ValueError("Unknown algorithm. Use 'nj' or 'upgma'.")

    # 4) Print the resulting Newick-format tree
    print(newick_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path", type=str,
                        help="Path to FASTA file containing multiple DNA sequences.",
                        required=True)
    parser.add_argument("--algo", type=str,
                        help="Either 'nj' or 'upgma'.",
                        required=True)

    args = parser.parse_args()
    gen_newick_tree(args.fasta_path, args.algo)
