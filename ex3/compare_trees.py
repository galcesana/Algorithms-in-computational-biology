import argparse
import os
import re
import matplotlib.pyplot as plt
from Bio import SeqIO

###############################################################################
#               Step 1: FASTA Parsing & Distance Matrix Computation
###############################################################################

def parse_fasta_file(file_path: str):
    """
    Parses a FASTA file and returns a mapping of sequence identifiers to sequences.
    """
    sequences = {}
    with open(file_path, 'r') as file_handle:
        for record in SeqIO.parse(file_handle, "fasta"):
            sequences[record.id] = str(record.seq)
    return sequences

def compute_pairwise_distance(seq1: str, seq2: str) -> float:
    """
    Computes a simple p-distance (Hamming distance / length) between two sequences.
    """
    length = min(len(seq1), len(seq2))
    mismatches = sum(1 for i in range(length) if seq1[i] != seq2[i])
    return mismatches / length if length > 0 else 0.0

def build_distance_matrix(sequences: dict):
    """
    Returns a dictionary-of-dictionaries distance matrix and a list of labels.
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

###############################################################################
#                   Step 2: UPGMA & Neighbor Joining Implementations
###############################################################################

def upgma(distance_matrix, labels):
    """
    Constructs a UPGMA tree (Newick) with branch lengths of 3 decimal places.
    """
    clusters = {label: (label, 1) for label in labels}
    current_labels = labels[:]

    while len(current_labels) > 1:
        # Find the pair with the smallest distance
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
        (newick_c1, size_c1) = clusters[c1]
        (newick_c2, size_c2) = clusters[c2]

        # Merge and build new Newick label
        new_label = f"({newick_c1}:{min_dist/2:.3f},{newick_c2}:{min_dist/2:.3f})"
        new_size = size_c1 + size_c2
        clusters[new_label] = (new_label, new_size)

        # Update distances (size-weighted average)
        distance_matrix[new_label] = {}
        for other in current_labels:
            if other not in [c1, c2]:
                dist_to_new = (
                    (distance_matrix[c1][other] * size_c1) +
                    (distance_matrix[c2][other] * size_c2)
                ) / (size_c1 + size_c2)
                distance_matrix[new_label][other] = dist_to_new
                distance_matrix[other][new_label] = dist_to_new

        current_labels.remove(c1)
        current_labels.remove(c2)
        del distance_matrix[c1]
        del distance_matrix[c2]
        for k in distance_matrix:
            distance_matrix[k].pop(c1, None)
            distance_matrix[k].pop(c2, None)

        current_labels.append(new_label)

    return current_labels[0] + ";"

def neighbor_joining(distance_matrix, labels):
    """
    Constructs a Neighbor Joining tree (Newick) with branch lengths of 3 decimal places.
    """
    newick_dict = {lbl: lbl for lbl in labels}
    active_labels = labels[:]

    while len(active_labels) > 2:
        N = len(active_labels)
        # Compute r_i
        r = {}
        for i in active_labels:
            r[i] = sum(distance_matrix[i][j] for j in active_labels if j != i)

        # Compute Q-matrix
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

        # Distances
        dist_ij = distance_matrix[i][j]
        dist_i_u = 0.5 * dist_ij + (r[i] - r[j]) / (2 * (N - 2))
        dist_j_u = 0.5 * dist_ij + (r[j] - r[i]) / (2 * (N - 2))

        # New node in Newick format
        new_label = f"({newick_dict[i]}:{dist_i_u:.3f},{newick_dict[j]}:{dist_j_u:.3f})"
        newick_dict[new_label] = new_label

        # D(u,k)
        distance_matrix[new_label] = {}
        for k in active_labels:
            if k not in (i, j):
                d_u_k = 0.5 * (distance_matrix[i][k] + distance_matrix[j][k] - dist_ij)
                distance_matrix[new_label][k] = d_u_k
                distance_matrix[k][new_label] = d_u_k

        # Remove i, j
        active_labels.remove(i)
        active_labels.remove(j)
        for label in active_labels:
            distance_matrix[label].pop(i, None)
            distance_matrix[label].pop(j, None)
        del distance_matrix[i]
        del distance_matrix[j]

        # Add new node
        active_labels.append(new_label)

    # Join the final two
    if len(active_labels) == 2:
        a, b = active_labels
        dist_ab = distance_matrix[a][b]
        final_tree = f"({newick_dict[a]}:{dist_ab/2:.3f},{newick_dict[b]}:{dist_ab/2:.3f});"
    else:
        final_tree = active_labels[0] + ";"

    return final_tree

###############################################################################
#      Step 3: Parsing Newick & Plotting a Simple Horizontal Cladogram
###############################################################################

class TreeNode:
    """
    A simple tree node to store:
      - name: str (empty if internal node)
      - length: float (branch length from parent)
      - children: list of TreeNode
    """
    def __init__(self, name="", length=0.0):
        self.name = name
        self.length = length
        self.children = []

def parse_newick(newick_str):
    """
    Parses a (single) Newick string and returns the root TreeNode.

    Limitations:
      - Ignores any comment fields in brackets.
      - Assumes no internal node labels (like "Node1"). If found, they'll be ignored or
        interpreted as leaf names with length=0.
      - Works best for the typical format produced by our UPGMA/NJ code above.

    Basic approach: recursively parse parentheses. For something like:
      (A:0.1,(B:0.2,C:0.3):0.4);

    We'll build a structure of nested TreeNodes with branch lengths.

    Returns:
        TreeNode: root of the parsed tree
    """

    # Remove trailing semicolon if present
    newick_str = newick_str.strip()
    if newick_str.endswith(";"):
        newick_str = newick_str[:-1]

    # A recursive function that parses a subtree starting at current index.
    # Returns (TreeNode, next_index).
    def parse_subtree(s, idx=0, parent_length=0.0):
        """
        s: the full newick string (without final semicolon).
        idx: current position in the string.
        parent_length: the branch length to the parent node (0 if root).

        We'll parse either:
          - ( subtree1, subtree2, ... ) : length
          - or a leaf: name : length
        """
        node = TreeNode(length=parent_length)
        while idx < len(s):
            if s[idx] == '(':
                # Parse a group of children
                idx += 1  # skip '('
                # Keep parsing children separated by commas until we hit ')'
                while True:
                    child, idx = parse_subtree(s, idx)
                    node.children.append(child)
                    if s[idx] == ',':
                        idx += 1  # skip comma
                    elif s[idx] == ')':
                        idx += 1  # skip ')'
                        break
                # Now we might see ":<number>" for the length of this group
                if idx < len(s) and s[idx] == ':':
                    # parse branch length
                    idx += 1
                    length_str = []
                    while idx < len(s) and (s[idx].isdigit() or s[idx] in '.eE-+'):
                        length_str.append(s[idx])
                        idx += 1
                    blen = float(''.join(length_str)) if length_str else 0.0
                    node.length = blen
                return node, idx

            # If we hit a parenthesis close or comma, we've ended the current subtree
            if s[idx] in [')', ',', ';']:
                return node, idx

            # Otherwise, we might be parsing a leaf "Name:0.123"
            # Grab the leaf name up to ':' or ',' or ')' ...
            name_buf = []
            while idx < len(s) and s[idx] not in [':', ',', '(', ')']:
                name_buf.append(s[idx])
                idx += 1
            node.name = ''.join(name_buf).strip()

            # Now see if there's a colon (branch length)
            blen = 0.0
            if idx < len(s) and s[idx] == ':':
                idx += 1
                length_str = []
                while idx < len(s) and (s[idx].isdigit() or s[idx] in '.eE-+'):
                    length_str.append(s[idx])
                    idx += 1
                blen = float(''.join(length_str)) if length_str else 0.0
            node.length = blen
            return node, idx

        return node, idx

    root, _ = parse_subtree(newick_str, 0, 0.0)
    return root

def layout_tree(root):
    """
    Compute x,y coordinates for each node in a left-to-right cladogram.
    Returns a dict: node -> (x, y).

    Steps:
      1) Count how many leaves are in the tree.
      2) Do a DFS, assigning each leaf a unique y-coordinate (like 0,1,2...).
      3) Each internal node's y = average of children’s y.
      4) Each node’s x = parent's x + branch_length.

    We'll do a two-pass approach:
      - First pass: assign y to leaves (and propagate up).
      - Second pass: assign x by accumulating distances from the root.

    For simplicity, we treat the "root.length" as 0 from an imaginary parent.
    """
    # 1) Gather all leaves in a consistent order
    leaves = []
    def collect_leaves(node):
        if not node.children:
            leaves.append(node)
        else:
            for c in node.children:
                collect_leaves(c)
    collect_leaves(root)

    # We'll assign leaves y positions: 0..len(leaves)-1
    leaf_y = {}
    for i, lf in enumerate(leaves):
        leaf_y[lf] = i

    # 2) DFS to assign a "y_val" attribute to each node
    def assign_y(node):
        if not node.children:
            # It's a leaf
            node.y_val = leaf_y[node]
        else:
            for c in node.children:
                assign_y(c)
            # average children y
            sum_y = sum(child.y_val for child in node.children)
            node.y_val = sum_y / len(node.children)
    assign_y(root)

    # 3) Assign x positions with another DFS
    def assign_x(node, current_x=0.0):
        node.x_val = current_x
        for c in node.children:
            assign_x(c, current_x + c.length)
    assign_x(root, 0.0)

    # 4) Build dictionary of positions
    positions = {}
    def collect_positions(node):
        positions[node] = (node.x_val, node.y_val)
        for c in node.children:
            collect_positions(c)
    collect_positions(root)

    return positions

def plot_tree(root, outfile="tree.png"):
    """
    Plots a simple horizontal cladogram using matplotlib, saving to `outfile`.
    """
    positions = layout_tree(root)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(8, 4))
    # Collect all edges by DFS
    edges = []
    stack = [root]
    while stack:
        node = stack.pop()
        for c in node.children:
            edges.append((node, c))
            stack.append(c)

    # Draw edges
    for (parent, child) in edges:
        x1, y1 = positions[parent]
        x2, y2 = positions[child]
        ax.plot([x1, x2], [y1, y2], color="black")

    # Draw nodes (usually just for leaves)
    # We'll label all leaves with node.name if it's not empty
    for node, (x, y) in positions.items():
        if not node.children:
            # It's a leaf, draw a small circle and label
            ax.plot(x, y, "o", color="red")
            ax.text(x, y, " " + node.name, verticalalignment="center", fontsize=9)
        else:
            # internal node
            ax.plot(x, y, ".", color="black")

    ax.set_title("Phylogenetic Tree")
    ax.set_xlabel("Distance")
    ax.set_ylabel("")
    ax.yaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Saved tree figure to {outfile}")

###############################################################################
#                   Step 4: Main Function & Command-Line
###############################################################################

def gen_newick_tree(fasta_path: str, algo: str):
    """
    Reads a FASTA file, computes a distance matrix, and generates a Newick-format
    UPGMA or NJ tree. Saves the Newick to "ex3.{basename}.{ALGO}.tree",
    and also plots a simple cladogram to "ex3.{basename}.{ALGO}.png".
    """
    # Parse sequences
    sequences = parse_fasta_file(fasta_path)
    if len(sequences) == 0:
        print("No sequences found in the file. Exiting.")
        return

    # Build initial distance matrix
    distance_matrix, labels = build_distance_matrix(sequences)

    # Select algorithm
    if algo.lower() == "upgma":
        newick_str = upgma(distance_matrix, labels)
    elif algo.lower() == "nj":
        newick_str = neighbor_joining(distance_matrix, labels)
    else:
        raise ValueError("Unknown algorithm. Use 'nj' or 'upgma'.")

    # Print the result (optional)
    print(newick_str)

    # Construct output filenames
    base = os.path.basename(fasta_path)          # e.g. '50.fa'
    name_only = os.path.splitext(base)[0]        # e.g. '50'
    out_newick_file = f"ex3.{name_only}.{algo.upper()}.tree"
    out_plot_file   = f"ex3.{name_only}.{algo.upper()}.png"

    # Save Newick tree to file
    with open(out_newick_file, 'w') as outfile:
        outfile.write(newick_str)
    print(f"Tree written to {out_newick_file}")

    # -- Now parse the Newick string into our TreeNode structure
    root_node = parse_newick(newick_str)

    # -- Plot the tree
    plot_tree(root_node, out_plot_file)

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
