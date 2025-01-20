import argparse
from Bio import Phylo
from io import StringIO

def compare_newick_trees(tree1_path: str, tree2_str: str):
    """
    Compares a Newick tree from a file path to another Newick tree provided as a string.

    Parameters:
        tree1_path (str): Path to the first Newick tree file.
        tree2_str (str): String representation of the second Newick tree.

    Returns:
        None: Prints differences in structure and distance.
    """
    # Load the first tree from the file
    tree1 = Phylo.read(tree1_path, "newick")

    # Load the second tree from the string
    tree2 = Phylo.read(StringIO(tree2_str), "newick")

    # Compare tree structure
    print("Tree 1:")
    Phylo.draw_ascii(tree1)
    print("\nTree 2:")
    Phylo.draw_ascii(tree2)

    # Compare branch lengths
    tree1_distances = tree1.distance(tree1.root)
    tree2_distances = tree2.distance(tree2.root)

    if tree1_distances == tree2_distances:
        print("\nThe trees have identical branch lengths.")
    else:
        print("\nThe trees have different branch lengths.")
        print("Tree 1 distances:", tree1_distances)
        print("Tree 2 distances:", tree2_distances)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_1", type=str, required=True, help="Path to the first Newick tree file.")
    parser.add_argument("--tree_2", type=str, required=True, help="Newick format string for the second tree.")

    args = parser.parse_args()

    compare_newick_trees(args.tree_1, args.tree_2)
