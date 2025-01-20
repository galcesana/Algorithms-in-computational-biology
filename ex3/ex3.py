import argparse
from Bio import SeqIO
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from Bio.Phylo import write
from Bio.Phylo.Newick import Tree
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator

def parse_fasta_file(file_path: str):
    """
    Parses a FASTA file (plain or gzipped) and returns a mapping of sequence identifiers to nucleotide sequences.

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

def gen_newick_tree(fasta_path: str, algo: str):
    """
    Reads a FASTA file, computes a distance matrix, and generates a Newick-format UPGMA or NJ tree.

    Parameters:
        fasta_path (str): Path to the FASTA file.
        algo (str): Which algorithm will be used for tree construction. Either "nj" or "upgma".

    Prints:
        str: The Newick-format tree string. With the original leaf names, and without internal nodes names.
        for example: ((C:0.018,F:0.018):0.135,(D:0.125,(A:0.085,(B:0.072,E:0.072):0.0127):0.041):0.028);
    """
    # Parse sequences from the FASTA file
    sequences = parse_fasta_file(fasta_path)

    # Prepare a MultipleSeqAlignment object
    alignments = MultipleSeqAlignment([
        SeqRecord(Seq(seq), id=seq_id) for seq_id, seq in sequences.items()
    ])

    # Compute distance matrix
    calculator = DistanceCalculator('identity')
    distance_matrix = calculator.get_distance(alignments)

    # Construct tree using the selected algorithm
    constructor = DistanceTreeConstructor()
    if algo.lower() == 'upgma':
        tree = constructor.upgma(distance_matrix)
    elif algo.lower() == 'nj':
        tree = constructor.nj(distance_matrix)
    else:
        raise ValueError("Unsupported algorithm. Use 'nj' or 'upgma'.")

    # Remove internal node names and format branch lengths
    for clade in tree.find_clades():
        if not clade.is_terminal():
            clade.name = None

    # Generate Newick string
    newick_str = tree.format("newick")
    print(newick_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path", type=str, help="Path to FASTA file containing multiple DNA sequences.",
                        required=True)
    parser.add_argument("--algo", type=str, help="Either nj or upgma.",
                        required=True)

    args = parser.parse_args()

    gen_newick_tree(args.fasta_path, args.algo)
