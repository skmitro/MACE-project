import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF

def parse_indices(arg_list, total_atoms):
    """
    Parses a list of command-line arguments into a list of atom indices.
    Handles explicit lists, ranges (e.g., '1-100'), and 'last n' (e.g., '-500').
    """
    indices = []
    # Case 1: Last N atoms (e.g., ['-500'])
    if len(arg_list) == 1 and arg_list[0].startswith('-') and arg_list[0][1:].isdigit():
        num_last = int(arg_list[0][1:])
        if num_last > total_atoms or num_last <= 0:
            raise ValueError(f"Cannot select last {num_last} atoms from a system with {total_atoms} atoms.")
        start_index = total_atoms - num_last + 1
        indices = list(range(start_index, total_atoms + 1))
    # Case 2: Range (e.g., ['1-100'])
    elif len(arg_list) == 1 and '-' in arg_list[0]:
        parts = arg_list[0].split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            start, end = int(parts[0]), int(parts[1])
            if start > end or start < 1 or end > total_atoms:
                raise ValueError(f"Invalid range '{arg_list[0]}'. Check start/end values.")
            indices = list(range(start, end + 1))
        else:
            raise ValueError(f"Invalid range format: {arg_list[0]}. Use 'start-end'.")
    # Case 3: Explicit list (e.g., ['1', '5', '10'])
    else:
        indices = [int(i) for i in arg_list]

    return indices

def analyze_and_plot(args):
    """
    Main function to perform RDF analysis and generate outputs.
    """
    # --- 1. Load Data ---
    rmin, rmax, nbins = args.min_max_bins[0], args.min_max_bins[1], int(args.min_max_bins[2])

    # Load Universe: either a single structure or a topology + trajectory
    if args.trajectory:
        print(f"Loading topology from {args.topology} and trajectory from {args.trajectory}...")
        u = mda.Universe(args.topology, args.trajectory)
    else:
        print(f"Loading static structure from {args.topology}...")
        u = mda.Universe(args.topology)
    
    total_atoms = len(u.atoms)

    # --- 2. Define Atom Groups based on Arguments ---
    try:
        # Define Group 1
        group1_indices = parse_indices(args.group1, total_atoms)
        group1 = u.atoms[np.array(group1_indices) - 1]
        print(f"Group 1 defined with {len(group1)} atoms.")

        # Define Group 2
        group2_args = args.group2
        # ... (rest of group2 logic remains the same)
        is_indices = False
        try:
            group2_indices = parse_indices(group2_args, total_atoms)
            is_indices = True
        except ValueError:
            pass

        if is_indices:
            group2 = u.atoms[np.array(group2_indices) - 1]
            print(f"Group 2 defined with {len(group2)} atoms from provided indices/range.")
        elif len(group2_args) == 1 and group2_args[0].lower() == 'all':
            group2 = u.select_atoms("not group group1", group1=group1)
            print(f"Group 2 defined as all atoms not in Group 1 ({len(group2)} atoms).")
        elif len(group2_args) == 1:
            element = group2_args[0]
            group2 = u.select_atoms(f"name {element} and not group group1", group1=group1)
            print(f"Group 2 defined as element '{element}' not in Group 1 ({len(group2)} atoms).")
        else:
            raise ValueError("Invalid format for --group2.")

    except (ValueError, IndexError) as e:
        print(f"Error parsing atom groups: {e}")
        return

    if len(group1) == 0 or len(group2) == 0:
        print("Error: One or both atom groups are empty. Please check your selections.")
        return

    # --- 3. MDAnalysis RDF Calculation ---
    print(f"Calculating rPDF over {len(u.trajectory)} frame(s)...")
    rdf = InterRDF(group1, group2, nbins=nbins, range=(rmin, rmax))
    rdf.run()

    # --- 4. Output Generation ---
    base_name = os.path.splitext(os.path.basename(args.topology))[0]
    # ... (rest of output generation and plotting remains the same)
    rdf_txt_file = f"{base_name}_rPDF.txt"
    bins_txt_file = f"{base_name}_rPDF-bins.txt"
    png_file = f"{base_name}_rPDF.png"

    with open(rdf_txt_file, "w") as f:
        f.write("# r (Angstrom)\t g(r)\n")
        np.savetxt(f, np.c_[rdf.results.bins, rdf.results.rdf], fmt=['%.4f', '%.6f'], delimiter='\t')
    print(f"Saved rPDF data to: {rdf_txt_file}")

    with open(bins_txt_file, "w") as f:
        f.write("# r (Angstrom)\t count\n")
        np.savetxt(f, np.c_[rdf.results.bins, rdf.results.count], fmt=['%.4f', '%d'], delimiter='\t')
    print(f"Saved bin count data to: {bins_txt_file}")

    # --- 5. Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(rdf.results.bins, rdf.results.rdf, lw=2)
    ax.set_xlabel(r"r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_title(f"Radial Pair Distribution Function for {base_name}")
    ax.axhline(1, color='grey', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(png_file, dpi=300)
    print(f"Saved plot to: {png_file}")
    plt.close(fig)

if __name__ == '__main__':
    # --- Matplotlib Style Setup ---
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    plt.rc('font', size=SMALL_SIZE)
    # ... (rest of rc params)

    # --- Argparse Setup ---
    parser = argparse.ArgumentParser(
        description="Welcome to the plotting script for radial pair distribution functions based on MDAnalysis!\nIf you use this script to generate figures for publication, be sure to cite MDAnalysis (doi: 10.1002/jcc.21787 / 10.25080/Majora-629e541a-00e).\nCalculate and plot a radial pair distribution function (rPDF) from a structure or trajectory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--topology', required=True, help='Path to the topology file (e.g., PDB, GRO, TPR).')
    parser.add_argument('--trajectory', help='(Optional) Path to the trajectory file (e.g., XTC, DCD, TRR).')
    parser.add_argument('--min_max_bins', required=True, nargs=3, type=float, 
                        metavar=('R_MIN', 'R_MAX', 'N_BINS'),
                        help='Lower limit, upper limit, and number of bins for the RDF.')
    parser.add_argument('--group1', required=True, nargs='+',
                        help="Define Group 1. Use a space-separated list of 1-based indices, a range 'start-end', or '-N' for the last N atoms.")
    parser.add_argument('--group2', required=True, nargs='+',
                        help="Define Group 2. Accepts the same formats as --group1, OR a single element symbol (e.g., 'O'), OR the keyword 'all'.")

    args = parser.parse_args()
    analyze_and_plot(args)
