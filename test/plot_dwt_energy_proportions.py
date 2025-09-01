import json
import argparse
import subprocess
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot DWT energy proportions by corruption type.')
    parser.add_argument('--plot_bands', nargs='+', default=['LL','LH','HL','HH'],
                        choices=['LL','LH','HL','HH'],
                        help='List of bands to plot (e.g., LL LH)')
    parser.add_argument('--plot_type', default='bar', choices=['bar', 'line'],
                        help='Plot type: bar or line')
    args = parser.parse_args()

    # Load data from JSON file
    json_path = os.path.join(os.path.dirname(__file__), 'dwt.json')
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract data for selected bands
    corruptions = []
    proportions = {band: [] for band in args.plot_bands}

    for entry in data:
        corruptions.append(entry['corruption'])
        for band in args.plot_bands:
            proportions[band].append(entry['proportions'][band])

    # Set up plot
    fig, ax = plt.subplots(figsize=(14, 8))
    index = np.arange(len(corruptions))
    colors = {'LL': '#1f77b4', 'LH': '#ff7f0e', 'HL': '#2ca02c', 'HH': '#d62728'}

    # Create plot based on selected type
    if args.plot_type == 'bar':
        bar_width = 0.8 / len(args.plot_bands)
        for i, band in enumerate(args.plot_bands):
            ax.bar(index + i * bar_width, proportions[band], bar_width, 
                   label=band, color=colors[band])
        ax.set_xticks(index + bar_width * (len(args.plot_bands)-1)/2)
    elif args.plot_type == 'line':
        for band in args.plot_bands:
            ax.plot(index, proportions[band], 'o-', label=band, color=colors[band])
        ax.set_xticks(index)

    # Configure plot
    ax.set_xlabel('Corruption Type', fontsize=12)
    ax.set_ylabel('Energy Proportion', fontsize=12)
    ax.set_title(f'DWT Energy Proportions: {", ".join(args.plot_bands)} ({args.plot_type} plot)', fontsize=14)
    ax.set_xticklabels(corruptions, rotation=45, ha='right', fontsize=10)
    ax.legend(title='DWT Bands', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), f'dwt_energy_proportions_{args.plot_type}.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    # Attempt to import matplotlib, install if missing
    try:
        import matplotlib
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.8.0", "numpy==1.26.0"])
    
    main()
