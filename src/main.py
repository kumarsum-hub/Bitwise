import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN for compatibility with Windows

import json
import argparse
from services.generate_synthetic_data import generate_synthetic_data

def main(config_path, output_dir):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    os.makedirs(output_dir, exist_ok=True)
    # Generate synthetic data
    generate_synthetic_data(config, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic accelerometer data.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file.')
    parser.add_argument('--out', type=str, required=True, help='Output directory for generated CSV files.')
    args = parser.parse_args()

    main(args.config, args.out)