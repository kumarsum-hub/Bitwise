import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN for compatibility with Windows

import json
from services.generate_synthetic_data import generate_synthetic_data
from modeling.train import train_model
import argparse

def main(config_path, output_dir):
    # Load configuration
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Generate synthetic data
    os.makedirs(output_dir, exist_ok=True)
    generate_synthetic_data(config, output_dir)

    # Train the model
    model = train_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic accelerometer data and train model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file.')
    parser.add_argument('--out', type=str, required=True, help='Output directory for generated CSV files and model.')
    args = parser.parse_args()

    main(args.config, args.out)