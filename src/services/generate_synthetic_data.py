import numpy as np
import pandas as pd
import os
import json
import argparse

def generate_accelerometer_data(activity, sample_duration, sample_count, acc_mg_range):
    data = []
    for _ in range(sample_count):
        # Generate time series data for the specified duration
        time_series = np.linspace(0, sample_duration / 1000, int(sample_duration / 1000 * 26))
        x = np.random.uniform(acc_mg_range['min'], acc_mg_range['max'], len(time_series))
        y = np.random.uniform(acc_mg_range['min'], acc_mg_range['max'], len(time_series))
        z = np.random.uniform(acc_mg_range['min'], acc_mg_range['max'], len(time_series))
        data.append(np.column_stack((time_series, x, y, z)))

    return data

def save_data_to_csv(data, activity, output_dir):
    activity_dir = os.path.join(output_dir, activity)
    if not os.path.exists(activity_dir):
        os.makedirs(activity_dir)
    
    for i, sample in enumerate(data):
        df = pd.DataFrame(sample, columns=['Time', 'X', 'Y', 'Z'])
        df.to_csv(os.path.join(activity_dir, f"sample_{i+1}.csv"), index=False)

def generate_synthetic_data(config, output_dir):
    activities = config['activities']
    sample_count_per_activity = config['sample_count_per_activity']
    sample_duration_in_ms = config['sample_duration_in_ms']
    acc_mg_range = config['acc_mg_range']

    for activity in activities:
        synthetic_data = generate_accelerometer_data(activity, sample_duration_in_ms, sample_count_per_activity, acc_mg_range)
        save_data_to_csv(synthetic_data, activity, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic accelerometer data.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file.')
    parser.add_argument('--out', type=str, required=True, help='Output directory for generated CSV files.')
    args = parser.parse_args()

    generate_synthetic_data(args.config, args.out)