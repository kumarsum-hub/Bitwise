import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN for compatibility with Windows

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import Input
import glob
import argparse
import json
import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def load_wisdm_accel_data(watch_dir, phone_dir, code_to_idx):
    X_data = []
    for accel_dir in [watch_dir, phone_dir]:
        if not os.path.isabs(accel_dir):
            accel_dir = os.path.join(os.getcwd(), accel_dir)
        print(f"Processing directory: {accel_dir}")
        txt_files = glob.glob(os.path.join(os.getcwd(), accel_dir, '*.txt'))
        print(f"Found {len(txt_files)} text files.")
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                for line in f:
                    row = line.strip().split(',')
                    if len(row) < 6:
                        continue
                    _, code, _, x_str, y_str, z_str = row
                    z_str = z_str.rstrip(';')
                    # print(code, x_str, y_str, z_str)
                    if code.strip() not in code_to_idx:
                        continue
                    try:
                        x_val = float(x_str)
                        y_val = float(y_str)
                        z_val = float(z_str)
                    except ValueError:
                        print (f"Skipping row with invalid values: {line}")
                        continue
                    one_hot = np.zeros(len(code_to_idx), dtype=np.float32)
                    one_hot[code_to_idx[code.strip()]] = 1.0
                    X_data.append([x_val, y_val, z_val, *one_hot])
    print(f"Total valid rows found: {len(X_data)}")
    if not X_data:
        raise ValueError("No valid accelerometer data found in watch/phone accel directories.")
    return np.array(X_data, dtype=np.float32)

def build_generator(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # Force final output dimension to 3 for x, y, z
    model.add(Dense(3, activation='tanh'))
    return model

def train_generative_model(watch_dir, phone_dir, output_path, code_to_idx):
    X_data = load_wisdm_accel_data(watch_dir, phone_dir, code_to_idx)
    input_dim = X_data.shape[1]
    generator = build_generator(input_dim)
    generator.compile(optimizer='adam', loss=custom_mse)
    # Train predicting only the first three columns
    generator.fit(X_data, X_data[:, :3], epochs=5, batch_size=32)
    # Ensure the output path includes a valid file name with a .keras or .h5 extension
    if not output_path.endswith(('.keras', '.h5')):
        output_path = os.path.join(output_path, 'generator_model.keras')
    generator.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the generative model using config.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file.')
    args = parser.parse_args()

    with open(args.config, 'r') as cfg_file:
        config = json.load(cfg_file)

    # Build reverse mapping from WISDM code -> index
    # e.g. {'A': 0, 'B': 1, 'D': 2, 'C': 3} for walking, running, stationary, cycling
    code_to_idx = {}
    idx = 0
    for friendly_name, code in config['activity_code_map'].items():
        code_to_idx[code] = idx
        idx += 1

    watch_dir = config['training_watch_accel_dir']
    phone_dir = config['training_phone_accel_dir']
    generator_model_path = config['generator_model']
    train_generative_model(watch_dir, phone_dir, generator_model_path, code_to_idx)
