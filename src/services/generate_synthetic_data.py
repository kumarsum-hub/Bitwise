import numpy as np
import pandas as pd
import os
import argparse
import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def generate_accelerometer_data(activity, sample_duration, sample_count, acc_mg_range, generator_model, name_to_code, code_to_idx):
    if not os.path.exists(generator_model):
        raise FileNotFoundError(f"{generator_model} not found. Train the generator model first.")

    generator = tf.keras.models.load_model(generator_model, custom_objects={'custom_mse': custom_mse})
    w_code = name_to_code[activity]  # e.g. "walking" -> "A"
    one_hot_dim = len(code_to_idx)
    one_hot_vec = np.zeros(one_hot_dim, dtype=np.float32)
    one_hot_vec[code_to_idx[w_code]] = 1.0
    one_hot_vec = one_hot_vec.reshape(1, -1)  # Ensure one_hot_vec is 2D

    data = []
    for _ in range(sample_count):
        random_latent = np.random.normal(0, 1, (1, 78))
        combined_input = np.hstack((random_latent, one_hot_vec))
        combined_input = combined_input.reshape(1, -1)  # Ensure combined_input is 2D
        synthetic_sample = generator.predict(combined_input)
        time_series = np.linspace(0, sample_duration / 1000, synthetic_sample.shape[0])
        x = synthetic_sample[:, 0]
        y = synthetic_sample[:, 1]
        z = synthetic_sample[:, 2]
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
    generator_model = config['generator_model']
    # Build forward mapping for activity name -> WISDM code
    name_to_code = config['activity_code_map']  # e.g. {"walking": "A", ...}
    # And reverse mapping code -> index
    code_to_idx = {}
    idx = 0
    for friendly_name, code in name_to_code.items():
        code_to_idx[code] = idx
        idx += 1

    for activity in activities:
        synthetic_data = generate_accelerometer_data(
            activity,
            config['sample_duration_in_ms'],
            config['sample_count_per_activity'],
            config['acc_mg_range'],
            generator_model,
            name_to_code,
            code_to_idx
        )
        save_data_to_csv(synthetic_data, activity, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic accelerometer data.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file.')
    parser.add_argument('--out', type=str, required=True, help='Output directory for generated CSV files.')
    args = parser.parse_args()

    generate_synthetic_data(args.config, args.out)