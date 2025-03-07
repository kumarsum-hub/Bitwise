import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import os
import glob
from tqdm import tqdm

class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.total_epochs = self.params['epochs']
        self.pbar = tqdm(total=self.total_epochs, desc="Training", unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)
        self.pbar.set_description_str(f"Epoch {epoch+1}/{self.total_epochs}")
        self.pbar.set_postfix({
            'loss': f"{logs['loss']:.4f}",
            'acc': f"{logs['accuracy']:.4f}",
            'val_loss': f"{logs['val_loss']:.4f}",
            'val_acc': f"{logs['val_accuracy']:.4f}"
        })
        if epoch == (self.total_epochs - 1):
            self.pbar.close()

def load_csv_data(output_dir):
    data = []
    labels = []
    # Read CSV files from each subdirectory
    activity_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) 
                     if os.path.isdir(os.path.join(output_dir, d))]
    for activity_path in activity_dirs:
        activity_name = os.path.basename(activity_path)
        csv_files = glob.glob(os.path.join(activity_path, '*.csv'))
        for file in csv_files:
            df = pd.read_csv(file)
            # Take only x, y, z columns
            arr = df[['X','Y','Z']].values
            data.append(arr)
            labels.append(activity_name)
    return np.array(data), np.array(labels)

def preprocess_data(data):
    # Flatten the data for LSTM input
    data = np.array([sample.flatten() for sample in data])
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
    return data, scaler

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(output_dir):
    data, labels = load_csv_data(output_dir)
    # Convert string labels to integer labels
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    data, scaler = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    num_classes = len(np.unique(labels))
    model = build_model((X_train.shape[1], 1), num_classes)
    
    # Add TQDMProgressBar callback
    tqdm_callback = TQDMProgressBar()
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0,
        callbacks=[tqdm_callback]
    )
    
    model.save(os.path.join(output_dir, 'synthetic_data_generator_model.keras'))  # Save the trained model
    
    return model, scaler

if __name__ == "__main__":
    output_dir = 'data/output'  # Path to the output directory
    model, scaler = train_model(output_dir)