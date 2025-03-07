# Generative AI Accelerometer Data Generation Project

## Overview
This project aims to create a generative AI model and helper scripts for generating synthetic 3D MEMS accelerometer data for Human Activities Recognition. The data is generated based on various human activities and is intended to be used for training and evaluating machine learning models.

## Features
- Generates synthetic accelerometer data for multiple activities.
- Supports configurable parameters through a JSON input file.
- Outputs data in CSV format, organized by activity.
- Includes a machine learning model for classifying the generated data.

## Activities Covered
The following activities are included in the synthetic data generation:
- Walking
- Running
- Stationary
- Cycling
- Nordic Walking
- Ascending Stairs
- Descending Stairs
- Ironing
- House Cleaning
- Playing Soccer
- Rope Jumping

## Project Structure
```
generative-ai-accelerometer
├── src
│   ├── modeling
│   │   └── train.py              # Implementation of the generative AI model
│   ├── services
│   │   └── generate_synthetic_data.py  # Logic for generating synthetic data
│   └── main.py                   # Entry point for the application
├── config
│   └── configuration.json         # Configuration settings for the project
├── output
│   └── README.md                 # Documentation for the output directory
├── requirements.txt              # Project dependencies
└── README.md                     # Overall documentation for the project
```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd generative-ai-accelerometer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the input settings in `config/configuration.json` as needed.

## Usage
To generate synthetic accelerometer data and train the model, run the following command:
```bash
python src/main.py --config config/configuration.json --out output_dir_name
```

## Acceptance Criteria
The generated synthetic data should achieve at least 99.5% classification accuracy when evaluated using a discriminative AI model.

## Artifacts
- The trained machine learning model will be saved in Keras compatible format (.keras, .h5).
- Generated synthetic samples will be saved in CSV files organized by activity in the specified output directory.

For further details on the output structure, refer to the `output/README.md` file.