import json

def parse_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def get_activities(config):
    return config.get("activities", [])

def get_sample_count(config):
    return config.get("sample_count_per_activity", 0)

def get_sample_duration(config):
    return config.get("sample_duration_in_ms", 1000)

def get_acceleration_range(config):
    return config.get("acc_mg_range", {"min": -4000, "max": 4000})