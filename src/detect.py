import pandas as pd
import glob
import os

# Function to load tags
def load_tags(tags_file_path):
    tags_df = pd.read_csv(tags_file_path)
    tags_df['timestamp'] = pd.to_datetime(tags_df['timestamp'])
    return tags_df

# Function to load smartwatch data
def load_smartwatch_data(data_folder_path):
    all_files = glob.glob(os.path.join(data_folder_path, "*.csv"))
    data_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        data_list.append(df)
    data = pd.concat(data_list, ignore_index=True)
    return data

# Function to isolate data around tags
def isolate_data_around_tags(smartwatch_data, tags, time_window='5T'):
    isolated_data_list = []
    for tag in tags['timestamp']:
        start_time = tag - pd.Timedelta(time_window)
        end_time = tag + pd.Timedelta(time_window)
        isolated_data = smartwatch_data[(smartwatch_data['timestamp'] >= start_time) & (smartwatch_data['timestamp'] <= end_time)]
        isolated_data_list.append(isolated_data)
    return isolated_data_list

# Function to save isolated data
def save_isolated_data(isolated_data_list, output_folder_path):
    for i, data in enumerate(isolated_data_list):
        output_file_path = os.path.join(output_folder_path, f"isolated_data_{i+1}.csv")
        data.to_csv(output_file_path, index=False)

# Main function to process data
def process_data(tags_file_path, smartwatch_data_folder_path, output_folder_path, time_window='5T'):
    tags = load_tags(tags_file_path)
    smartwatch_data = load_smartwatch_data(smartwatch_data_folder_path)
    isolated_data_list = isolate_data_around_tags(smartwatch_data, tags, time_window)
    save_isolated_data(isolated_data_list, output_folder_path)
    print("Data processing complete!")

# Example usage
tags_file_path = 'tags.csv'
smartwatch_data_folder_path = 'smartwatch_data' 
output_folder_path = 'isolated_data'
time_window = '5T' 

os.makedirs(output_folder_path, exist_ok=True)
process_data(tags_file_path, smartwatch_data_folder_path, output_folder_path, time_window)
