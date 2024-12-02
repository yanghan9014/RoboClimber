import os
import shutil
from datetime import datetime

# Define the log directory and the cutoff timestamp
log_dir = "./"  # Replace with your actual log directory
cutoff_time = datetime.strptime("29-11-2024_17-15-04", "%d-%m-%Y_%H-%M-%S")

# Iterate over all subdirectories
for subdir in os.listdir(log_dir):
    subdir_path = os.path.join(log_dir, subdir)
    
    # Ensure it's a directory
    if os.path.isdir(subdir_path):
        try:
            # Extract the timestamp from the directory name
            parts = subdir.split("_")[-2:]  # Last two parts are usually the date and time
            timestamp_str = "_".join(parts)
            timestamp = datetime.strptime(timestamp_str, "%d-%m-%Y_%H-%M-%S")
            
            # Compare the timestamp
            if timestamp < cutoff_time:
                print(f"Deleting: {subdir_path}")
                shutil.rmtree(subdir_path)  # Delete the directory
        except Exception as e:
            print(f"Skipping {subdir}: {e}")  # Handle directories without proper timestamps

