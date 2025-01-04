import os
import pandas as pd

def convert_feather_to_csv(directory: str):
    """
    Converts all `.feather` files in the given directory to `.csv` files.

    Args:
        directory (str): The path to the directory containing `.feather` files.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Iterate through all files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.feather'):
            feather_path = os.path.join(directory, file_name)
            csv_path = os.path.join(directory, file_name.replace('.feather', '.csv'))

            # Read the Feather file and write it as a CSV
            try:
                print(f"Converting {feather_path} to {csv_path}")
                df = pd.read_feather(feather_path)
                df.to_csv(csv_path, index=False)
                print(f"Successfully converted {file_name} to {csv_path}")
            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")

# Set the directory containing the Feather files
directory = r"/freqtrade/user_data/data/binance"

# Convert all Feather files in the directory to CSV
convert_feather_to_csv(directory)
