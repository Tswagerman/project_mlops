import subprocess
import os
import pandas as pd 

def get_dvc_remote_path(remote_name):
    result = subprocess.run(['dvc', 'remote', 'default', remote_name], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise RuntimeError(f"Failed to get DVC remote path: {result.stderr.strip()}")

def get_data():
    # Run DVC pull to fetch data from the remote
    os.system('dvc pull -r public-remote')

    # Retrieve the local DVC cache path from the DVC configuration
    dvc_remote_path = get_dvc_remote_path('public-remote')

    # Load the CSV file into a Pandas DataFrame
    csv_file_path = os.path.join(dvc_remote_path, 'data/raw/news.csv')  # Adjust the path to your CSV file
    df = pd.read_csv(csv_file_path)

    print(df.head())

if __name__ == '__main__':
    get_data()
