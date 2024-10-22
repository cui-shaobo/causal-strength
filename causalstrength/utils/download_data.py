import requests
import os
from tqdm import tqdm
import hashlib

def download_ceq_data(data_dir='data'):
    """
    Downloads the causes.pkl and effects.pkl files required for the CEQ model.

    Parameters:
    - data_dir (str): The directory where data files will be stored.
    """
    # Google Drive file IDs for causes.pkl and effects.pkl
    causes_file_id = 'your_causes_file_id_here'   # Replace with your actual file ID
    effects_file_id = 'your_effects_file_id_here'  # Replace with your actual file ID

    # Expected SHA256 checksums for causes.pkl and effects.pkl
    causes_checksum = 'your_causes_checksum_here'   # Replace with the actual checksum
    effects_checksum = 'your_effects_checksum_here'  # Replace with the actual checksum

    os.makedirs(data_dir, exist_ok=True)

    causes_path = os.path.join(data_dir, 'causes.pkl')
    effects_path = os.path.join(data_dir, 'effects.pkl')

    print("Downloading causes.pkl...")
    _download_file_from_google_drive(causes_file_id, causes_path)

    # Verify the checksum of causes.pkl
    if _verify_file(causes_path, causes_checksum):
        print("causes.pkl downloaded and verified successfully.")
    else:
        print("Error: causes.pkl checksum verification failed.")
        os.remove(causes_path)
        raise ValueError("Downloaded causes.pkl file is corrupted. Please try downloading again.")

    print("Downloading effects.pkl...")
    _download_file_from_google_drive(effects_file_id, effects_path)

    # Verify the checksum of effects.pkl
    if _verify_file(effects_path, effects_checksum):
        print("effects.pkl downloaded and verified successfully.")
    else:
        print("Error: effects.pkl checksum verification failed.")
        os.remove(effects_path)
        raise ValueError("Downloaded effects.pkl file is corrupted. Please try downloading again.")

def _download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    # Initial request to get confirmation token
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def _save_response_content(response, destination):
    CHUNK_SIZE = 32768  # Adjust the chunk size if needed

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f'Downloading {os.path.basename(destination)}')

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # Filter out keep-alive new chunks
                f.write(chunk)
                progress_bar.update(len(chunk))
    progress_bar.close()

def _verify_file(file_path, expected_checksum):
    sha256_hash = hashlib.sha256()
    with open(file_path,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    calculated_checksum = sha256_hash.hexdigest()
    return calculated_checksum == expected_checksum
