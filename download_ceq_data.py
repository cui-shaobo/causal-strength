"""
@Project  : causal-strength
@File     : download_ceq_data.py
@Author   : Shaobo Cui
@Date     : 23.10.2024 16:18
"""

import os
import zipfile
import hashlib
import gdown

def download_file_from_google_drive(file_id, destination):
    """
    Downloads a file from Google Drive using gdown.

    Parameters:
    - file_id (str): The Google Drive file ID.
    - destination (str): The local path where the file will be saved.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

def unzip_file(zip_path, extract_to):
    """
    Unzips a ZIP file to the specified directory.

    Parameters:
    - zip_path (str): The path to the ZIP file.
    - extract_to (str): The directory where the contents will be extracted.
    """
    print(f'Unzipping {os.path.basename(zip_path)}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f'Unzipped to {extract_to}')
    # Optionally, remove the zip file after extraction
    os.remove(zip_path)
    print(f'Removed {os.path.basename(zip_path)}')

def verify_checksum(file_path, expected_checksum):
    """
    Verifies the SHA256 checksum of a file.

    Parameters:
    - file_path (str): Path to the file to verify.
    - expected_checksum (str): The expected SHA256 checksum.

    Returns:
    - bool: True if the checksum matches, False otherwise.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path,"rb") as f:
        for byte_block in iter(lambda: f.read(8192),b""):
            sha256_hash.update(byte_block)
    calculated_checksum = sha256_hash.hexdigest()
    return calculated_checksum == expected_checksum

if __name__ == "__main__":
    # Google Drive file IDs for causes.pkl.zip and effects.pkl.zip

    causes_file_id = '1TTYk5fXE3dcbobWIwl93DFtiYYRf7TYy'
    effects_file_id = '1GC9bvpKe9FPP5LCT4yj1Z6YK9S0Ne0GD'

    # Expected SHA256 checksums for the ZIP files
    causes_checksum = '124116ee65624f230d62eb371a2a652b5053b9614bafebcd5742bb2527cc7df8'
    effects_checksum = '1af3660ce51a18b1ffc8b34105fc4d0850b6b3fbf7769ce0d8028c5d7133f6af'


    data_dir = os.path.join(os.path.dirname(__file__), '', 'data')
    os.makedirs(data_dir, exist_ok=True)

    causes_zip_path = os.path.join(data_dir, 'causes.pkl.zip')
    effects_zip_path = os.path.join(data_dir, 'effects.pkl.zip')

    # Download the zipped causes.pkl
    print("Downloading causes.pkl.zip...")
    download_file_from_google_drive(causes_file_id, causes_zip_path)

    # Unzip causes.pkl.zip
    unzip_file(causes_zip_path, data_dir)

    # Verify checksum for causes.pkl
    causes_pkl_path = os.path.join(data_dir, 'causes.pkl')
    print("Verifying causes.pkl...")
    if verify_checksum(causes_pkl_path, causes_checksum):
        print("Checksum verification passed for causes.pkl.")
    else:
        print("Checksum verification failed for causes.pkl.")
        print("The file may be corrupted. Please try downloading again.")
        os.remove(causes_pkl_path)
        exit(1)

    # Download the zipped effects.pkl
    print("Downloading effects.pkl.zip...")
    download_file_from_google_drive(effects_file_id, effects_zip_path)

    # Unzip effects.pkl.zip
    unzip_file(effects_zip_path, data_dir)

    # Verify checksum for effects.pkl
    effects_pkl_path = os.path.join(data_dir, 'effects.pkl')
    print("Verifying effects.pkl...")
    if verify_checksum(effects_pkl_path, effects_checksum):
        print("Checksum verification passed for effects.pkl.")
    else:
        print("Checksum verification failed for effects.pkl.")
        print("The file may be corrupted. Please try downloading again.")
        os.remove(effects_pkl_path)
        exit(1)


