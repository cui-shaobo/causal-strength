import hashlib

def compute_checksum(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    calculated_checksum = sha256_hash.hexdigest()
    return calculated_checksum

# Example usage:
causes_checksum = compute_checksum('path/to/causes.pkl')
effects_checksum = compute_checksum('path/to/effects.pkl')

print('causes.pkl checksum:', causes_checksum)
print('effects.pkl checksum:', effects_checksum)