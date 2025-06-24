import os
from s3_utils import upload_image_to_s3

PENDING_DIR = 'pending_uploads'

def retry_pending_uploads():
    for filename in os.listdir(PENDING_DIR):
        file_path = os.path.join(PENDING_DIR, filename)
        if os.path.isfile(file_path):
            print(f"Retrying upload for: {filename}")
            link, pending = upload_image_to_s3(file_path, filename)
            if not pending:
                print(f"Uploaded: {filename} -> {link}")
                os.remove(file_path)
            else:
                print(f"Still pending: {filename}")

if __name__ == '__main__':
    retry_pending_uploads() 