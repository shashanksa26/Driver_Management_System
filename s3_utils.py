import os
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError

AWS_BUCKET = os.environ.get('AWS_S3_BUCKET')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
PENDING_DIR = 'pending_uploads'

if not os.path.exists(PENDING_DIR):
    os.makedirs(PENDING_DIR)

def upload_image_to_s3(image_path, filename):
    """
    Uploads an image to AWS S3. Returns (link, pending:bool).
    If upload fails, saves to pending_uploads and returns (local_path, True).
    """
    s3 = boto3.client('s3', region_name=AWS_REGION)
    try:
        s3.upload_file(image_path, AWS_BUCKET, filename, ExtraArgs={'ACL': 'public-read', 'ContentType': 'image/jpeg'})
        link = f'https://{AWS_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{filename}'
        return link, False
    except (BotoCoreError, NoCredentialsError, ClientError) as e:
        fallback_path = os.path.join(PENDING_DIR, filename)
        if not os.path.exists(fallback_path):
            os.rename(image_path, fallback_path)
        print(f"S3 upload failed, saved to pending: {fallback_path}. Error: {e}")
        return fallback_path, True 