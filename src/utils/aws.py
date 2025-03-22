import boto3
from urllib.parse import urlparse
import logging
from ..config.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, REGION

logger = logging.getLogger(__name__)

def create_s3_client():
    """Create and return an S3 client with the configured credentials."""
    logger.debug("Creating S3 client with static credentials.")
    
    # Validate credentials
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not REGION:
        logger.error("Missing AWS credentials or REGION environment variables.")
        raise RuntimeError("AWS credentials (key/secret) or region not set.")
    
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=REGION
    )

def create_dynamodb_client():
    """Create and return a DynamoDB client with the configured credentials."""
    logger.debug("Creating DynamoDB client with static credentials.")
    
    # Validate credentials
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not REGION:
        logger.error("Missing AWS credentials or REGION environment variables.")
        raise RuntimeError("AWS credentials (key/secret) or region not set.")
    
    return boto3.client(
        'dynamodb',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=REGION
    )

def create_dynamodb_resource():
    """Create and return a DynamoDB resource with the configured credentials."""
    logger.debug("Creating DynamoDB resource with static credentials.")
    
    # Validate credentials
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not REGION:
        logger.error("Missing AWS credentials or REGION environment variables.")
        raise RuntimeError("AWS credentials (key/secret) or region not set.")
    
    return boto3.resource(
        'dynamodb',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=REGION
    )

def parse_s3_uri(s3_uri):
    """
    Parse an S3 URI into bucket and key components.
    
    Args:
        s3_uri (str): URI in format s3://bucket/key
        
    Returns:
        tuple: (bucket, key)
    """
    logger.debug(f"Parsing S3 URI: {s3_uri}")
    parsed = urlparse(s3_uri, allow_fragments=False)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key

def list_csv_files_in_s3_folder(s3_uri):
    """
    List all CSV files in an S3 folder.
    
    Args:
        s3_uri (str): URI in format s3://bucket/path/
        
    Returns:
        list: List of S3 URIs for CSV files
    """
    logger.debug(f"Listing CSV files in S3 folder: {s3_uri}")
    s3_client = create_s3_client()
    bucket, prefix = parse_s3_uri(s3_uri)
    if not prefix.endswith('/'):
        prefix += '/'
    try:
        csv_files = []
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.lower().endswith('.csv'):
                    csv_files.append(f"s3://{bucket}/{key}")
        logger.debug(f"Found {len(csv_files)} CSV files in {s3_uri}")
        return csv_files
    except Exception as e:
        logger.error(f"Failed to list CSV files in folder {s3_uri}: {e}")
        return []

def upload_to_s3(local_path, s3_uri):
    """
    Upload a local file to S3 and verify the upload.
    
    Args:
        local_path (str): Path to the local file
        s3_uri (str): Destination S3 URI
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.debug(f"Uploading {local_path} to {s3_uri}")
    import os
    try:
        s3_client = create_s3_client()
        bucket, key = parse_s3_uri(s3_uri)
        logger.debug("Uploading file to S3...")
        s3_client.upload_file(local_path, bucket, key)

        # Verify the upload
        try:
            logger.debug("Verifying S3 upload.")
            s3_client.head_object(Bucket=bucket, Key=key)
            logger.info(f"Successfully verified upload of {local_path} to {s3_uri}")
            os.remove(local_path)
            logger.info(f"Deleted local file {local_path} after successful S3 upload")
            return True
        except Exception as e:
            logger.error(f"Failed to verify S3 upload of {local_path}: {e}")
            return False

    except Exception as e:
        logger.error(f"Failed to upload {local_path} to S3: {e}")
        return False