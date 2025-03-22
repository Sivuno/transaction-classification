import os
import logging
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

# API Keys and Credentials
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
JINA_API_KEY = os.getenv('JINA_API_KEY')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
REGION = os.getenv('REGION')
RDS_USERNAME = os.getenv('RDS_USERNAME')
RDS_PASSWORD = os.getenv('RDS_PASSWORD')

# ML Model Settings
EMBEDDING_MODEL = "jina-embeddings-v3"
EMBEDDING_DIMENSION = 1024
DEFAULT_COLLECTION = "unspsc_categories"
GPT_MODEL_INITIAL = "gpt-4o-mini"
GPT_MODEL_BEST_MATCH = "gpt-4o-mini"
GPT_MODEL_LEVEL1 = "gpt-4o-mini"

# Rate Limits
JINA_RATE_LIMIT_PER_MINUTE = 1500
OPENAI_RATE_LIMIT_PER_MINUTE = 10000
MAX_CONCURRENT_EMBEDDINGS = 15
MAX_CONCURRENT_CHAT_CALLS = 100

# Database Settings
RDS_HOST = "sivuno-vector-database.cluster-csk0izuonaa1.us-east-1.rds.amazonaws.com"
RDS_PORT = 5432
RDS_MASTER_TABLES_CLUSTER = "sivuno-master-tables-instance-1.csk0izuonaa1.us-east-1.rds.amazonaws.com"
RDS_MASTER_TABLES_DATABASE = "master-tables"

# Search and Match Settings
L3_MATCHES = 25
L4_MATCHES = 25

# S3 Settings
S3_INPUT_PREFIX = "s3://sivuno-transaction-processing/input/"
S3_OUTPUT_BUCKET = "sivuno-transaction-processing"
S3_PROCESSING_PREFIX = "processing/"
S3_OUTPUT_PREFIX = "output/"

# DynamoDB Settings
DYNAMODB_TRANSACTIONS_TABLE = "sivuno_transactions"
DYNAMODB_RESULTS_TABLE = "sivuno_classification_results"
USE_DYNAMODB = True  # Set to True to use DynamoDB instead of S3

# Other Constants
DEFAULT_CHUNK_SIZE = 500

def validate_credentials() -> Optional[str]:
    """Validates that required credentials are present"""
    missing_keys = []
    
    if not OPENAI_API_KEY:
        missing_keys.append('OPENAI_API_KEY')
    if not JINA_API_KEY:
        missing_keys.append('JINA_API_KEY')
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not REGION:
        missing_keys.append('AWS credentials')
    if not RDS_USERNAME or not RDS_PASSWORD:
        missing_keys.append('RDS credentials')
    
    if missing_keys:
        return f"Missing required credentials: {', '.join(missing_keys)}"
    
    return None

def setup_logging(level=logging.INFO):
    """Configure logging for the application"""
    log_filename = "transaction_matching.log"
    
    # Configure root logger with the specified level
    logging.getLogger().setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # File handler - logs everything at the specified level
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler - logs only errors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(console_formatter)
    
    # Get root logger and remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)