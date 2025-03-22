# Transaction Classification System

A system for classifying and categorizing transactions using machine learning and semantic search.

## Overview

This application processes transaction data from CSV files stored in S3, classifies them using LLMs (OpenAI), and stores the results in a PostgreSQL database with vector search capabilities. The classification process uses a combination of:

1. Text embedding models (Jina AI) for vector representations
2. OpenAI's GPT models for classification and decision-making
3. Vector similarity search for finding matching categories
4. Database storage for persistence and reuse of previous classifications

## Project Structure

```
transaction_classification/
├── .env                           # Environment variables (credentials)
├── main.py                        # Main entry point
├── requirements.txt               # Python dependencies
├── src/                           # Source code
│   ├── config/                    # Configuration
│   │   └── config.py              # Configuration settings
│   ├── data/                      # Data handling
│   │   ├── s3_reader.py           # S3 file reading utilities
│   │   └── s3_storage.py          # S3 storage and CSV handling
│   ├── models/                    # ML model interfaces
│   │   └── classifier.py          # Classification logic
│   ├── services/                  # Service integrations
│   │   ├── api.py                 # External API integrations (OpenAI, Jina)
│   │   └── database.py            # Database connection and queries
│   └── utils/                     # Utilities
│       ├── aws.py                 # AWS utilities
│       ├── progress.py            # Progress tracking
│       └── rate_limiter.py        # API rate limiting
└── tests/                         # Unit and integration tests
```

## Setup and Installation

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Set up the environment variables in `.env` file:

```env
# API Keys
OPENAI_API_KEY=your_openai_key
JINA_API_KEY=your_jina_key

# AWS Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
REGION=us-east-1

# Database Credentials
RDS_USERNAME=your_db_username
RDS_PASSWORD=your_db_password
```

## Usage

To run the classification process:

```bash
python main.py
```

The application will:
1. Load CSV files from the S3 input location
2. Process transactions in batches
3. Classify each transaction using LLMs and vector search
4. Store results in both the database and output CSV files
5. Retry failed transactions
6. Verify all input transactions were processed

## Configuration

Key settings can be modified in `src/config/config.py`:

- `S3_INPUT_PREFIX`: S3 URI for input files
- `S3_OUTPUT_BUCKET`: S3 bucket for output files
- `S3_PROCESSING_PREFIX`: Prefix for temporary processing files
- `S3_OUTPUT_PREFIX`: Prefix for final output files
- `L3_MATCHES` and `L4_MATCHES`: Number of category matches to retrieve
- `MAX_CONCURRENT_EMBEDDINGS` and `MAX_CONCURRENT_CHAT_CALLS`: Concurrency settings
- `JINA_RATE_LIMIT_PER_MINUTE` and `OPENAI_RATE_LIMIT_PER_MINUTE`: API rate limits

## Error Handling and Resilience

The system includes several resilience features:
- Retry mechanisms for failed API calls
- Chunked processing of large files
- Transaction-level retry for failed classifications
- Verification of processed transactions
- Automatic reprocessing of missing transactions

## Architecture

This application uses a streaming processing architecture to handle large files efficiently:

1. Files are loaded from S3 in chunks to minimize memory usage
2. Concurrent processing utilizes asyncio for parallel operations
3. Rate limiters prevent API throttling
4. Results are written incrementally to avoid memory issues
5. Progress tracking provides visibility into processing status

## Development

To set up a development environment:

1. Clone the repository
2. Install development dependencies: `pip install -r requirements-dev.txt`
3. Configure pre-commit hooks: `pre-commit install`
4. Run tests: `pytest tests/`

### Database Setup

The system requires PostgreSQL with the vector extension installed. To set up the database schema:

```bash
psql -U your_username -d your_database -f database_schema.sql
```

The database schema includes:
- UNSPSC categories table with vector search capabilities
- Customer collection tables for storing transaction classifications
- UNSPSC sectors table for industry filtering

### Testing

Run the test suite with:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_s3_reader.py
```

## License

Proprietary - All rights reserved