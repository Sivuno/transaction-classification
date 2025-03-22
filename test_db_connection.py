#!/usr/bin/env python3
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database credentials
RDS_HOST = "sivuno-vector-database.cluster-csk0izuonaa1.us-east-1.rds.amazonaws.com"
RDS_PORT = 5432
RDS_USERNAME = os.getenv('RDS_USERNAME')
RDS_PASSWORD = os.getenv('RDS_PASSWORD')

async def test_customer_vectors_connection():
    print(f"Testing connection to customer_vectors database...")
    print(f"Host: {RDS_HOST}")
    print(f"Port: {RDS_PORT}")
    print(f"Username: {RDS_USERNAME}")
    print(f"Password: {'*' * len(RDS_PASSWORD) if RDS_PASSWORD else 'Not set'}")
    
    try:
        conn = await asyncpg.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            database="customer_vectors",
            user=RDS_USERNAME,
            password=RDS_PASSWORD
        )
        
        print("Connection successful!")
        
        # Test a simple query
        result = await conn.fetchval("SELECT 1")
        print(f"Test query result: {result}")
        
        # Close the connection
        await conn.close()
        print("Connection closed properly")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

async def test_connection_pool():
    print(f"\nTesting connection pool to customer_vectors database...")
    
    try:
        # Create pool
        pool = await asyncpg.create_pool(
            host=RDS_HOST,
            port=RDS_PORT,
            database="customer_vectors",
            user=RDS_USERNAME,
            password=RDS_PASSWORD,
            min_size=1,
            max_size=10
        )
        
        print("Pool creation successful!")
        
        # Test query with pool
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            print(f"Test query result from pool: {result}")
        
        # Close the pool
        await pool.close()
        print("Pool closed properly")
        return True
    except Exception as e:
        print(f"Pool creation/usage failed: {e}")
        return False

async def main():
    # Test direct connection
    conn_success = await test_customer_vectors_connection()
    
    # Test connection pool
    if conn_success:
        pool_success = await test_connection_pool()
    
    print("\nConnection Test Summary:")
    print(f"Direct connection: {'SUCCESS' if conn_success else 'FAILED'}")
    print(f"Connection pool: {'SUCCESS' if conn_success and pool_success else 'FAILED'}")
    
    if not conn_success:
        print("\nPossible issues:")
        print("1. Network connectivity to database host")
        print("2. Incorrect database credentials")
        print("3. Database server not running or unreachable")
        print("4. Firewall blocking the connection")
        print("5. Database 'customer_vectors' does not exist")

if __name__ == "__main__":
    asyncio.run(main())