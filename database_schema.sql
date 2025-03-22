-- PostgreSQL database schema for transaction classification system

-- Extension for vector operations (must be installed on the PostgreSQL server)
CREATE EXTENSION IF NOT EXISTS vector;

-- UNSPSC categories table for vector search
CREATE TABLE IF NOT EXISTS unspsc_categories (
    id SERIAL PRIMARY KEY,
    unspsc_id VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    level VARCHAR(10) NOT NULL,
    level_1_lookup INTEGER[] DEFAULT '{}',
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for vector similarity search
CREATE INDEX IF NOT EXISTS unspsc_categories_embedding_idx ON unspsc_categories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Function to create customer collection tables
CREATE OR REPLACE FUNCTION create_customer_collection(customer_id TEXT) 
RETURNS VOID AS $$
DECLARE
    table_name TEXT;
BEGIN
    table_name := 'customer_' || customer_id || '_collection';
    
    -- Check if table exists, create if not
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I (
            id SERIAL PRIMARY KEY,
            transaction_id VARCHAR(255) UNIQUE NOT NULL,
            embedding vector(1024) NOT NULL,
            unspsc_id VARCHAR(50) NOT NULL,
            cleansed_description TEXT,
            sourceable_flag BOOLEAN DEFAULT FALSE,
            category_description TEXT,
            transaction_type VARCHAR(100),
            llm_reasoning TEXT,
            single_word VARCHAR(100),
            level_1_categories INTEGER[] DEFAULT '{}',
            level VARCHAR(10),
            searchable BOOLEAN DEFAULT FALSE,
            rfp_description TEXT,
            matched_transaction_id VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )', table_name);
        
    -- Create vector index
    EXECUTE format('
        CREATE INDEX IF NOT EXISTS %I ON %I USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    ', table_name || '_embedding_idx', table_name);
END;
$$ LANGUAGE plpgsql;

-- Table for UNSPSC sectors
CREATE TABLE IF NOT EXISTS unspsc_sectors (
    id SERIAL PRIMARY KEY,
    unspsc_sector INTEGER NOT NULL,
    unspsc_sector_description TEXT NOT NULL,
    industry_filter INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster lookup of sectors by industry filter
CREATE INDEX IF NOT EXISTS unspsc_sectors_industry_filter_idx ON unspsc_sectors(industry_filter);