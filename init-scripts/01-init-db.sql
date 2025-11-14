-- Initialize the loan_default database
-- This script runs automatically when the PostgreSQL container starts for the first time

-- Create schema for raw data
CREATE SCHEMA IF NOT EXISTS raw;

-- Create schema for staging data
CREATE SCHEMA IF NOT EXISTS staging;

-- Create schema for intermediate data
CREATE SCHEMA IF NOT EXISTS intermediate;

-- Create schema for marts (final tables)
CREATE SCHEMA IF NOT EXISTS marts;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA raw TO loan_user;
GRANT ALL PRIVILEGES ON SCHEMA staging TO loan_user;
GRANT ALL PRIVILEGES ON SCHEMA intermediate TO loan_user;
GRANT ALL PRIVILEGES ON SCHEMA marts TO loan_user;

-- Set default search path
ALTER DATABASE loan_default SET search_path TO public, raw, staging, intermediate, marts;

-- Create extension for better performance with indexes
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create extension for UUID support (if needed in the future)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

COMMENT ON SCHEMA raw IS 'Raw data loaded from CSV files';
COMMENT ON SCHEMA staging IS 'Cleaned and standardized data';
COMMENT ON SCHEMA intermediate IS 'Feature engineering and derived calculations';
COMMENT ON SCHEMA marts IS 'Final ML-ready datasets';
