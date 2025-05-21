-- Create user if not exists
DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'huyvu') THEN
      CREATE USER huyvu WITH PASSWORD 'password';
      ALTER USER huyvu WITH SUPERUSER;
   END IF;
END
$$;

-- Create databases (ignore errors if they exist)
CREATE DATABASE raw_data OWNER huyvu;
CREATE DATABASE feature_db OWNER huyvu;
CREATE DATABASE postgre OWNER huyvu;
CREATE DATABASE mlflow OWNER huyvu;

