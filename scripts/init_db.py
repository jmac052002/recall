# scripts/init_db.py
# Run once to initialize the PostgreSQL schema.
# Safe to re-run - all statements use IF NOT EXISTS.

import psycopg2
from pathlib import Path
from recall.config import settings 


def run() -> None:
    schema_path = Path(__file__).parent.parent / "recall" / "memory" / "schema.sql"
    sql = schema_path.read_text()
    
    print(f"Connecting to {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}...")
    
    conn = psycopg2.connect(settings.postgres_dsn)
    conn.autocommit = True # DDL statements dont't need explicit transactions
    
    try:
        with conn.cursor() as cur:
             cur.execute(sql)
        print("PostgreSQL schema initialized successfully.")
    finally:
        conn.close()
        
        
if __name__ == "__main__":
    run()