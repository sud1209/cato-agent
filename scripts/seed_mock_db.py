import sqlite3
import pandas as pd
from pathlib import Path

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "properties.db"
USER_CSV = PROJECT_ROOT / "data" / "users.csv"
PROP_CSV = PROJECT_ROOT / "data" / "properties.csv"

def seed_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    
    try:
        df_props = pd.read_csv(PROP_CSV)
        df_users = pd.read_csv(USER_CSV)
        
        df_props.to_sql("properties", conn, if_exists="replace", index=False)
        df_users.to_sql("users", conn, if_exists="replace", index=False)
        
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX idx_phone ON users(phone_number)")
        cursor.execute("CREATE INDEX idx_prop_id ON properties(property_id)")
        
        conn.commit()
        print(f"Successfully seeded {len(df_users)} users and {len(df_props)} properties.")
        
    except Exception as e:
        print(f"Error during seeding: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    seed_database()