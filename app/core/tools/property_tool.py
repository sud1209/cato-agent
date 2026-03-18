import sqlite3
from pathlib import Path
from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "properties.db"

@tool
def fetch_user_property_profile(identifier: str) -> str:
    """
    Fetches the combined user and property profile. 
    Input 'identifier' can be a phone number (+1XXXXXXXXXX) or an address.
    Use this to get FICO scores, names, and home equity details in one step.
    """
    if not DB_PATH.exists():
        return f"Database error: File not found at {DB_PATH}"

    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        query = """
            SELECT 
                u.full_name, u.fico_score, u.has_liens,
                p.address, p.home_value, p.home_equity_pct
            FROM users u
            JOIN properties p ON u.property_id = p.property_id
            WHERE u.phone_number = ? OR p.address LIKE ?
        """
        
        cursor.execute(query, (identifier, f"%{identifier}%"))
        result = cursor.fetchone()
        conn.close()

        if result:
            name, fico, liens, addr, val, equity = result
            # Returns a comprehensive profile for the Master/Qualifier agents
            return (
                f"User: {name} | FICO: {fico} | Liens: {liens} | "
                f"Address: {addr} | Value: ${val:,.0f} | Equity: {equity*100}%"
            )
        
        return "Profile not found. Ask the user to provide their full address."
    
    except Exception as e:
        return f"System error: {str(e)}"