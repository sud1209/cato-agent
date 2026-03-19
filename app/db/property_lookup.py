from __future__ import annotations
import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "properties.db"


@dataclass
class PropertyRecord:
    full_name: str
    fico_score: int
    has_liens: bool
    address: str
    home_value: float
    equity_pct: float

    @property
    def mortgage_balance(self) -> float:
        """Derive mortgage balance from equity percentage."""
        return self.home_value * (1 - self.equity_pct)


def _query_sync(name: str | None, address: str | None) -> PropertyRecord | None:
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cursor = conn.cursor()
        conditions, params = [], []
        if name:
            conditions.append("u.full_name LIKE ?")
            params.append(f"%{name}%")
        if address:
            conditions.append("p.address LIKE ?")
            params.append(f"%{address}%")
        if not conditions:
            return None
        sql = f"""
            SELECT u.full_name, u.fico_score, u.has_liens,
                   p.address, p.home_value, p.home_equity_pct
            FROM users u
            JOIN properties p ON u.property_id = p.property_id
            WHERE {" OR ".join(conditions)}
            LIMIT 1
        """
        cursor.execute(sql, params)
        row = cursor.fetchone()
        if not row:
            return None
        name_db, fico, liens, addr, val, equity = row
        return PropertyRecord(
            full_name=name_db,
            fico_score=int(fico),
            has_liens=bool(liens),
            address=addr,
            home_value=float(val),
            equity_pct=float(equity),
        )
    finally:
        conn.close()


async def lookup_property(name: str | None, address: str | None) -> PropertyRecord | None:
    """Async property lookup by name and/or address. Runs SQLite in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _query_sync, name, address)
