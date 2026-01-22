"""
SQLite database setup and connection management for Position Manager
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional
from contextlib import contextmanager


# Database path
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DB_PATH = os.path.join(DB_DIR, 'positions.db')


def get_db_path() -> str:
    """Get the database file path"""
    return DB_PATH


def init_database():
    """Initialize the SQLite database with required tables"""
    # Create data directory if it doesn't exist
    os.makedirs(DB_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create positions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            entry_price REAL NOT NULL,
            quantity REAL NOT NULL DEFAULT 1.0,
            entry_date TEXT NOT NULL,
            current_price REAL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            exit_price REAL,
            exit_date TEXT,
            source TEXT NOT NULL DEFAULT 'MANUAL',
            signal_id TEXT,
            pnl REAL,
            pnl_percentage REAL,
            result TEXT DEFAULT 'HOLD',
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')

    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_positions_result ON positions(result)
    ''')

    conn.commit()
    conn.close()

    return DB_PATH


class PositionDB:
    """Database connection manager for positions"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DB_PATH
        # Initialize database on first use
        if not os.path.exists(self.db_path):
            init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def execute(self, query: str, params: tuple = ()):
        """Execute a query and return the cursor"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor

    def fetch_one(self, query: str, params: tuple = ()):
        """Execute a query and fetch one result"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()

    def fetch_all(self, query: str, params: tuple = ()):
        """Execute a query and fetch all results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def insert(self, table: str, data: dict) -> int:
        """Insert a record and return the last row id"""
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(data.values()))
            conn.commit()
            return cursor.lastrowid

    def update(self, table: str, data: dict, where: str, where_params: tuple) -> int:
        """Update records and return the number of affected rows"""
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(data.values()) + where_params)
            conn.commit()
            return cursor.rowcount

    def delete(self, table: str, where: str, where_params: tuple) -> int:
        """Delete records and return the number of affected rows"""
        query = f"DELETE FROM {table} WHERE {where}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, where_params)
            conn.commit()
            return cursor.rowcount
