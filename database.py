import sqlite3
from datetime import datetime

DB_NAME = "users.db"

def init_db():
    """Initialise la base de données avec la table des utilisateurs."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            balance_minutes REAL NOT NULL DEFAULT 10.0, -- Offert à l'inscription (ex: 10 min)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def create_user(email: str, api_key: str, initial_minutes: float = 10.0):
    """Crée un nouvel utilisateur avec un solde initial de minutes."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (email, api_key, balance_minutes) VALUES (?, ?, ?)",
            (email, api_key, initial_minutes)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user_by_api_key(api_key: str):
    """Récupère les informations d'un utilisateur par sa clé API."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, balance_minutes FROM users WHERE api_key = ?", (api_key,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return {"id": user[0], "email": user[1], "balance_minutes": user[2]}
    return None

def deduct_user_minutes(api_key: str, minutes_to_deduct: float):
    """Déduit les minutes consommées du solde de l'utilisateur."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET balance_minutes = balance_minutes - ? WHERE api_key = ?",
        (minutes_to_deduct, api_key)
    )
    conn.commit()
    conn.close()