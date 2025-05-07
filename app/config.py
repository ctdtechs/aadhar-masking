import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SCHEDULER_INTERVAL_SECONDS = int(os.getenv("SCHEDULER_INTERVAL_SECONDS", 300))
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", 3306))
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")