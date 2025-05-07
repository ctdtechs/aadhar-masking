import logging
from .db import get_db_connection

logging.basicConfig(level=logging.INFO)

def scheduled_task():
    logging.info("Running scheduled task...")

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM file_details WHERE LOWER(process_status) = 'not processed'"
        cursor.execute(query)
        results = cursor.fetchall()
        logging.info(f"Found {len(results)} unprocessed records.")
    except Exception as e:
        logging.error(f"Error in scheduled task: {e}")
    finally:
        cursor.close()
        connection.close()