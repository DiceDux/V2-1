# utils/logger.py

import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    os.makedirs("logs", exist_ok=True)
    file_path = os.path.join("logs", log_file)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    handler = logging.FileHandler(file_path, encoding="utf-8")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger

