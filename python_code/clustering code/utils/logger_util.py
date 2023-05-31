import logging
import os
import sys


def create_logger(name, save_to=None, std=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if std:
        std_handler = logging.StreamHandler(sys.stdout)
        std_handler.setLevel(logging.DEBUG)
        std_handler.setFormatter(formatter)
        logger.addHandler(std_handler)
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        file_handler = logging.FileHandler(save_to,  mode='a+')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
