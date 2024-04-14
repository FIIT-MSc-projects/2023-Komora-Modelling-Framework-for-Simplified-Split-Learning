
import logging
import os
import sys


def init_logging(logger_name, log_file_path):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    format = logging.Formatter("%(asctime)s: %(message)s")

    print("LOG FILE PATH: ", log_file_path)

    if not os.path.isdir(log_file_path):
        os.makedirs(log_file_path, exist_ok=True)

    fh = logging.FileHandler(filename=f"{log_file_path}/{logger_name}.log",mode='w')
    fh.setFormatter(format)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(format)
    sh.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger