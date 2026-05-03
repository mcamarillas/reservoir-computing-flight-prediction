import logging
import sys

def get_logger(name: str="default") -> logging.Logger:
    """
    Given a name returns a logger formatted
    """
    logger = logging.getLogger(name)
    
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] - %(levelname)s [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger