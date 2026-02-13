import logging
import os

def setup_logger():
    """Setup the logger"""
    # Get the log level from the environment variable
    # os.getenv() returns the value of the environment variable
    # Default to INFO if not set
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Configure the logger
    logging.basicConfig(
        # Set the log level
        # getattr() returns the value of the named attribute of an object
        # If the attribute is not found, it returns the default value
        level=getattr(logging, log_level, logging.INFO),
        # Set the log format
        # %(levelname)s: the level of the log message
        # %(name)s: the name of the logger
        # %(asctime)s: the time the log message was created
        format='[ %(levelname)s - %(name)s - %(asctime)s ] %(message)s',
        # Set the log handlers
        # StreamHandler: logs to the console
        # FileHandler: logs to a file
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log")
        ]
    )