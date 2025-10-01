 import logging
 import sys


 def setup_logger(level: str = "INFO", name: str = "cifar10_v7") -> logging.Logger:
     logger = logging.getLogger(name)
     logger.setLevel(getattr(logging, level.upper()))
     logger.handlers.clear()
     fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     ch = logging.StreamHandler(sys.stdout)
     ch.setLevel(getattr(logging, level.upper()))
     ch.setFormatter(fmt)
     logger.addHandler(ch)
     return logger


