import logging
import os
from datetime import datetime
from typing import Optional
import sys

class Logger:
    def __init__(self, log_dir: str = 'logs', log_level: int = logging.INFO):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'trading_system_{timestamp}.log')
        
        # Configure logging
        self.logger = logging.getLogger('TradingSystem')
        self.logger.setLevel(log_level)
        
        # Create handlers
        self.setup_handlers(log_file, log_level)
        
        self.logger.info("Logger initialized")

    def setup_handlers(self, log_file: str, log_level: int):
        """Setup file and console handlers"""
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def log(self, level: str, message: str, extra: Optional[dict] = None):
        """Log a message with specified level"""
        if extra is None:
            extra = {}

        log_func = getattr(self.logger, level.lower())
        log_func(message, extra=extra)

    def info(self, message: str, extra: Optional[dict] = None):
        """Log info message"""
        self.log('info', message, extra)

    def warning(self, message: str, extra: Optional[dict] = None):
        """Log warning message"""
        self.log('warning', message, extra)

    def error(self, message: str, extra: Optional[dict] = None):
        """Log error message"""
        self.log('error', message, extra)

    def critical(self, message: str, extra: Optional[dict] = None):
        """Log critical message"""
        self.log('critical', message, extra)

    def debug(self, message: str, extra: Optional[dict] = None):
        """Log debug message"""
        self.log('debug', message, extra)

    def get_latest_logs(self, n: int = 100) -> list:
        """Get the latest n log entries"""
        log_files = sorted([f for f in os.listdir(self.log_dir) if f.endswith('.log')])
        if not log_files:
            return []

        latest_log = os.path.join(self.log_dir, log_files[-1])
        with open(latest_log, 'r') as f:
            logs = f.readlines()[-n:]
        return logs

    def clear_old_logs(self, days: int = 30):
        """Clear logs older than specified days"""
        current_time = datetime.now().timestamp()
        for filename in os.listdir(self.log_dir):
            filepath = os.path.join(self.log_dir, filename)
            if os.path.getmtime(filepath) < current_time - (days * 86400):
                os.remove(filepath)
                self.logger.info(f"Removed old log file: {filename}")
