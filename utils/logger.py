import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import requests
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class BotLogger:
    """Custom logger for the trading bot with optional Telegram notifications."""

    def __init__(self, config: dict):
        """
        Initialize the logger with configuration settings.

        Args:
            config: Logging configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("TradingBot")
        self.setup_logger()

    def setup_logger(self):
        """Configure logging handlers and formatting."""
        # Set logging level
        level = getattr(logging, self.config["level"].upper())
        self.logger.setLevel(level)

        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        # File handler with rotation
        file_handler = RotatingFileHandler(
            self.config["file"], maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setFormatter(detailed_formatter)

        # Daily rotating handler for date-based log files
        daily_handler = TimedRotatingFileHandler(
            "logs/daily.log", when="midnight", interval=1, backupCount=30
        )
        daily_handler.setFormatter(detailed_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(daily_handler)
        self.logger.addHandler(console_handler)

        # Create separate error log
        error_handler = RotatingFileHandler(
            "logs/error.log", maxBytes=10 * 1024 * 1024, backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(error_handler)

    def log_trade(self, trade_data: dict):
        """
        Log trade execution details.

        Args:
            trade_data: Dictionary containing trade details
        """
        trade_msg = (
            f"Trade executed: {trade_data['symbol']} | "
            f"Type: {trade_data['type']} | "
            f"Price: {trade_data['price']} | "
            f"Size: {trade_data['size']} | "
            f"Confidence: {trade_data['confidence']:.2f}"
        )
        self.logger.info(trade_msg)

        if self.config["telegram"]["enabled"]:
            self.send_telegram_alert(trade_msg)

    def log_position_update(self, position_data: dict):
        """
        Log position update details.

        Args:
            position_data: Dictionary containing position details
        """
        position_msg = (
            f"Position update: {position_data['symbol']} | "
            f"PnL: {position_data['pnl']:.2f} | "
            f"Current Price: {position_data['current_price']} | "
            f"Stop Loss: {position_data['stop_loss']} | "
            f"Take Profit: {position_data['take_profit']}"
        )
        self.logger.info(position_msg)

    def log_error(self, error: Exception, additional_info: Optional[str] = None):
        """
        Log error details with stack trace.

        Args:
            error: Exception object
            additional_info: Additional context about the error
        """
        error_msg = f"Error occurred: {str(error)}"
        if additional_info:
            error_msg += f" | Context: {additional_info}"

        self.logger.exception(error_msg)

        if self.config["telegram"]["enabled"]:
            self.send_telegram_alert(f"ERROR: {error_msg}")

    def log_performance(self, performance_data: dict):
        """
        Log performance metrics.

        Args:
            performance_data: Dictionary containing performance metrics
        """
        performance_msg = (
            f"Performance update:\n"
            f"Return: {performance_data['return']:.2f}%\n"
            f"Sharpe Ratio: {performance_data['sharpe']:.2f}\n"
            f"Win Rate: {performance_data['win_rate']:.2f}%\n"
            f"Profit Factor: {performance_data['profit_factor']:.2f}"
        )
        self.logger.info(performance_msg)

        if self.config["telegram"]["enabled"]:
            self.send_telegram_alert(performance_msg)

    def send_telegram_alert(self, message: str):
        """
        Send alert to Telegram channel.

        Args:
            message: Message to send
        """
        if not self.config["telegram"]["enabled"]:
            return

        try:
            url = f"https://api.telegram.org/bot{self.config['telegram']['bot_token']}/sendMessage"
            payload = {
                "chat_id": self.config["telegram"]["chat_id"],
                "text": message,
                "parse_mode": "HTML",
            }

            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()

        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {str(e)}")

    def log_model_metrics(self, metrics: dict):
        """
        Log model training and prediction metrics.

        Args:
            metrics: Dictionary containing model metrics
        """
        metrics_msg = (
            f"Model metrics:\n"
            f"Accuracy: {metrics['accuracy']:.2f}\n"
            f"Precision: {metrics['precision']:.2f}\n"
            f"Recall: {metrics['recall']:.2f}\n"
            f"F1 Score: {metrics['f1']:.2f}"
        )
        self.logger.info(metrics_msg)
