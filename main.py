import os
import sys
from datetime import datetime
from src.core.model import TradingModel
from src.core.evolution import ModelPopulation
from src.core.trader import Trader
from src.data.processor import DataProcessor
from src.config.config import Config
from src.utils.logger import Logger
from src.core.market_trainer import MarketTrainer
import argparse

def setup_environment():
    """Setup the trading environment"""
    # Create necessary directories
    directories = ['models', 'data', 'logs', 'config', 'backups']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Initialize configuration
    config = Config()
    
    # Initialize logger
    logger = Logger()
    
    return config, logger

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Trading System')
    parser.add_argument('--mode', choices=['train', 'trade', 'backtest'],
                       default='train', help='Operation mode')
    parser.add_argument('--tickers', nargs='+', help='List of tickers to process')
    parser.add_argument('--config', help='Path to custom configuration file')
    return parser.parse_args()

def main():
    """Main entry point"""
    # Setup environment
    config, logger = setup_environment()
    logger.info("Starting trading system")

    # Parse arguments
    args = parse_arguments()

    try:
        if args.mode == 'train':
            # Initialize market trainer
            trainer = MarketTrainer()
            
            # Start or resume training
            trainer.train_market()
            
        elif args.mode == 'trade':
            # Initialize trader
            trader = Trader(
                initial_capital=config.trading_config['initial_capital'],
                risk_per_trade=config.trading_config['risk_per_trade'],
                max_positions=config.trading_config['max_positions']
            )
            
            # Start trading
            # Note: Implement real-time trading logic here
            pass
            
        elif args.mode == 'backtest':
            # Initialize components for backtesting
            processor = DataProcessor()
            trader = Trader(
                initial_capital=config.trading_config['initial_capital'],
                risk_per_trade=config.trading_config['risk_per_trade']
            )
            
            # Perform backtesting
            # Note: Implement backtesting logic here
            pass

    except KeyboardInterrupt:
        logger.info("System shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
