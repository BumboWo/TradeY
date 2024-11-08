import click
import os
import sys
from datetime import datetime

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.core.market_trainer import MarketTrainer
from src.core.trader import Trader
from src.utils.logger import Logger
from src.config.config import Config

@click.group()
def cli():
    """Advanced Trading System CLI"""
    pass

@cli.command()
@click.option('--mode', type=click.Choice(['full', 'selective', 'resume']), default='full',
              help='Training mode')
@click.option('--tickers', multiple=True, help='Specific tickers to train on')
def train(mode, tickers):
    """Train trading models"""
    trainer = MarketTrainer()
    
    if mode == 'resume':
        click.echo("Resuming previous training session...")
        trainer.train_market()
    elif mode == 'selective' and tickers:
        click.echo(f"Training on selected tickers: {', '.join(tickers)}")
        for ticker in tickers:
            trainer.process_ticker(ticker)
    else:
        click.echo("Starting full market training...")
        trainer.train_market()

@cli.command()
@click.option('--mode', type=click.Choice(['live', 'paper']), default='paper',
              help='Trading mode')
@click.option('--tickers', multiple=True, help='Specific tickers to trade')
def trade(mode, tickers):
    """Start trading"""
    config = Config()
    trader = Trader(
        initial_capital=config.trading_config['initial_capital'],
        risk_per_trade=config.trading_config['risk_per_trade']
    )
    
    click.echo(f"Starting {mode} trading...")
    # Implement trading logic here

@cli.command()
@click.option('--tickers', multiple=True, required=True, help='Tickers to backtest')
@click.option('--start-date', type=click.DateTime(), help='Start date for backtesting')
@click.option('--end-date', type=click.DateTime(), help='End date for backtesting')
def backtest(tickers, start_date, end_date):
    """Run backtesting"""
    click.echo(f"Running backtest for {', '.join(tickers)}")
    # Implement backtesting logic here

@cli.command()
@click.option('--mode', type=click.Choice(['models', 'trading', 'training', 'comparison']),
              required=True, help='Stats mode')
@click.option('--live', is_flag=True, help='Show live stats')
def stats(mode, live):
    """View system statistics"""
    trainer = MarketTrainer()
    
    if mode == 'models':
        stats = trainer.evaluate_market_models()
        for ticker, performance in stats.items():
            click.echo(f"\n{ticker}:")
            click.echo(f"MSE: {performance['mse']:.6f}")
            click.echo(f"MAE: {performance['mae']:.6f}")
            click.echo(f"Fitness: {performance['fitness']:.6f}")
    
    elif mode == 'training':
        progress = trainer.get_training_progress()
        click.echo("\nTraining Progress:")
        click.echo(f"Total Tickers: {progress['total_tickers']}")
        click.echo(f"Processed Tickers: {progress['processed_tickers']}")
        click.echo(f"Progress: {progress['progress_percentage']:.2f}%")

@cli.command()
@click.option('--days', default=30, help='Number of days of old data to clean')
def cleanup(days):
    """Clean up old data and logs"""
    trainer = MarketTrainer()
    trainer.cleanup_old_data(days)
    click.echo(f"Cleaned up data older than {days} days")

if __name__ == '__main__':
    cli()
