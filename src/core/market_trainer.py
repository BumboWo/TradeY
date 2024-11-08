import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import os
import json
import logging
from tqdm import tqdm
import time

from src.data.processor import DataProcessor
from src.core.evolution import ModelPopulation
from src.core.model import TradingModel
from src.utils.logger import Logger
from src.config.config import Config

class MarketTrainer:
    def __init__(self, save_dir: str = 'market_training'):
        self.save_dir = save_dir
        self.checkpoint_file = os.path.join(save_dir, 'training_checkpoint.json')
        self.config = Config()
        self.logger = Logger()
        
        # Create necessary directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'data'), exist_ok=True)
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.checkpoint = self._load_checkpoint()
        self.should_stop = False

    def _load_checkpoint(self) -> Dict:
        """Load training checkpoint if exists"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {
            'processed_tickers': [],
            'current_generation': 0,
            'last_update': None,
            'performance_metrics': {},
            'best_models': {}
        }

    def _save_checkpoint(self):
        """Save training checkpoint"""
        self.checkpoint['last_update'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=4)

    def get_market_tickers(self) -> List[str]:
        """Get list of all available market tickers"""
        try:
            # Get S&P 500 tickers
            sp500 = pd.read_html(
                'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            )[0]
            sp500_tickers = sp500['Symbol'].tolist()

            # Get NASDAQ tickers
            nasdaq = pd.read_csv('https://www.nasdaq.com/market-activity/stocks/screener',
                               usecols=['Symbol'])
            nasdaq_tickers = nasdaq['Symbol'].tolist()

            # Combine and remove duplicates
            all_tickers = list(set(sp500_tickers + nasdaq_tickers))
            
            # Save tickers list
            with open(os.path.join(self.save_dir, 'market_tickers.json'), 'w') as f:
                json.dump(all_tickers, f)
            
            return all_tickers
        except Exception as e:
            self.logger.error(f"Error fetching market tickers: {str(e)}")
            
            # Try to load from backup
            if os.path.exists(os.path.join(self.save_dir, 'market_tickers.json')):
                with open(os.path.join(self.save_dir, 'market_tickers.json'), 'r') as f:
                    return json.load(f)
            raise

    def download_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Download historical data for a ticker"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data
            
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval='5m'
            )
            
            if len(data) < 100:  # Minimum data requirement
                self.logger.warning(f"Insufficient data for {ticker}")
                return None
                
            # Save data to cache
            cache_file = os.path.join(self.save_dir, 'data', f'{ticker}_data.parquet')
            data.to_parquet(cache_file)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error downloading data for {ticker}: {str(e)}")
            return None

    def train_ticker_model(self, ticker: str, data: pd.DataFrame) -> Optional[Dict]:
        """Train model for a specific ticker"""
        try:
            # Prepare data
            processed_data = self.data_processor.add_technical_indicators(data)
            X, y = self.data_processor.prepare_data(processed_data)
            
            if len(X) < 1000:  # Minimum samples requirement
                self.logger.warning(f"Insufficient samples for {ticker}")
                return None
            
            # Initialize population
            population = ModelPopulation(
                input_shape=(X.shape[1], X.shape[2]),
                population_size=self.config.training_config['population_size'],
                mutation_rate=self.config.training_config['mutation_rate']
            )
            
            # Train through generations
            for generation in range(self.config.training_config['generations']):
                if self.should_stop:
                    break
                    
                population.evolve(X, y)
                
                # Log progress
                stats = population.get_population_stats()
                self.logger.info(f"Ticker: {ticker}, Generation: {generation}, "
                               f"Best Fitness: {stats['best_fitness']:.4f}")
            
            # Save best model
            best_model = population.get_best_model()
            model_path = os.path.join(self.save_dir, 'models', f'{ticker}_model.h5')
            best_model.save(model_path)
            
            return {
                'ticker': ticker,
                'model_path': model_path,
                'fitness': population.best_fitness,
                'generations': generation + 1,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error training model for {ticker}: {str(e)}")
            return None

    def process_ticker(self, ticker: str) -> Optional[Dict]:
        """Process a single ticker"""
        if ticker in self.checkpoint['processed_tickers']:
            self.logger.info(f"Ticker {ticker} already processed")
            return None
            
        try:
            # Download data
            data = self.download_ticker_data(ticker)
            if data is None:
                return None
            
            # Train model
            result = self.train_ticker_model(ticker, data)
            if result is not None:
                self.checkpoint['processed_tickers'].append(ticker)
                self.checkpoint['best_models'][ticker] = result
                self._save_checkpoint()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {str(e)}")
            return None

    def train_market(self):
        """Main training loop for all market tickers"""
        try:
            # Get market tickers
            tickers = self.get_market_tickers()
            
            # Filter out already processed tickers
            remaining_tickers = [t for t in tickers 
                               if t not in self.checkpoint['processed_tickers']]
            
            self.logger.info(f"Starting market training with {len(remaining_tickers)} tickers")
            
            # Process tickers in parallel
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                futures = [executor.submit(self.process_ticker, ticker) 
                          for ticker in remaining_tickers]
                
                for future in tqdm(futures):
                    if self.should_stop:
                        break
                    
                    result = future.result()
                    if result:
                        self.logger.info(f"Successfully processed {result['ticker']}")
            
            if not self.should_stop:
                self.logger.info("Market training completed successfully!")
            else:
                self.logger.info("Market training paused. Progress saved.")
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.should_stop = True
        except Exception as e:
            self.logger.error(f"Error in market training: {str(e)}")
        finally:
            self._save_checkpoint()

    def get_training_progress(self) -> Dict:
        """Get current training progress"""
        total_tickers = len(self.get_market_tickers())
        processed_tickers = len(self.checkpoint['processed_tickers'])
        
        return {
            'total_tickers': total_tickers,
            'processed_tickers': processed_tickers,
            'progress_percentage': (processed_tickers / total_tickers) * 100,
            'last_update': self.checkpoint['last_update'],
            'current_generation': self.checkpoint['current_generation']
        }

    def evaluate_market_models(self) -> Dict:
        """Evaluate performance of all trained models"""
        performance = {}
        
        for ticker, model_info in self.checkpoint['best_models'].items():
            try:
                # Load model
                model = TradingModel.load(model_info['model_path'])
                
                # Get recent data for evaluation
                data = self.download_ticker_data(ticker)
                if data is None:
                    continue
                
                # Prepare evaluation data
                processed_data = self.data_processor.add_technical_indicators(data)
                X, y = self.data_processor.prepare_data(processed_data)
                
                # Evaluate model
                predictions = model.predict(X)
                mse = np.mean((y - predictions.flatten()) ** 2)
                mae = np.mean(np.abs(y - predictions.flatten()))
                
                performance[ticker] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'fitness': model_info['fitness'],
                    'last_evaluated': datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating model for {ticker}: {str(e)}")
                continue
        
        # Save evaluation results
        self.checkpoint['performance_metrics'] = performance
        self._save_checkpoint()
        
        return performance

    def cleanup_old_data(self, days: int = 30):
        """Clean up old cached data"""
        current_time = time.time()
        data_dir = os.path.join(self.save_dir, 'data')
        
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.getmtime(filepath) < current_time - (days * 86400):
                os.remove(filepath)
                self.logger.info(f"Removed old data file: {filename}")
