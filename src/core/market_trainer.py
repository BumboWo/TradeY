import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import os
import json
import time
from tqdm import tqdm

from src.data.processor import DataProcessor
from src.core.evolution import ModelPopulation
from src.core.model import TradingModel
from src.utils.logger import Logger
from src.config.config import Config

class MarketTrainer:
    def __init__(self, save_dir: str = 'market_training'):
        """Initialize the market trainer with necessary configurations and directories."""
        self.save_dir = save_dir
        self.checkpoint_file = os.path.join(save_dir, 'training_checkpoint.json')
        self.config = Config()
        self.logger = Logger()
        
        # Create necessary directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'data'), exist_ok=True)
        
        self.data_processor = DataProcessor()
        self.checkpoint = self._load_checkpoint()
        self.should_stop = False

    def _load_checkpoint(self) -> Dict:
        """Load training checkpoint if exists."""
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
        """Save training checkpoint."""
        self.checkpoint['last_update'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=4)

    def get_market_tickers(self) -> List[str]:
        """Get list of top 100 market tickers."""
        TOP_100_TICKERS = [
            # Technology
            'AAPL',  # Apple
            'MSFT',  # Microsoft
            'GOOGL', # Alphabet Class A
            'AMZN',  # Amazon
            'NVDA',  # NVIDIA
            'META',  # Meta Platforms
            'TSLA',  # Tesla
            'TSM',   # Taiwan Semiconductor
            'AVGO',  # Broadcom
            'ORCL',  # Oracle
            
            # Finance
            'JPM',   # JPMorgan Chase
            'V',     # Visa
            'MA',    # Mastercard
            'BAC',   # Bank of America
            'WFC',   # Wells Fargo
            'MS',    # Morgan Stanley
            'GS',    # Goldman Sachs
            'BLK',   # BlackRock
            'C',     # Citigroup
            'AXP',   # American Express
            
            # Healthcare
            'JNJ',   # Johnson & Johnson
            'UNH',   # UnitedHealth
            'PFE',   # Pfizer
            'MRK',   # Merck
            'ABT',   # Abbott Laboratories
            'TMO',   # Thermo Fisher Scientific
            'DHR',   # Danaher
            'BMY',   # Bristol-Myers Squibb
            'LLY',   # Eli Lilly
            'ABBV',  # AbbVie
            
            # Consumer
            'PG',    # Procter & Gamble
            'KO',    # Coca-Cola
            'PEP',   # PepsiCo
            'COST',  # Costco
            'WMT',   # Walmart
            'NKE',   # Nike
            'MCD',   # McDonald's
            'SBUX',  # Starbucks
            'DIS',   # Disney
            'HD',    # Home Depot
            
            # Industrial
            'CAT',   # Caterpillar
            'BA',    # Boeing
            'HON',   # Honeywell
            'GE',    # General Electric
            'MMM',   # 3M
            'UPS',   # United Parcel Service
            'RTX',   # Raytheon Technologies
            'LMT',   # Lockheed Martin
            'DE',    # Deere & Company
            'FDX',   # FedEx
            
            # Energy
            'XOM',   # ExxonMobil
            'CVX',   # Chevron
            'COP',   # ConocoPhillips
            'SLB',   # Schlumberger
            'EOG',   # EOG Resources
            'PSX',   # Phillips 66
            'VLO',   # Valero Energy
            'MPC',   # Marathon Petroleum
            'OXY',   # Occidental Petroleum
            
            # Telecommunications
            'T',     # AT&T
            'VZ',    # Verizon
            'TMUS',  # T-Mobile
            'CMCSA', # Comcast
            'NFLX',  # Netflix
            'CRM',   # Salesforce
            'CSCO',  # Cisco
            'INTC',  # Intel
            'AMD',   # Advanced Micro Devices
            'QCOM',  # Qualcomm
            
            # Real Estate & Materials
            'AMT',   # American Tower
            'PLD',   # Prologis
            'CCI',   # Crown Castle
            'SPG',   # Simon Property Group
            'LIN',   # Linde
            'APD',   # Air Products & Chemicals
            'SHW',   # Sherwin-Williams
            'FCX',   # Freeport-McMoRan
            'NEM',   # Newmont
            'DOW',   # Dow Inc.
            
            # Financial Services & Insurance
            'BRK-B', # Berkshire Hathaway
            'SPY',   # SPDR S&P 500 ETF
            'QQQ',   # Invesco QQQ Trust
            'IWM',   # iShares Russell 2000 ETF
            'EFA',   # iShares MSCI EAFE ETF
            'PRU',   # Prudential
            'AIG',   # American International Group
            'ALL',   # Allstate
            'TRV',   # Travelers Companies
            
            # Others
            'PYPL',  # PayPal
            'SQ',    # Block (Square)
            'UBER',  # Uber
            'ABNB',  # Airbnb
            'ZM',    # Zoom
            'SHOP',  # Shopify
            'SNAP',  # Snap
            'PINS',  # Pinterest
            'U',     # Unity Software
            'PLTR'   # Palantir
        ]
        
        try:
            with open(os.path.join(self.save_dir, 'market_tickers.json'), 'w') as f:
                json.dump(TOP_100_TICKERS, f)
            return TOP_100_TICKERS
        except Exception as e:
            self.logger.error(f"Error saving market tickers: {str(e)}")
            return TOP_100_TICKERS

    def download_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Download historical data for a ticker."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval='5m'
            )
            
            if len(data) < 100:
                self.logger.warning(f"Insufficient data for {ticker}")
                return None
                
            cache_file = os.path.join(self.save_dir, 'data', f'{ticker}_data.parquet')
            data.to_parquet(cache_file)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error downloading data for {ticker}: {str(e)}")
            return None

    def train_ticker_model(self, ticker: str, data: pd.DataFrame) -> Optional[Dict]:
        """Train model for a specific ticker."""
        try:
            processed_data = self.data_processor.add_technical_indicators(data)
            processed_data = self.data_processor.add_time_features(processed_data)
            
            X, y = self.data_processor.prepare_data(
                processed_data,
                lookback=self.config.training_config.get('sequence_length', 60)
            )
            
            if len(X) < 1000:
                self.logger.warning(f"Insufficient samples for {ticker}")
                return None
            
            self.logger.info(f"Training data shapes - X: {X.shape}, y: {y.shape}")
            self.logger.info(f"Number of features: {X.shape[2]}")
            
            population = ModelPopulation(
                input_shape=(X.shape[1], X.shape[2]),
                population_size=self.config.training_config['population_size'],
                mutation_rate=self.config.training_config['mutation_rate'],
                mutation_scale=self.config.training_config.get('mutation_scale', 0.1)
            )
            
            best_generation = 0
            for generation in range(self.config.training_config['generations']):
                if self.should_stop:
                    break
                
                population.evolve(X, y)
                
                stats = population.get_population_stats()
                self.logger.info(
                    f"Ticker: {ticker}, Generation: {generation}, "
                    f"Best Fitness: {stats['best_fitness']:.4f}, "
                    f"Mean Fitness: {stats['mean_fitness']:.4f}"
                )
                best_generation = generation
            
            best_model = population.get_best_model()
            if best_model is None:
                self.logger.error(f"No best model found for {ticker}")
                return None
                
            model_dir = os.path.join(self.save_dir, 'models', ticker)
            os.makedirs(model_dir, exist_ok=True)
            
            try:
                population.save_population(model_dir)
                processor_path = os.path.join(model_dir, 'processor.npy')
                self.data_processor.save_state(processor_path)
                
            except Exception as e:
                self.logger.error(f"Error saving model for {ticker}: {str(e)}")
                return None
            
            final_stats = population.get_population_stats()
            
            return {
                'ticker': ticker,
                'model_dir': model_dir,
                'fitness': final_stats['best_fitness'],
                'mean_fitness': final_stats['mean_fitness'],
                'generations': best_generation + 1,
                'timestamp': datetime.now().isoformat(),
                'feature_columns': self.data_processor.feature_columns,
                'training_stats': final_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error training model for {ticker}: {str(e)}")
            return None

    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate the shape and content of the training data."""
        if X.ndim != 3:
            self.logger.error(f"Expected X to have 3 dimensions, got shape {X.shape}")
            return False
        if y.ndim != 1:
            self.logger.error(f"Expected y to have 1 dimension, got shape {y.shape}")
            return False
        if np.isnan(X).any() or np.isnan(y).any():
            self.logger.error("Data contains NaN values")
            return False
        if np.isinf(X).any() or np.isinf(y).any():
            self.logger.error("Data contains infinite values")
            return False
        return True

    def process_ticker(self, ticker: str) -> Optional[Dict]:
        """Process a single ticker."""
        if ticker in self.checkpoint['processed_tickers']:
            self.logger.info(f"Ticker {ticker} already processed")
            return None
            
        try:
            data = self.download_ticker_data(ticker)
            if data is None:
                return None
            
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
        """Main training loop for all market tickers."""
        try:
            tickers = self.get_market_tickers()
            remaining_tickers = [t for t in tickers 
                                 if t not in self.checkpoint['processed_tickers']]
            
            self.logger.info(f"Starting market training with {len(remaining_tickers)} tickers")
            
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

    def evaluate_market_models(self) -> Dict:
        """Evaluate performance of all trained models."""
        performance = {}
        
        for ticker, model_info in self.checkpoint['best_models'].items():
            try:
                model = TradingModel.load(model_info['model_path'])
                data_processor = DataProcessor.load_state(model_info['processor_path'])
                
                data = self.download_ticker_data(ticker)
                if data is None:
                    continue
                
                processed_data = data_processor.add_technical_indicators(data)
                processed_data = data_processor.add_time_features(processed_data)
                
                X, y = data_processor.prepare_data(processed_data)
                
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
        
        self.checkpoint['performance_metrics'] = performance
        self._save_checkpoint()
        
        return performance

    def get_training_progress(self) -> Dict:
        """Get current training progress."""
        total_tickers = len(self.get_market_tickers())
        processed_tickers = len(self.checkpoint['processed_tickers'])
        
        return {
            'total_tickers': total_tickers,
            'processed_tickers': processed_tickers,
            'progress_percentage': (processed_tickers / total_tickers) * 100,
            'last_update': self.checkpoint['last_update'],
            'current_generation': self.checkpoint['current_generation']
        }

    def cleanup_old_data(self, days: int = 30):
        """Clean up old cached data."""
        current_time = time.time()
        data_dir = os.path.join(self.save_dir, 'data')
        
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.getmtime(filepath) < current_time - (days * 86400):
                os.remove(filepath)
                self.logger.info(f"Removed old data file: {filename}")
