import json
from typing import Dict, Any
import os

class Config:
    def __init__(self, config_dir: str = 'config'):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        
        self.training_config = self.load_config('training.json')
        self.trading_config = self.load_config('trading.json')
        self.model_config = self.load_config('model.json')
        
        # Create default configs if they don't exist
        self.create_default_configs()

    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from file"""
        filepath = os.path.join(self.config_dir, filename)
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_config(self, config: Dict[str, Any], filename: str):
        """Save configuration to file"""
        filepath = os.path.join(self.config_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)

    def create_default_configs(self):
        """Create default configuration files if they don't exist"""
        # Default training configuration
        default_training = {
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2,
            'learning_rate': 0.001,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'min_lr': 0.0001,
            'population_size': 20,
            'mutation_rate': 0.1,
            'mutation_scale': 0.1,
            'generations': 50,
            'lookback_period': 60
        }

        # Default trading configuration
        default_trading = {
            'initial_capital': 100000,
            'risk_per_trade': 0.02,
            'max_positions': 5,
            'stop_loss_atr_multiplier': 2,
            'take_profit_atr_multiplier': 3,
            'max_drawdown': 0.2,
            'min_trading_volume': 100000,
            'position_sizing_method': 'risk_based',
            'trading_timeframe': '5m',
            'max_spread_percent': 0.001
        }

        # Default model configuration
        default_model = {
            'lstm_layers': [100, 100, 50],
            'dense_layers': [50, 25],
            'dropout_rate': 0.2,
            'activation': 'relu',
            'output_activation': 'linear',
            'loss_function': 'huber',
            'metrics': ['mae'],
            'optimizer': 'adam'
        }

        # Save default configs if they don't exist
        if not self.training_config:
            self.save_config(default_training, 'training.json')
            self.training_config = default_training

        if not self.trading_config:
            self.save_config(default_trading, 'trading.json')
            self.trading_config = default_trading

        if not self.model_config:
            self.save_config(default_model, 'model.json')
            self.model_config = default_model

    def update_config(self, config_type: str, updates: Dict[str, Any]):
        """Update specific configuration with new values"""
        if config_type == 'training':
            self.training_config.update(updates)
            self.save_config(self.training_config, 'training.json')
        elif config_type == 'trading':
            self.trading_config.update(updates)
            self.save_config(self.trading_config, 'trading.json')
        elif config_type == 'model':
            self.model_config.update(updates)
            self.save_config(self.model_config, 'model.json')
        else:
            raise ValueError(f"Unknown config type: {config_type}")

    def get_config(self, config_type: str) -> Dict[str, Any]:
        """Get specific configuration"""
        if config_type == 'training':
            return self.training_config
        elif config_type == 'trading':
            return self.trading_config
        elif config_type == 'model':
            return self.model_config
        else:
            raise ValueError(f"Unknown config type: {config_type}")
