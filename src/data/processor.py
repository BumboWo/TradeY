import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import MinMaxScaler
import ta

class DataProcessor:
    def __init__(self, feature_columns: List[str] = None):
        self.feature_columns = feature_columns or [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'MACD',
            'RSI', 'Stoch', 'BB_upper', 'BB_middle', 'BB_lower',
            'ATR', 'OBV', 'Hour', 'Minute', 'DayOfWeek'
        ]
        self.scaler = MinMaxScaler()
        self._is_fitted = False

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset"""
        df = data.copy()
        
        # Trend Indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        
        # Momentum Indicators
        df['RSI'] = ta.momentum.rsi(df['Close'])
        df['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volatility Indicators
        bb_bands = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb_bands.bollinger_hband()
        df['BB_middle'] = bb_bands.bollinger_mavg()
        df['BB_lower'] = bb_bands.bollinger_lband()
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Volume Indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['VWAP'] = ta.volume.volume_weighted_average_price(
            df['High'], df['Low'], df['Close'], df['Volume']
        )
        
        return df

    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = data.copy()
        df['Hour'] = df.index.hour
        df['Minute'] = df.index.minute
        df['DayOfWeek'] = df.index.dayofweek
        return df

    def prepare_data(self, data: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        # Add technical indicators
        df = self.add_technical_indicators(data)
        
        # Add time features
        df = self.add_time_features(df)
        
        # Fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        if not self._is_fitted:
            self.scaler.fit(df[self.feature_columns])
            self._is_fitted = True
        
        scaled_data = self.scaler.transform(df[self.feature_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, df.columns.get_loc('Close')])
        
        return np.array(X), np.array(y)

    def inverse_transform_prices(self, scaled_prices: np.ndarray) -> np.ndarray:
        """Convert scaled prices back to original scale"""
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        # Create dummy array with same shape as feature set
        dummy = np.zeros((len(scaled_prices), len(self.feature_columns)))
        dummy[:, self.feature_columns.index('Close')] = scaled_prices
        
        # Inverse transform
        dummy_inverse = self.scaler.inverse_transform(dummy)
        return dummy_inverse[:, self.feature_columns.index('Close')]

    def calculate_price_derivatives(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price derivatives (momentum, acceleration)"""
        df = data.copy()
        
        # Price changes
        df['Price_Change'] = df['Close'].diff()
        df['Price_Change_Pct'] = df['Close'].pct_change()
        
        # Momentum (rate of change)
        df['Momentum'] = df['Price_Change'].rolling(window=5).mean()
        
        # Acceleration (change in momentum)
        df['Acceleration'] = df['Momentum'].diff()
        
        return df

    def detect_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect candlestick patterns"""
        df = data.copy()
        
        # Basic candlestick features
        df['Body'] = df['Close'] - df['Open']
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Body_Size'] = abs(df['Body'])
        
        # Pattern detection (example patterns)
        df['Doji'] = (df['Body_Size'] <= 0.1 * (df['High'] - df['Low']))
        df['Hammer'] = ((df['Lower_Shadow'] > 2 * df['Body_Size']) & 
                       (df['Upper_Shadow'] <= 0.1 * df['Lower_Shadow']))
        df['Shooting_Star'] = ((df['Upper_Shadow'] > 2 * df['Body_Size']) & 
                             (df['Lower_Shadow'] <= 0.1 * df['Upper_Shadow']))
        
        return df

    def save_state(self, path: str):
        """Save processor state"""
        state = {
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'is_fitted': self._is_fitted
        }
        np.save(path, state, allow_pickle=True)

    @classmethod
    def load_state(cls, path: str) -> 'DataProcessor':
        """Load processor state"""
        state = np.load(path, allow_pickle=True).item()
        processor = cls(feature_columns=state['feature_columns'])
        processor.scaler = state['scaler']
        processor._is_fitted = state['is_fitted']
        return processor
