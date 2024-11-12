import pandas as pd
import numpy as np
from typing import Tuple, List
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
        """Add various technical indicators to the dataset."""
        df = data.copy()

        # Ensure necessary columns exist
        required_columns = ['Close', 'High', 'Low', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure data is in the correct format
        df[required_columns] = df[required_columns].astype(float)

        # Calculate trend indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'].squeeze(), window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'].squeeze(), window=50)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'].squeeze(), window=20)
        df['EMA_50'] = ta.trend.ema_indicator(df['Close'].squeeze(), window=50)
        df['MACD'] = ta.trend.macd_diff(df['Close'].squeeze())
        df['ADX'] = ta.trend.adx(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze())

        # Calculate momentum indicators
        df['RSI'] = ta.momentum.rsi(df['Close'].squeeze())
        df['Stoch'] = ta.momentum.stoch(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze())
        df['MFI'] = ta.volume.money_flow_index(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze(), df['Volume'].squeeze())

        # Calculate volatility indicators
        bb_bands = ta.volatility.BollingerBands(df['Close'].squeeze())
        df['BB_upper'] = bb_bands.bollinger_hband()
        df['BB_middle'] = bb_bands.bollinger_mavg()
        df['BB_lower'] = bb_bands.bollinger_lband()
        df['ATR'] = ta.volatility.average_true_range(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze())

        # Calculate volume indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'].squeeze(), df['Volume'].squeeze())
        df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze(), df['Volume'].squeeze())

        # Handle NaN values introduced by indicators
        df = df.dropna()

        return df


    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['Hour'] = df.index.hour
        df['Minute'] = df.index.minute
        df['DayOfWeek'] = df.index.dayofweek
        return df

    def prepare_data(self, data: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        df = self.add_technical_indicators(data)
        df = self.add_time_features(df)
        df = df.ffill().bfill()

        if not self._is_fitted:
            self.scaler.fit(df[self.feature_columns])
            self._is_fitted = True
        
        scaled_data = self.scaler.transform(df[self.feature_columns])

        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, df.columns.get_loc('Close')])

        return np.array(X), np.array(y)

    def save_state(self, path: str):
        state = {
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'is_fitted': self._is_fitted
        }
        np.save(path, state, allow_pickle=True)

    @classmethod
    def load_state(cls, path: str) -> 'DataProcessor':
        state = np.load(path, allow_pickle=True).item()
        processor = cls(feature_columns=state['feature_columns'])
        processor.scaler = state['scaler']
        processor._is_fitted = state['is_fitted']
        return processor
