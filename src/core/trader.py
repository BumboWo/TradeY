from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging

class TradePosition:
    def __init__(self, ticker: str, entry_price: float, position_size: float,
                 position_type: str, stop_loss: float, take_profit: float):
        self.ticker = ticker
        self.entry_price = entry_price
        self.position_size = position_size
        self.position_type = position_type  # 'long' or 'short'
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = datetime.now()
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.status = 'open'

class Trader:
    def __init__(self, initial_capital: float = 100000, risk_per_trade: float = 0.02,
                 max_positions: int = 5):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.positions: Dict[str, TradePosition] = {}
        self.closed_positions: List[TradePosition] = []
        self.performance_metrics = self._initialize_metrics()

    def _initialize_metrics(self) -> Dict:
        """Initialize performance tracking metrics"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0
        }

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management rules"""
        risk_amount = self.current_capital * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss) / entry_price
        return risk_amount / price_risk

    def open_position(self, ticker: str, entry_price: float, stop_loss: float,
                     take_profit: float, position_type: str) -> Optional[TradePosition]:
        """Open a new trading position"""
        if len(self.positions) >= self.max_positions:
            logging.warning("Maximum number of positions reached")
            return None

        if ticker in self.positions:
            logging.warning(f"Position already exists for {ticker}")
            return None

        position_size = self.calculate_position_size(entry_price, stop_loss)
        
        position = TradePosition(
            ticker=ticker,
            entry_price=entry_price,
            position_size=position_size,
            position_type=position_type,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[ticker] = position
        return position

    def close_position(self, ticker: str, exit_price: float) -> Optional[TradePosition]:
        """Close an existing position"""
        if ticker not in self.positions:
            logging.warning(f"No position found for {ticker}")
            return None

        position = self.positions[ticker]
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        
        # Calculate PnL
        if position.position_type == 'long':
            position.pnl = (exit_price - position.entry_price) * position.position_size
        else:  # short
            position.pnl = (position.entry_price - exit_price) * position.position_size

        position.status = 'closed'
        self.current_capital += position.pnl
        
        # Update metrics
        self._update_metrics(position)
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[ticker]
        
        return position

    def _update_metrics(self, position: TradePosition):
        """Update performance metrics after closing a position"""
        self.performance_metrics['total_trades'] += 1
        
        if position.pnl > 0:
            self.performance_metrics['winning_trades'] += 1
            self.performance_metrics['total_profit'] += position.pnl
        else:
            self.performance_metrics['losing_trades'] += 1
            self.performance_metrics['total_loss'] -= position.pnl

        # Update win rate
        self.performance_metrics['win_rate'] = (
            self.performance_metrics['winning_trades'] / 
            self.performance_metrics['total_trades']
        )

        # Update profit factor
        if self.performance_metrics['total_loss'] != 0:
            self.performance_metrics['profit_factor'] = (
                self.performance_metrics['total_profit'] / 
                self.performance_metrics['total_loss']
            )

        # Update max drawdown
        equity_curve = self._calculate_equity_curve()
        self.performance_metrics['max_drawdown'] = self._calculate_max_drawdown(equity_curve)

        # Update Sharpe ratio
        returns = self._calculate_returns()
        self.performance_metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)

    def _calculate_equity_curve(self) -> np.ndarray:
        """Calculate equity curve from closed positions"""
        equity = self.initial_capital
        equity_curve = [equity]
        
        for position in self.closed_positions:
            equity += position.pnl
            equity_curve.append(equity)
        
        return np.array(equity_curve)

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown from equity curve"""
        peak = equity_curve[0]
        max_drawdown = 0
        
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown

    def _calculate_returns(self) -> np.ndarray:
        """Calculate returns from closed positions"""
        returns = []
        for position in self.closed_positions:
            returns.append(position.pnl / self.initial_capital)
        return np.array(returns)

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(returns) < 2:
            return 0.0
        
        # Assuming risk-free rate of 0 and annualizing
        sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns))
        return sharpe_ratio

    def check_positions(self, current_prices: Dict[str, float]):
        """Check and update all open positions"""
        for ticker, position in list(self.positions.items()):
            current_price = current_prices.get(ticker)
            if current_price is None:
                continue

            # Check stop loss
            if (position.position_type == 'long' and 
                current_price <= position.stop_loss) or \
               (position.position_type == 'short' and 
                current_price >= position.stop_loss):
                self.close_position(ticker, current_price)
                logging.info(f"Stop loss triggered for {ticker}")

            # Check take profit
            elif (position.position_type == 'long' and 
                  current_price >= position.take_profit) or \
                 (position.position_type == 'short' and 
                  current_price <= position.take_profit):
                self.close_position(ticker, current_price)
                logging.info(f"Take profit triggered for {ticker}")

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        return {
            'current_capital': self.current_capital,
            'open_positions': len(self.positions),
            'total_positions': len(self.closed_positions),
            'performance_metrics': self.performance_metrics,
            'current_drawdown': self._calculate_current_drawdown()
        }

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        equity_curve = self._calculate_equity_curve()
        if len(equity_curve) == 0:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)[-1]
        current_equity = self.current_capital
        return (peak - current_equity) / peak if peak > current_equity else 0.0
