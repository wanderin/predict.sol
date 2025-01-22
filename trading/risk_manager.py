import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class RiskManager:
    """Manages trading risk and position sizing."""
    
    def __init__(self, config: Dict):
        """
        Initialize the risk manager.
        
        Args:
            config: Risk management configuration dictionary
        """
        self.config = config
        self.portfolio_value = 0.0
        self.daily_pnl = []
        self.positions = {}
        self.last_reset = datetime.now()
        
    def calculate_position_size(self, signal: int, confidence: float, 
                              current_price: float) -> Optional[float]:
        """
        Calculate the appropriate position size for a trade.
        
        Args:
            signal: Trading signal (1 for buy, -1 for sell)
            confidence: Model prediction confidence
            current_price: Current asset price
            
        Returns:
            Position size in base currency units or None if trade should not be taken
        """
        try:
            # Check if we're within risk limits
            if not self.check_risk_limits():
                return None
                
            # Calculate Kelly Criterion position size
            win_prob = confidence
            win_loss_ratio = abs(self.config['trading']['take_profit'] / 
                               self.config['trading']['stop_loss'])
                               
            kelly_size = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
            
            # Apply Kelly fraction
            kelly_size *= self.config['risk']['position_sizing']['kelly_fraction']
            
            # Calculate position size based on portfolio value and risk per trade
            risk_amount = self.portfolio_value * self.config['risk']['position_sizing']['risk_per_trade']
            position_size = risk_amount / (current_price * abs(self.config['trading']['stop_loss']))
            
            # Apply Kelly criterion constraint
            position_size *= kelly_size
            
            # Apply maximum position size constraint
            max_position_size = self.portfolio_value * self.config['trading']['position_size']
            position_size = min(position_size, max_position_size)
            
            return position_size if position_size > 0 else None
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return None
            
    def check_risk_limits(self) -> bool:
        """
        Check if current risk levels are within acceptable limits.
        
        Returns:
            Boolean indicating whether new trades can be taken
        """
        # Check daily loss limit
        daily_loss = self.calculate_daily_loss()
        if daily_loss < -self.config['risk']['max_daily_loss'] * self.portfolio_value:
            logger.warning("Daily loss limit reached")
            return False
            
        # Check maximum drawdown
        if self.calculate_drawdown() > self.config['risk']['max_drawdown']:
            logger.warning("Maximum drawdown reached")
            return False
            
        # Check maximum positions
        if len(self.positions) >= self.config['trading']['max_positions']:
            logger.warning("Maximum number of positions reached")
            return False
            
        return True
        
    def calculate_daily_loss(self) -> float:
        """Calculate the current day's profit/loss."""
        if datetime.now().date() > self.last_reset.date():
            self.daily_pnl = []
            self.last_reset = datetime.now()
            
        return sum(self.daily_pnl)
        
    def calculate_drawdown(self) -> float:
        """Calculate the current drawdown percentage."""
        if not self.daily_pnl:
            return 0.0
            
        cumulative_returns = pd.Series(self.daily_pnl).cumsum()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return abs(float(drawdown.min()))
        
    def update_portfolio_value(self, value: float):
        """Update the current portfolio value."""
        self.portfolio_value = value
        
    def record_trade_pnl(self, pnl: float):
        """Record the profit/loss from a trade."""
        self.daily_pnl.append(pnl)
        
    def calculate_stop_loss(self, entry_price: float, position_type: str) -> float:
        """Calculate stop loss price for a position."""
        stop_percentage = self.config['trading']['stop_loss']
        
        if position_type == 'long':
            return entry_price * (1 - stop_percentage)
        else:
            return entry_price * (1 + stop_percentage)
            
    def calculate_take_profit(self, entry_price: float, position_type: str) -> float:
        """Calculate take profit price for a position."""
        take_profit_percentage = self.config['trading']['take_profit']
        
        if position_type == 'long':
            return entry_price * (1 + take_profit_percentage)
        else:
            return entry_price * (1 - take_profit_percentage)
            
    def update_trailing_stop(self, position_id: str, current_price: float):
        """Update trailing stop for a position."""
        if position_id not in self.positions:
            return
            
        position = self.positions[position_id]
        trailing_distance = self.config['trading']['trailing_distance']
        
        if position['type'] == 'long':
            new_stop = current_price * (1 - trailing_distance)
            if new_stop > position['stop_loss']:
                position['stop_loss'] = new_stop
        else:
            new_stop = current_price * (1 + trailing_distance)
            if new_stop < position['stop_loss']:
                position['stop_loss'] = new_stop
                
    def should_close_position(self, position_id: str, current_price: float) -> bool:
        """Check if a position should be closed based on risk parameters."""
        if position_id not in self.positions:
            return False
            
        position = self.positions[position_id]
        
        # Check stop loss
        if position['type'] == 'long' and current_price <= position['stop_loss']:
            return True
        if position['type'] == 'short' and current_price >= position['stop_loss']:
            return True
            
        # Check take profit
        if position['type'] == 'long' and current_price >= position['take_profit']:
            return True
        if position['type'] == 'short' and current_price <= position['take_profit']:
            return True
            
        return False        