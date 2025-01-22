import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class PerformanceMetrics:
    """Calculate and track trading performance metrics."""

    def __init__(self):
        """Initialize the performance metrics tracker."""
        self.trades: List[Dict] = []
        self.daily_returns: List[float] = []
        self.portfolio_values: List[Dict] = []

    def add_trade(self, trade: Dict):
        """
        Add a completed trade to the history.

        Args:
            trade: Dictionary containing trade details
        """
        self.trades.append(
            {
                "timestamp": datetime.now(),
                "symbol": trade["symbol"],
                "type": trade["type"],
                "entry_price": trade["entry_price"],
                "exit_price": trade["exit_price"],
                "size": trade["size"],
                "pnl": trade["pnl"],
                "hold_time": (trade["exit_time"] - trade["entry_time"]).total_seconds()
                / 3600,
            }
        )

    def update_portfolio_value(self, value: float):
        """
        Update portfolio value history.

        Args:
            value: Current portfolio value
        """
        self.portfolio_values.append({"timestamp": datetime.now(), "value": value})

        # Calculate daily return if we have enough data
        if len(self.portfolio_values) >= 2:
            prev_value = self.portfolio_values[-2]["value"]
            daily_return = (value - prev_value) / prev_value
            self.daily_returns.append(daily_return)

    def calculate_returns(self) -> Dict[str, float]:
        """Calculate various return metrics."""
        if not self.portfolio_values:
            return {
                "total_return": 0.0,
                "daily_return": 0.0,
                "annual_return": 0.0,
                "volatility": 0.0,
            }

        initial_value = self.portfolio_values[0]["value"]
        current_value = self.portfolio_values[-1]["value"]

        total_return = (current_value - initial_value) / initial_value

        if self.daily_returns:
            daily_return = np.mean(self.daily_returns)
            volatility = np.std(self.daily_returns) * np.sqrt(252)  # Annualized
            annual_return = daily_return * 252
        else:
            daily_return = volatility = annual_return = 0.0

        return {
            "total_return": total_return * 100,  # Convert to percentage
            "daily_return": daily_return * 100,
            "annual_return": annual_return * 100,
            "volatility": volatility * 100,
        }

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        if not self.daily_returns:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
            }

        # Risk-free rate (assuming 2% annual)
        rf_daily = 0.02 / 252

        # Sharpe Ratio
        excess_returns = np.array(self.daily_returns) - rf_daily
        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(self.daily_returns)

        # Sortino Ratio
        negative_returns = [r for r in excess_returns if r < 0]
        sortino = (
            np.sqrt(252) * np.mean(excess_returns) / np.std(negative_returns)
            if negative_returns
            else 0
        )

        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + np.array(self.daily_returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(min(drawdowns))

        # Value at Risk (95% confidence)
        var_95 = np.percentile(self.daily_returns, 5)

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown * 100,  # Convert to percentage
            "var_95": var_95 * 100,
        }

    def calculate_trade_metrics(self) -> Dict[str, float]:
        """Calculate trade-specific metrics."""
        if not self.trades:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_profit_per_trade": 0.0,
                "avg_hold_time": 0.0,
            }

        # Calculate basic trade statistics
        profitable_trades = [t for t in self.trades if t["pnl"] > 0]
        losing_trades = [t for t in self.trades if t["pnl"] <= 0]

        win_rate = len(profitable_trades) / len(self.trades)

        total_profit = sum(t["pnl"] for t in profitable_trades)
        total_loss = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss != 0 else float("inf")

        avg_profit = sum(t["pnl"] for t in self.trades) / len(self.trades)
        avg_hold_time = sum(t["hold_time"] for t in self.trades) / len(self.trades)

        return {
            "win_rate": win_rate * 100,  # Convert to percentage
            "profit_factor": profit_factor,
            "avg_profit_per_trade": avg_profit,
            "avg_hold_time": avg_hold_time,
        }

    def calculate_position_metrics(self) -> Dict[str, float]:
        """Calculate position-related metrics."""
        if not self.trades:
            return {
                "avg_position_size": 0.0,
                "largest_position": 0.0,
                "avg_leverage": 0.0,
            }

        position_sizes = [t["size"] * t["entry_price"] for t in self.trades]

        return {
            "avg_position_size": np.mean(position_sizes),
            "largest_position": max(position_sizes),
            "avg_leverage": np.mean([t.get("leverage", 1.0) for t in self.trades]),
        }

    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive performance summary."""
        return {
            "returns": self.calculate_returns(),
            "risk": self.calculate_risk_metrics(),
            "trades": self.calculate_trade_metrics(),
            "positions": self.calculate_position_metrics(),
        }

    def export_trade_history(self, format: str = "csv") -> Optional[str]:
        """
        Export trade history to CSV or JSON.

        Args:
            format: Export format ('csv' or 'json')

        Returns:
            Path to exported file
        """
        if not self.trades:
            return None

        df = pd.DataFrame(self.trades)

        if format == "csv":
            path = f'data/trade_history_{datetime.now().strftime("%Y%m%d")}.csv'
            df.to_csv(path, index=False)
            return path
        elif format == "json":
            path = f'data/trade_history_{datetime.now().strftime("%Y%m%d")}.json'
            df.to_json(path, orient="records")
            return path
        else:
            return None
