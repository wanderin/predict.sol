import logging
import asyncio
from typing import Dict, Optional, List
from datetime import datetime
import uuid

import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)


class TradingEngine:
    """Handles trading execution and order management."""

    def __init__(self, config: Dict, risk_manager: "RiskManager"):
        """
        Initialize the trading engine.

        Args:
            config: Trading configuration dictionary
            risk_manager: Instance of RiskManager
        """
        self.config = config
        self.risk_manager = risk_manager
        self.exchange = getattr(ccxt, config["exchange"]["name"])(
            {
                "apiKey": config["exchange"]["api_key"],
                "secret": config["exchange"]["api_secret"],
                "enableRateLimit": True,
            }
        )
        self.active_orders: Dict = {}
        self.positions: Dict = {}
        self.trade_history: List = []

    async def execute_trade(
        self, symbol: str, signal: int, confidence: float, current_price: float
    ) -> Optional[str]:
        """
        Execute a trade based on the signal.

        Args:
            symbol: Trading pair symbol
            signal: Trading signal (1 for buy, -1 for sell)
            confidence: Model prediction confidence
            current_price: Current asset price

        Returns:
            Position ID if trade is executed, None otherwise
        """
        try:
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                signal, confidence, current_price
            )

            if position_size is None:
                logger.info("Trade skipped due to risk management")
                return None

            # Determine order type
            order_type = "market"  # Could be modified to use limit orders
            side = "buy" if signal == 1 else "sell"

            # Place order
            order = await self.exchange.create_order(
                symbol,
                order_type,
                side,
                position_size,
                None,  # Price is None for market orders
            )

            # Generate position ID
            position_id = str(uuid.uuid4())

            # Record position
            position = {
                "id": position_id,
                "symbol": symbol,
                "type": "long" if signal == 1 else "short",
                "size": position_size,
                "entry_price": float(order["price"]),
                "entry_time": datetime.now(),
                "stop_loss": self.risk_manager.calculate_stop_loss(
                    float(order["price"]), "long" if signal == 1 else "short"
                ),
                "take_profit": self.risk_manager.calculate_take_profit(
                    float(order["price"]), "long" if signal == 1 else "short"
                ),
                "order_id": order["id"],
            }

            self.positions[position_id] = position
            self.risk_manager.positions[position_id] = position

            logger.info(f"Trade executed: {position}")
            return position_id

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None

    async def close_position(self, position_id: str) -> bool:
        """Close a position by ID."""
        try:
            if position_id not in self.positions:
                return False

            position = self.positions[position_id]

            # Create closing order
            order = await self.exchange.create_order(
                position["symbol"],
                "market",
                "sell" if position["type"] == "long" else "buy",
                position["size"],
            )

            # Calculate P&L
            exit_price = float(order["price"])
            entry_price = position["entry_price"]
            position_type = position["type"]

            if position_type == "long":
                pnl = (exit_price - entry_price) * position["size"]
            else:
                pnl = (entry_price - exit_price) * position["size"]

            # Record trade in history
            trade_record = {
                "position_id": position_id,
                "symbol": position["symbol"],
                "type": position_type,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "size": position["size"],
                "pnl": pnl,
                "entry_time": position["entry_time"],
                "exit_time": datetime.now(),
            }

            self.trade_history.append(trade_record)
            self.risk_manager.record_trade_pnl(pnl)

            # Remove position from tracking
            del self.positions[position_id]
            del self.risk_manager.positions[position_id]

            logger.info(f"Position closed: {trade_record}")
            return True

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    async def update_positions(self, current_prices: Dict[str, float]):
        """Update and manage all open positions."""
        for position_id in list(self.positions.keys()):
            position = self.positions[position_id]
            current_price = current_prices.get(position["symbol"])

            if current_price is None:
                continue

            # Update trailing stops
            if self.config["trading"]["trailing_stop"]:
                self.risk_manager.update_trailing_stop(position_id, current_price)

            # Check if position should be closed
            if self.risk_manager.should_close_position(position_id, current_price):
                await self.close_position(position_id)

    async def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        try:
            balance = await self.exchange.fetch_balance()
            return float(balance["total"]["USDT"])
        except Exception as e:
            logger.error(f"Error fetching portfolio value: {e}")
            return 0.0

    async def start(self):
        """Start the trading engine."""
        try:
            # Initialize portfolio value
            portfolio_value = await self.get_portfolio_value()
            self.risk_manager.update_portfolio_value(portfolio_value)

            logger.info("Trading engine started successfully")

        except Exception as e:
            logger.error(f"Error starting trading engine: {e}")
            raise

    async def stop(self):
        """Stop the trading engine and close all positions."""
        try:
            # Close all open positions
            for position_id in list(self.positions.keys()):
                await self.close_position(position_id)

            # Close exchange connection
            await self.exchange.close()

            logger.info("Trading engine stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping trading engine: {e}")
            raise
