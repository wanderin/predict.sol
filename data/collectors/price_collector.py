import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import ccxt.async_support as ccxt
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class PriceCollector:
    """Collects price data from cryptocurrency exchanges."""
    
    def __init__(self, 
                 exchange_name: str,
                 api_key: str,
                 api_secret: str,
                 trading_pairs: List[str],
                 timeframes: List[str],
                 lookback_period: int,
                 update_interval: int):
        """
        Initialize the price collector.
        
        Args:
            exchange_name: Name of the exchange (e.g., 'binance')
            api_key: Exchange API key
            api_secret: Exchange API secret
            trading_pairs: List of trading pairs to monitor
            timeframes: List of timeframes to collect
            lookback_period: Number of historical candles to maintain
            update_interval: Data collection interval in seconds
        """
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        self.trading_pairs = trading_pairs
        self.timeframes = timeframes
        self.lookback_period = lookback_period
        self.update_interval = update_interval
        self.price_data: Dict = {}
        
    async def fetch_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch OHLCV data for a specific trading pair and timeframe."""
        try:
            # Fetch historical candles
            candles = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                limit=self.lookback_period
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate additional metrics
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['upper_band'] = df['close'].rolling(window=20).mean() + (df['close'].rolling(window=20).std() * 2)
            df['lower_band'] = df['close'].rolling(window=20).mean() - (df['close'].rolling(window=20).std() * 2)
            df['rsi'] = self.calculate_rsi(df['close'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol} ({timeframe}): {e}")
            return pd.DataFrame()
            
    @staticmethod
    def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        
        gains = delta.copy()
        gains[gains < 0] = 0
        
        losses = delta.copy()
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = gains.rolling(window=periods).mean()
        avg_loss = losses.rolling(window=periods).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    async def collect_data(self) -> Dict:
        """Collect price data for all trading pairs and timeframes."""
        tasks = []
        
        for symbol in self.trading_pairs:
            for timeframe in self.timeframes:
                tasks.append(self.fetch_ohlcv(symbol, timeframe))
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update price data dictionary
        for i, result in enumerate(results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                symbol = self.trading_pairs[i // len(self.timeframes)]
                timeframe = self.timeframes[i % len(self.timeframes)]
                
                if symbol not in self.price_data:
                    self.price_data[symbol] = {}
                    
                self.price_data[symbol][timeframe] = result
                
        return self.price_data
        
    async def start_collection(self):
        """Start continuous price data collection."""
        while True:
            try:
                await self.collect_data()
                logger.info("Price data collected successfully")
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in price data collection: {e}")
                await asyncio.sleep(self.update_interval)
                

    Price Data Collector

    def get_latest_data(self, symbol: Optional[str] = None, 
                        timeframe: Optional[str] = None) -> Dict:
            """
            Get the latest collected price data.
            
            Args:
                symbol: Specific trading pair to retrieve (optional)
                timeframe: Specific timeframe to retrieve (optional)
                
            Returns:
                Dictionary containing the requested price data
            """
            if symbol and timeframe:
                return self.price_data.get(symbol, {}).get(timeframe, pd.DataFrame())
            elif symbol:
                return self.price_data.get(symbol, {})
            else:
                return self.price_data
                
        async def close(self):
            """Close the exchange connection."""
            await self.exchange.close()