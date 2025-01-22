async def stop(self):
    import asyncio


import logging
import yaml
from pathlib import Path

from data.collectors.blockchain_collector import BlockchainCollector
from data.collectors.price_collector import PriceCollector
from models.feature_engineering import FeatureEngineer
from models.trainer import ModelTrainer
from trading.risk_manager import RiskManager
from trading.trading_engine import TradingEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/trading_bot.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot class that coordinates all components."""

    def __init__(self, config_path: str):
        """Initialize the trading bot."""
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.blockchain_collector = BlockchainCollector(
            self.config["data"]["blockchain"]["rpc_endpoint"],
            self.config["data"]["blockchain"]["metrics"],
            self.config["data"]["blockchain"]["update_interval"],
        )

        self.price_collector = PriceCollector(
            self.config["exchange"]["name"],
            self.config["exchange"]["api_key"],
            self.config["exchange"]["api_secret"],
            self.config["exchange"]["trading_pairs"],
            self.config["data"]["price"]["timeframes"],
            self.config["data"]["price"]["lookback_period"],
            self.config["data"]["price"]["update_interval"],
        )

        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(self.config["model"])
        self.risk_manager = RiskManager(self.config)
        self.trading_engine = TradingEngine(self.config, self.risk_manager)

        self.is_running = False

    async def process_data(self):
        """Process incoming data and generate trading signals."""
        try:
            # Get latest data
            blockchain_data = self.blockchain_collector.get_latest_data()
            price_data = self.price_collector.get_latest_data()

            if not blockchain_data or not price_data:
                return

            # Create features
            for symbol in self.config["exchange"]["trading_pairs"]:
                # Get price features
                df = price_data[symbol][self.config["model"]["prediction"]["horizon"]]
                price_features = self.feature_engineer.create_price_features(df)

                # Create blockchain features
                blockchain_features = self.feature_engineer.create_blockchain_features(
                    blockchain_data
                )

                # Combine features
                features = self.feature_engineer.combine_features(
                    price_features, blockchain_features
                )

                # Get latest feature vector
                latest_features = features.iloc[-1].values

                # Generate trading signal
                signal, confidence = self.model_trainer.get_trading_signal(
                    latest_features
                )

                if signal != 0:  # If we have a trade signal
                    current_price = float(df.iloc[-1]["close"])
                    await self.trading_engine.execute_trade(
                        symbol, signal, confidence, current_price
                    )

                # Update open positions
                current_prices = {
                    symbol: float(price_data[symbol][tf].iloc[-1]["close"])
                    for symbol in self.config["exchange"]["trading_pairs"]
                    for tf in self.config["data"]["price"]["timeframes"]
                }
                await self.trading_engine.update_positions(current_prices)

        except Exception as e:
            logger.error(f"Error processing data: {e}")

    async def train_model(self):
        """Periodically retrain the model with new data."""
        while self.is_running:
            try:
                # Get all historical data
                blockchain_data = self.blockchain_collector.get_latest_data()
                price_data = self.price_collector.get_latest_data()

                if blockchain_data and price_data:
                    # Prepare training data
                    all_features = []
                    all_targets = []

                    for symbol in self.config["exchange"]["trading_pairs"]:
                        df = price_data[symbol][
                            self.config["model"]["prediction"]["horizon"]
                        ]
                        price_features = self.feature_engineer.create_price_features(df)
                        blockchain_features = (
                            self.feature_engineer.create_blockchain_features(
                                blockchain_data
                            )
                        )
                        features = self.feature_engineer.combine_features(
                            price_features, blockchain_features
                        )

                        X, y = self.feature_engineer.prepare_ml_data(
                            features,
                            target_horizon=int(
                                self.config["model"]["prediction"]["horizon"][0]
                            ),
                        )

                        all_features.append(X)
                        all_targets.append(y)

                    # Combine data from all trading pairs
                    X = np.concatenate(all_features)
                    y = np.concatenate(all_targets)

                    # Train model
                    metrics = self.model_trainer.train_model(X, y)
                    logger.info(f"Model training completed with metrics: {metrics}")

                # Wait before next training
                await asyncio.sleep(3600)  # Retrain every hour

            except Exception as e:
                logger.error(f"Error training model: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    async def start(self):
        """Start the trading bot."""
        try:
            self.is_running = True

            # Start data collectors
            asyncio.create_task(self.blockchain_collector.start_collection())
            asyncio.create_task(self.price_collector.start_collection())

            # Start trading engine
            await self.trading_engine.start()

            # Start model training
            asyncio.create_task(self.train_model())

            # Main loop
            while self.is_running:
                await self.process_data()
                await asyncio.sleep(10)  # Process every 10 seconds

        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            await self.stop()

        """Stop the trading bot and cleanup resources."""
        try:
            self.is_running = False

            # Stop trading engine
            await self.trading_engine.stop()

            # Close price collector
            await self.price_collector.close()

            # Save final model state
            self.model_trainer.save_model("models/latest_model.lgb")

            logger.info("Trading bot stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
            raise

    async def run_forever(self):
        """Run the trading bot until interrupted."""
        try:
            await self.start()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            await self.stop()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            await self.stop()
            raise


async def main():
    """Main entry point for the trading bot."""
    try:
        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)

        # Initialize and run trading bot
        bot = TradingBot("config/config.yaml")
        await bot.run_forever()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
