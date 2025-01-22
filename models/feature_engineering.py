import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Creates features from raw blockchain and price data."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators from price data."""
        features = df.copy()

        # Price momentum features
        for period in [5, 10, 20, 50]:
            features[f"returns_{period}"] = features["close"].pct_change(period)
            features[f"ma_{period}"] = features["close"].rolling(window=period).mean()
            features[f"std_{period}"] = features["close"].rolling(window=period).std()

        # Volume features
        features["volume_ma_ratio"] = (
            features["volume"] / features["volume"].rolling(window=20).mean()
        )

        # Volatility features
        features["atr"] = self.calculate_atr(df)
        features["volatility"] = features["returns_20"].rolling(window=20).std()

        # Price patterns
        features["higher_highs"] = (
            features["high"] > features["high"].shift(1)
        ).astype(int)
        features["lower_lows"] = (features["low"] < features["low"].shift(1)).astype(
            int
        )

        return features

    def create_blockchain_features(self, blockchain_data: Dict) -> pd.DataFrame:
        """Create features from blockchain data."""
        features = pd.DataFrame()

        if blockchain_data and "metrics" in blockchain_data:
            metrics = blockchain_data["metrics"]

            # Transaction features
            if "avg_tps" in metrics:
                features["tps"] = metrics["avg_tps"]

            # Stake features
            if "active_stake" in metrics:
                features["stake_ratio"] = metrics["active_stake"] / (
                    metrics["active_stake"] + metrics["inactive_stake"]
                )

            # DeFi features
            if "tvl" in metrics:
                features["tvl"] = metrics["tvl"]
                features["tvl_change"] = features["tvl"].pct_change()

        return features

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        return true_range.rolling(period).mean()

    def combine_features(
        self, price_features: pd.DataFrame, blockchain_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine and normalize all features."""
        # Align timestamps and combine features
        combined = pd.concat([price_features, blockchain_features], axis=1)

        # Handle missing values
        combined = combined.fillna(method="ffill").fillna(method="bfill")

        # Scale features
        feature_cols = [
            col
            for col in combined.columns
            if col not in ["timestamp", "open", "high", "low", "close", "volume"]
        ]
        self.feature_columns = feature_cols

        combined[feature_cols] = self.scaler.fit_transform(combined[feature_cols])

        return combined

    def prepare_ml_data(
        self, data: pd.DataFrame, target_horizon: int = 12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning model.

        Args:
            data: DataFrame containing features
            target_horizon: Number of periods ahead to predict

        Returns:
            X: Feature matrix
            y: Target labels (1 for price increase, 0 for decrease)
        """
        # Create target variable (future returns)
        future_returns = data["close"].shift(-target_horizon).pct_change(target_horizon)

        # Create binary labels
        y = (future_returns > 0).astype(int)

        # Create feature matrix
        X = data[self.feature_columns].values

        # Remove rows with NaN values
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        return X, y
