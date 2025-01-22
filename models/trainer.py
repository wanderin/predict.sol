import logging
from typing import Dict, Tuple, Optional

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains and maintains the prediction model."""
    
    def __init__(self, config: Dict):
        """
        Initialize the model trainer.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.feature_importance = None
        
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the model on provided data.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            # Split data into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config['training']['validation_size'],
                shuffle=False  # Maintain temporal order
            )
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)
            
            # Set model parameters
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'learning_rate': self.config['training']['learning_rate'],
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            
            # Train model
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.config['training']['epochs'],
                valid_sets=[train_data, val_data],
                callbacks=[lgb.early_stopping(50)]
            )
            
            # Get feature importance
            self.feature_importance = dict(zip(
                range(X.shape[1]),
                self.model.feature_importance()
            ))
            
            # Make predictions on validation set
            val_preds = self.predict(X_val)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, val_preds > self.config['prediction']['threshold']),
                'precision': precision_score(y_val, val_preds > self.config['prediction']['threshold']),
                'recall': recall_score(y_val, val_preds > self.config['prediction']['threshold']),
                'f1': f1_score(y_val, val_preds > self.config['prediction']['threshold'])
            }
            
            logger.info(f"Model training completed. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        return self.model.predict(X)
        
    def get_trading_signal(self, X: np.ndarray) -> Tuple[int, float]:
        """
        Generate trading signal from model prediction.
        
        Args:
            X: Feature matrix for a single time point
            
        Returns:
            Tuple of (signal, confidence):
                signal: 1 for buy, -1 for sell, 0 for hold
                confidence: Prediction probability
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        prob = self.predict(X)[0]
        threshold = self.config['prediction']['threshold']
        
        if prob > threshold:
            return 1, prob
        elif prob < 1 - threshold:
            return -1, 1 - prob
        else:
            return 0, prob
            
    def save_model(self, path: str):
        """Save the trained model to disk."""
        if self.model is not None:
            self.model.save_model(path)
            logger.info(f"Model saved to {path}")
            
    def load_model(self, path: str):
        """Load a trained model from disk."""
        try:
            self.model = lgb.Booster(model_file=path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise