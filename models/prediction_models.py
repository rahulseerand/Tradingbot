import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
import logging

logger = logging.getLogger()

class LSTMPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 60  # Use 60 previous candles for prediction
        self.models_directory = "ai_models"
        self.features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)
    
    def prepare_data(self, df):
        """Prepare data for LSTM model"""
        # Ensure all required features exist
        for feature in self.features:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
                
        # Scale the data
        scaled_data = self.scaler.fit_transform(df[self.features].values)
        
        # Create sequences
        X = []
        y = []
        
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            # Target is the close price (index 3 in our feature set)
            y.append(scaled_data[i, 3])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, df, symbol, timeframe, epochs=50, batch_size=32):
        """Train the model"""
        try:
            logger.info(f"Preparing data for training {symbol} {timeframe} model")
            X, y = self.prepare_data(df)
            
            if len(X) < 100:
                logger.error(f"Not enough data for training. Need at least 100 samples, got {len(X)}")
                return None
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Build model
            logger.info(f"Building LSTM model for {symbol} {timeframe}")
            self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Train model
            logger.info(f"Training model with {len(X_train)} samples, {epochs} epochs")
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                           validation_data=(X_test, y_test), verbose=1)
            
            # Save model and scaler
            model_path = f"{self.models_directory}/{symbol}_{timeframe}_lstm_model.h5"
            scaler_path = f"{self.models_directory}/{symbol}_{timeframe}_scaler.pkl"
            
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Model saved to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def load_model(self, symbol, timeframe):
        """Load a trained model"""
        try:
            model_path = f"{self.models_directory}/{symbol}_{timeframe}_lstm_model.h5"
            scaler_path = f"{self.models_directory}/{symbol}_{timeframe}_scaler.pkl"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                logger.info(f"Loading model from {model_path}")
                self.model = tf.keras.models.load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                return True
            else:
                logger.warning(f"Model files not found for {symbol} {timeframe}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_next_price(self, recent_data):
        """Predict the next price using recent data"""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
                
            # Check if we have enough data
            if len(recent_data) < self.sequence_length:
                raise ValueError(f"Not enough data for prediction. Need {self.sequence_length} rows.")
            
            # Scale the input data
            scaled_data = self.scaler.transform(recent_data[self.features].values)
            
            # Create a sequence
            X = np.array([scaled_data[-self.sequence_length:]])
            
            # Make prediction
            scaled_prediction = self.model.predict(X)
            
            # Inverse transform to get the actual price
            # Create a dummy array to inverse transform
            dummy = np.zeros((1, len(self.features)))
            dummy[0, 3] = scaled_prediction[0, 0]  # 3 is the index of 'close' in our feature set
            
            # Inverse transform
            inverse_prediction = self.scaler.inverse_transform(dummy)
            
            return inverse_prediction[0, 3]  # Return the predicted close price
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None