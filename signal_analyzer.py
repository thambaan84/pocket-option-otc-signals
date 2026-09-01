import pandas as pd
import numpy as np
import ta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalAnalyzer:
    """
    Technical Analysis Signal Generator
    Uses multiple indicators for high-accuracy trading signals
    """
    
    def __init__(self, rsi_period=14, ema_fast=9, ema_slow=21, 
                 macd_fast=12, macd_slow=26, macd_signal=9):
        self.rsi_period = rsi_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.signals_history = []
        
    def prepare_data(self, ohlc_data):
        """
        Prepare OHLC data for analysis
        
        Args:
            ohlc_data (list): List of OHLC candles
            
        Returns:
            pd.DataFrame: DataFrame with OHLC data
        """
        try:
            df = pd.DataFrame(ohlc_data)
            df['close'] = pd.to_numeric(df['close'])
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['volume'] = pd.to_numeric(df.get('volume', 0))
            return df
        except Exception as e:
            logger.error(f"❌ Data preparation error: {str(e)}")
            return None
    
    def calculate_rsi(self, df):
        """Calculate RSI indicator"""
        try:
            df['RSI'] = ta.momentum.rsi(df['close'], length=self.rsi_period)
            return df
        except Exception as e:
            logger.error(f"❌ RSI calculation error: {str(e)}")
            return df
    
    def calculate_ema(self, df):
        """Calculate Fast and Slow EMAs"""
        try:
            df['EMA_FAST'] = ta.trend.ema_indicator(df['close'], length=self.ema_fast)
            df['EMA_SLOW'] = ta.trend.ema_indicator(df['close'], length=self.ema_slow)
            return df
        except Exception as e:
            logger.error(f"❌ EMA calculation error: {str(e)}")
            return df
    
    def calculate_macd(self, df):
        """Calculate MACD indicator"""
        try:
            macd = ta.trend.MACD(df['close'], 
                                window_fast=self.macd_fast,
                                window_slow=self.macd_slow,
                                window_sign=self.macd_signal)
            df['MACD'] = macd.macd()
            df['MACD_SIGNAL'] = macd.macd_signal()
            df['MACD_DIFF'] = macd.macd_diff()
            return df
        except Exception as e:
            logger.error(f"❌ MACD calculation error: {str(e)}")
            return df
    
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            bb = ta.volatility.BollingerBands(df['close'], window=period, window_dev=std_dev)
            df['BB_HIGH'] = bb.bollinger_hband()
            df['BB_MID'] = bb.bollinger_mavg()
            df['BB_LOW'] = bb.bollinger_lband()
            return df
        except Exception as e:
            logger.error(f"❌ Bollinger Bands error: {str(e)}")
            return df
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range (volatility)"""
        try:
            df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], length=period)
            return df
        except Exception as e:
            logger.error(f"❌ ATR calculation error: {str(e)}")
            return df
    
    def calculate_volume_signal(self, df):
        """Calculate volume-based signals"""
        try:
            df['VOLUME_SMA'] = df['volume'].rolling(window=20).mean()
            df['VOLUME_RATIO'] = df['volume'] / df['VOLUME_SMA']
            return df
        except Exception as e:
            logger.error(f"❌ Volume calculation error: {str(e)}")
            return df
    
    def generate_signals(self, df, rsi_overbought=70, rsi_oversold=30, 
                        min_signal_strength=0.75):
        """
        Generate buy/sell signals based on multiple indicators
        
        Args:
            df (pd.DataFrame): DataFrame with calculated indicators
            rsi_overbought (int): RSI overbought level
            rsi_oversold (int): RSI oversold level
            min_signal_strength (float): Minimum signal strength (0-1)
            
        Returns:
            dict: Signal data with direction and strength
        """
        try:
            current = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else None
            
            # Initialize signal scores
            buy_score = 0
            sell_score = 0
            signal_components = {}
            
            # ========== RSI SIGNALS ==========
            if current['RSI'] < rsi_oversold:
                buy_score += 0.25
                signal_components['RSI'] = 'OVERSOLD (BUY)'
            elif current['RSI'] > rsi_overbought:
                sell_score += 0.25
                signal_components['RSI'] = 'OVERBOUGHT (SELL)'
            else:
                signal_components['RSI'] = 'NEUTRAL'
            
            # ========== EMA CROSSOVER SIGNALS ==========
            if previous is not None:
                # Fast EMA above Slow EMA = Bullish
                if current['EMA_FAST'] > current['EMA_SLOW'] and \
                   previous['EMA_FAST'] <= previous['EMA_SLOW']:
                    buy_score += 0.30
                    signal_components['EMA'] = 'BULLISH CROSSOVER'
                
                # Fast EMA below Slow EMA = Bearish
                elif current['EMA_FAST'] < current['EMA_SLOW'] and \
                     previous['EMA_FAST'] >= previous['EMA_SLOW']:
                    sell_score += 0.30
                    signal_components['EMA'] = 'BEARISH CROSSOVER'
                else:
                    signal_components['EMA'] = 'NO CROSSOVER'
            
            # ========== MACD SIGNALS ==========
            if previous is not None:
                # MACD bullish crossover
                if current['MACD'] > current['MACD_SIGNAL'] and \
                   previous['MACD'] <= previous['MACD_SIGNAL']:
                    buy_score += 0.20
                    signal_components['MACD'] = 'BULLISH CROSSOVER'
                
                # MACD bearish crossover
                elif current['MACD'] < current['MACD_SIGNAL'] and \
                     previous['MACD'] >= previous['MACD_SIGNAL']:
                    sell_score += 0.20
                    signal_components['MACD'] = 'BEARISH CROSSOVER'
                else:
                    signal_components['MACD'] = 'NO CROSSOVER'
            
            # ========== BOLLINGER BANDS SIGNALS ==========
            if current['close'] < current['BB_LOW']:
                buy_score += 0.15
                signal_components['BB'] = 'BELOW LOWER BAND (BUY)'
            elif current['close'] > current['BB_HIGH']:
                sell_score += 0.15
                signal_components['BB'] = 'ABOVE UPPER BAND (SELL)'
            else:
                signal_components['BB'] = 'WITHIN BANDS'
            
            # ========== VOLUME SIGNALS ==========
            if current['VOLUME_RATIO'] > 1.5:
                if current['close'] > previous['close']:
                    buy_score += 0.10
                    signal_components['VOLUME'] = 'HIGH VOLUME UP'
                else:
                    sell_score += 0.10
                    signal_components['VOLUME'] = 'HIGH VOLUME DOWN'
            else:
                signal_components['VOLUME'] = 'NORMAL VOLUME'
            
            # ========== DETERMINE FINAL SIGNAL ==========
            total_score = buy_score + sell_score
            
            if total_score == 0:
                return {
                    'signal': 'NEUTRAL',
                    'direction': None,
                    'strength': 0,
                    'components': signal_components,
                    'buy_score': 0,
                    'sell_score': 0
                }
            
            buy_strength = buy_score / total_score if total_score > 0 else 0
            sell_strength = sell_score / total_score if total_score > 0 else 0
            
            if buy_strength > sell_strength:
                signal = 'BUY' if buy_strength >= min_signal_strength else 'WEAK_BUY'
                direction = 'CALL'
                strength = buy_strength
            else:
                signal = 'SELL' if sell_strength >= min_signal_strength else 'WEAK_SELL'
                direction = 'PUT'
                strength = sell_strength
            
            return {
                'signal': signal,
                'direction': direction,
                'strength': round(strength, 3),
                'components': signal_components,
                'buy_score': round(buy_score, 3),
                'sell_score': round(sell_score, 3),
                'current_price': round(current['close'], 4),
                'rsi': round(current['RSI'], 2),
                'ema_fast': round(current['EMA_FAST'], 4),
                'ema_slow': round(current['EMA_SLOW'], 4)
            }
            
        except Exception as e:
            logger.error(f"❌ Signal generation error: {str(e)}")
            return {
                'signal': 'ERROR',
                'direction': None,
                'strength': 0,
                'components': {'error': str(e)}
            }
    
    def analyze(self, ohlc_data, asset_name="ASSET", **kwargs):
        """
        Complete analysis pipeline
        
        Args:
            ohlc_data (list): OHLC data
            asset_name (str): Asset identifier
            
        Returns:
            dict: Complete signal analysis
        """
        try:
            # Prepare data
            df = self.prepare_data(ohlc_data)
            if df is None or len(df) < self.ema_slow:
                return {'signal': 'INSUFFICIENT_DATA', 'asset': asset_name}
            
            # Calculate all indicators
            df = self.calculate_rsi(df)
            df = self.calculate_ema(df)
            df = self.calculate_macd(df)
            df = self.calculate_bollinger_bands(df)
            df = self.calculate_atr(df)
            df = self.calculate_volume_signal(df)
            
            # Generate signals
            signal_result = self.generate_signals(df, **kwargs)
            signal_result['asset'] = asset_name
            signal_result['timestamp'] = pd.Timestamp.now().isoformat()
            
            # Store in history
            self.signals_history.append(signal_result)
            
            return signal_result
            
        except Exception as e:
            logger.error(f"❌ Analysis error for {asset_name}: {str(e)}")
            return {'signal': 'ERROR', 'asset': asset_name, 'error': str(e)}
