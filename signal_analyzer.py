"""
Technical-analysis signal generator for OTC / forex-style OHLC candles.

Analysis only: it takes candles in and returns a signal dict. It does not
fetch data, send alerts, place trades or manage risk.
"""
import logging
from collections import deque

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Indicator weights. A full set of agreeing indicators sums to 1.0.
W_RSI = 0.25
W_EMA = 0.30
W_MACD = 0.20
W_BB = 0.15
W_VOLUME = 0.10

# ---------------------------------------------------------------------------
# Indicator helpers (pure pandas/numpy, so there is no dependency on the `ta`
# package and no API-version mismatch)
# ---------------------------------------------------------------------------


def _ema(series, span):
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close, period):
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0)  # no losses -> 100
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)  # flat -> 50
    return rsi


def _atr(high, low, close, period):
    """Wilder's Average True Range."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


class SignalAnalyzer:
    """
    Multi-indicator signal generator.

    Scoring model
    -------------
    Each indicator adds its weight to a buy score or a sell score (0..1).
    A BUY/SELL signal requires ALL of:
      * winning score >= min_signal_strength (absolute, not a ratio)
      * at least `min_agreeing` indicators voting for that side
      * winning score leads the other side by at least `min_margin`
    Anything weaker is WEAK_BUY / WEAK_SELL (direction is None, the leaning is
    reported in `bias`). Equal scores give NEUTRAL.
    """

    def __init__(self, rsi_period=14, ema_fast=9, ema_slow=21,
                 macd_fast=12, macd_slow=26, macd_signal=9,
                 bb_period=20, bb_std=2, atr_period=14, volume_period=20,
                 history_size=500):
        self.rsi_period = rsi_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.volume_period = volume_period
        self.signals_history = deque(maxlen=history_size)

    @property
    def min_candles(self):
        """Candles needed so every indicator is valid on the last two rows."""
        return max(
            self.ema_slow,
            self.macd_slow + self.macd_signal,
            self.bb_period,
            self.rsi_period + 1,
            self.atr_period + 1,
        ) + 1

    # ------------------------------------------------------------------ data
    def prepare_data(self, ohlc_data):
        """Return a clean, time-ordered DataFrame, or None if the data is unusable."""
        try:
            df = pd.DataFrame(ohlc_data)
            if df.empty:
                return None
            required = ['open', 'high', 'low', 'close']
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.error("Missing OHLC columns: %s", missing)
                return None
            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            else:
                df['volume'] = 0.0
            for tcol in ('time', 'timestamp'):
                if tcol in df.columns:
                    df = df.drop_duplicates(subset=tcol, keep='last').sort_values(tcol)
                    break
            df = df.dropna(subset=required).reset_index(drop=True)
            return df if len(df) else None
        except Exception:
            logger.exception("Data preparation error")
            return None

    # ------------------------------------------------------------ indicators
    def calculate_rsi(self, df):
        df['RSI'] = _rsi(df['close'], self.rsi_period)
        return df

    def calculate_ema(self, df):
        df['EMA_FAST'] = _ema(df['close'], self.ema_fast)
        df['EMA_SLOW'] = _ema(df['close'], self.ema_slow)
        return df

    def calculate_macd(self, df):
        macd = _ema(df['close'], self.macd_fast) - _ema(df['close'], self.macd_slow)
        signal = macd.ewm(span=self.macd_signal, adjust=False,
                          min_periods=self.macd_signal).mean()
        df['MACD'] = macd
        df['MACD_SIGNAL'] = signal
        df['MACD_DIFF'] = macd - signal
        return df

    def calculate_bollinger_bands(self, df):
        mid = df['close'].rolling(self.bb_period).mean()
        std = df['close'].rolling(self.bb_period).std(ddof=0)
        df['BB_MID'] = mid
        df['BB_HIGH'] = mid + self.bb_std * std
        df['BB_LOW'] = mid - self.bb_std * std
        return df

    def calculate_atr(self, df):
        df['ATR'] = _atr(df['high'], df['low'], df['close'], self.atr_period)
        return df

    def calculate_volume_signal(self, df):
        """Volume ratio. NaN when the feed has no real volume (typical for OTC)."""
        if df['volume'].nunique() <= 1:
            df['VOLUME_RATIO'] = np.nan
            return df
        sma = df['volume'].rolling(self.volume_period).mean()
        df['VOLUME_RATIO'] = (df['volume'] / sma.replace(0, np.nan))
        return df

    # --------------------------------------------------------------- signals
    @staticmethod
    def _result(signal='NEUTRAL', direction=None, bias=None, strength=0.0,
                components=None, buy_score=0.0, sell_score=0.0, **extra):
        """Every return path uses this, so the output schema is always the same."""
        result = {
            'signal': signal,
            'direction': direction,
            'bias': bias,
            'strength': round(float(strength), 3),
            'buy_score': round(float(buy_score), 3),
            'sell_score': round(float(sell_score), 3),
            'components': components or {},
            'current_price': None,
            'rsi': None,
            'ema_fast': None,
            'ema_slow': None,
            'atr': None,
        }
        result.update(extra)
        return result

    def generate_signals(self, df, rsi_overbought=70, rsi_oversold=30,
                         min_signal_strength=0.5, min_agreeing=2,
                         min_margin=0.2, min_atr_pct=0.0):
        """
        Score the last two candles of an indicator-enriched DataFrame.

        Args:
            min_signal_strength: minimum absolute score (0-1) of the winning side.
            min_agreeing: minimum number of indicators voting for the winner.
            min_margin: minimum score lead over the opposite side.
            min_atr_pct: skip when ATR/close is below this (0 disables the filter).
        """
        cols = ['RSI', 'EMA_FAST', 'EMA_SLOW', 'MACD', 'MACD_SIGNAL',
                'BB_LOW', 'BB_HIGH', 'ATR']
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return self._result('ERROR', components={'error': f'missing columns: {missing}'})
        if len(df) < 2:
            return self._result('INSUFFICIENT_DATA', components={'reason': 'need 2+ candles'})

        cur, prev = df.iloc[-1], df.iloc[-2]
        if cur[cols].isna().any() or prev[cols].isna().any():
            return self._result('INSUFFICIENT_DATA',
                                components={'reason': 'indicators not warmed up (NaN)'})

        scores = {'buy': 0.0, 'sell': 0.0}
        votes = {'buy': 0, 'sell': 0}
        comp = {}

        def vote(side, weight):
            scores[side] += weight
            votes[side] += 1

        # RSI (mean reversion)
        if cur['RSI'] < rsi_oversold:
            vote('buy', W_RSI)
            comp['RSI'] = 'OVERSOLD (BUY)'
        elif cur['RSI'] > rsi_overbought:
            vote('sell', W_RSI)
            comp['RSI'] = 'OVERBOUGHT (SELL)'
        else:
            comp['RSI'] = 'NEUTRAL'

        # EMA: votes only on a fresh crossover (no vote for an ongoing trend)
        if cur['EMA_FAST'] > cur['EMA_SLOW'] and prev['EMA_FAST'] <= prev['EMA_SLOW']:
            vote('buy', W_EMA)
            comp['EMA'] = 'BULLISH CROSSOVER'
        elif cur['EMA_FAST'] < cur['EMA_SLOW'] and prev['EMA_FAST'] >= prev['EMA_SLOW']:
            vote('sell', W_EMA)
            comp['EMA'] = 'BEARISH CROSSOVER'
        else:
            comp['EMA'] = 'NO CROSSOVER'

        # MACD: votes only on a fresh signal-line crossover
        if cur['MACD'] > cur['MACD_SIGNAL'] and prev['MACD'] <= prev['MACD_SIGNAL']:
            vote('buy', W_MACD)
            comp['MACD'] = 'BULLISH CROSSOVER'
        elif cur['MACD'] < cur['MACD_SIGNAL'] and prev['MACD'] >= prev['MACD_SIGNAL']:
            vote('sell', W_MACD)
            comp['MACD'] = 'BEARISH CROSSOVER'
        else:
            comp['MACD'] = 'NO CROSSOVER'

        # Bollinger Bands (mean reversion)
        if cur['close'] < cur['BB_LOW']:
            vote('buy', W_BB)
            comp['BB'] = 'BELOW LOWER BAND (BUY)'
        elif cur['close'] > cur['BB_HIGH']:
            vote('sell', W_BB)
            comp['BB'] = 'ABOVE UPPER BAND (SELL)'
        else:
            comp['BB'] = 'WITHIN BANDS'

        # Volume (only when the feed actually has volume)
        vol_ratio = cur['VOLUME_RATIO'] if 'VOLUME_RATIO' in df.columns else np.nan
        if pd.isna(vol_ratio):
            comp['VOLUME'] = 'N/A (no volume data)'
        elif vol_ratio > 1.5 and cur['close'] != prev['close']:
            if cur['close'] > prev['close']:
                vote('buy', W_VOLUME)
                comp['VOLUME'] = 'HIGH VOLUME UP'
            else:
                vote('sell', W_VOLUME)
                comp['VOLUME'] = 'HIGH VOLUME DOWN'
        else:
            comp['VOLUME'] = 'NORMAL VOLUME'

        buy_score = round(scores['buy'], 6)
        sell_score = round(scores['sell'], 6)
        info = dict(
            current_price=round(float(cur['close']), 5),
            rsi=round(float(cur['RSI']), 2),
            ema_fast=round(float(cur['EMA_FAST']), 5),
            ema_slow=round(float(cur['EMA_SLOW']), 5),
            atr=round(float(cur['ATR']), 6),
        )

        if min_atr_pct > 0 and cur['close'] > 0 and cur['ATR'] / cur['close'] < min_atr_pct:
            comp['FILTER'] = 'LOW VOLATILITY'
            return self._result('NEUTRAL', components=comp, buy_score=buy_score,
                                sell_score=sell_score, **info)

        if buy_score == sell_score:  # includes 0 == 0
            return self._result('NEUTRAL', components=comp, buy_score=buy_score,
                                sell_score=sell_score, **info)

        side = 'buy' if buy_score > sell_score else 'sell'
        win, lose = (buy_score, sell_score) if side == 'buy' else (sell_score, buy_score)
        qualifies = (win >= min_signal_strength
                     and votes[side] >= min_agreeing
                     and round(win - lose, 6) >= min_margin)

        label = side.upper()
        bias = 'CALL' if side == 'buy' else 'PUT'
        return self._result(
            signal=label if qualifies else f'WEAK_{label}',
            direction=bias if qualifies else None,
            bias=bias,
            strength=win,
            components=comp,
            buy_score=buy_score,
            sell_score=sell_score,
            **info,
        )

    # -------------------------------------------------------------- pipeline
    def analyze(self, ohlc_data, asset_name="ASSET", drop_last_candle=False, **kwargs):
        """
        Full pipeline: candles in, signal dict out.

        Args:
            ohlc_data: list of dicts with open/high/low/close (+ optional
                volume and time/timestamp), oldest first or sortable by time.
            drop_last_candle: set True if the last candle is still forming, so
                the signal is computed on closed candles only (avoids repainting).
            **kwargs: forwarded to generate_signals().
        """
        try:
            df = self.prepare_data(ohlc_data)
            if df is None:
                result = self._result('ERROR', components={'error': 'invalid OHLC data'})
            else:
                if drop_last_candle:
                    df = df.iloc[:-1].reset_index(drop=True)
                if len(df) < self.min_candles:
                    result = self._result(
                        'INSUFFICIENT_DATA',
                        components={'have': len(df), 'need': self.min_candles})
                else:
                    df = self.calculate_rsi(df)
                    df = self.calculate_ema(df)
                    df = self.calculate_macd(df)
                    df = self.calculate_bollinger_bands(df)
                    df = self.calculate_atr(df)
                    df = self.calculate_volume_signal(df)
                    result = self.generate_signals(df, **kwargs)
        except Exception as e:
            logger.exception("Analysis error for %s", asset_name)
            result = self._result('ERROR', components={'error': str(e)})

        result['asset'] = asset_name
        result['timestamp'] = pd.Timestamp.now().isoformat()
        self.signals_history.append(result)
        return result
