import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signal_analyzer import SignalAnalyzer, _rsi, _ema  # noqa: E402

SCHEMA = {'signal', 'direction', 'bias', 'strength', 'buy_score', 'sell_score',
          'components', 'current_price', 'rsi', 'ema_fast', 'ema_slow', 'atr',
          'asset', 'timestamp'}


def make_candles(n=120, seed=1, drift=0.0, volume=False):
    rng = np.random.default_rng(seed)
    close = 1.1 + np.cumsum(rng.normal(drift, 0.0005, n))
    candles = []
    for i, c in enumerate(close):
        o = close[i - 1] if i else c
        candle = {'time': i, 'open': o, 'high': max(o, c) + 0.0002,
                  'low': min(o, c) - 0.0002, 'close': c}
        if volume:
            candle['volume'] = float(rng.integers(100, 1000))
        candles.append(candle)
    return candles


def make_df(cur=None, prev=None):
    base = dict(close=1.0, RSI=50.0, EMA_FAST=1.0, EMA_SLOW=1.0, MACD=0.0,
                MACD_SIGNAL=0.0, BB_LOW=0.9, BB_HIGH=1.1, ATR=0.01,
                VOLUME_RATIO=np.nan)
    return pd.DataFrame([{**base, **(prev or {})}, {**base, **(cur or {})}])


def test_indicators_sane():
    up = pd.Series(np.arange(1.0, 60.0))
    assert _rsi(up, 14).iloc[-1] == 100.0
    assert _rsi(pd.Series([1.0] * 40), 14).iloc[-1] == 50.0
    assert abs(_ema(pd.Series([2.0] * 50), 9).iloc[-1] - 2.0) < 1e-12
    r = _rsi(pd.Series(np.random.default_rng(0).normal(size=200).cumsum()), 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_full_pipeline_schema():
    a = SignalAnalyzer()
    for vol in (False, True):
        res = a.analyze(make_candles(volume=vol), 'EURUSD_otc')
        assert res['signal'] != 'ERROR', res
        assert SCHEMA <= set(res), SCHEMA - set(res)
        assert res['rsi'] is not None and 0 <= res['rsi'] <= 100


def test_insufficient_data_and_bad_input():
    a = SignalAnalyzer()
    assert a.analyze(make_candles(n=10))['signal'] == 'INSUFFICIENT_DATA'
    assert a.analyze([])['signal'] == 'ERROR'
    assert a.analyze([{'close': 1}])['signal'] == 'ERROR'


def test_drop_last_candle():
    a = SignalAnalyzer()
    n = a.min_candles
    assert a.analyze(make_candles(n=n))['signal'] != 'INSUFFICIENT_DATA'
    assert a.analyze(make_candles(n=n), drop_last_candle=True)['signal'] == 'INSUFFICIENT_DATA'


def test_single_indicator_is_not_a_full_signal():
    # Old code: lone RSI -> strength 1.0 -> BUY. Now it must stay weak.
    res = SignalAnalyzer().generate_signals(make_df(cur={'RSI': 20.0}))
    assert res['signal'] == 'WEAK_BUY'
    assert res['direction'] is None and res['bias'] == 'CALL'


def test_tie_is_neutral():
    # sell: EMA trend 0.15 + MACD trend 0.10 = 0.25 ; buy: RSI oversold 0.25
    cur = {'RSI': 20.0, 'EMA_FAST': 0.99, 'EMA_SLOW': 1.0, 'MACD': -0.1, 'MACD_SIGNAL': 0.0}
    prev = {'EMA_FAST': 0.99, 'EMA_SLOW': 1.0, 'MACD': -0.1, 'MACD_SIGNAL': 0.0}
    res = SignalAnalyzer().generate_signals(make_df(cur, prev))
    assert res['signal'] == 'NEUTRAL' and res['direction'] is None


def test_confirmed_buy_and_sell():
    cur = {'RSI': 20.0, 'EMA_FAST': 1.01, 'EMA_SLOW': 1.0, 'MACD': 0.1, 'MACD_SIGNAL': 0.0}
    prev = {'EMA_FAST': 0.99, 'EMA_SLOW': 1.0, 'MACD': -0.1, 'MACD_SIGNAL': 0.0}
    res = SignalAnalyzer().generate_signals(make_df(cur, prev))
    assert res['signal'] == 'BUY' and res['direction'] == 'CALL'

    cur = {'RSI': 80.0, 'EMA_FAST': 0.99, 'EMA_SLOW': 1.0, 'MACD': -0.1, 'MACD_SIGNAL': 0.0}
    prev = {'EMA_FAST': 1.01, 'EMA_SLOW': 1.0, 'MACD': 0.1, 'MACD_SIGNAL': 0.0}
    res = SignalAnalyzer().generate_signals(make_df(cur, prev))
    assert res['signal'] == 'SELL' and res['direction'] == 'PUT'


def test_nan_row_not_scored():
    res = SignalAnalyzer().generate_signals(make_df(cur={'RSI': np.nan}))
    assert res['signal'] == 'INSUFFICIENT_DATA'


def test_volatility_filter():
    cur = {'RSI': 20.0, 'EMA_FAST': 1.01, 'EMA_SLOW': 1.0, 'MACD': 0.1, 'MACD_SIGNAL': 0.0}
    prev = {'EMA_FAST': 0.99, 'EMA_SLOW': 1.0, 'MACD': -0.1, 'MACD_SIGNAL': 0.0}
    res = SignalAnalyzer().generate_signals(make_df(cur, prev), min_atr_pct=0.5)
    assert res['signal'] == 'NEUTRAL' and res['components']['FILTER'] == 'LOW VOLATILITY'


def test_history_is_capped():
    a = SignalAnalyzer(history_size=5)
    for _ in range(12):
        a.analyze(make_candles(n=10))
    assert len(a.signals_history) == 5


if __name__ == '__main__':
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS', name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print('FAIL', name, repr(e))
    sys.exit(1 if failed else 0)
