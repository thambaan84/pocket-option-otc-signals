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
    # buy : RSI oversold 0.25 + close below lower band 0.15            = 0.40
    # sell: EMA bearish crossover 0.30 + high-volume down candle 0.10  = 0.40
    cur = {'close': 0.8, 'RSI': 20.0, 'BB_LOW': 0.9, 'VOLUME_RATIO': 2.0,
           'EMA_FAST': 0.99, 'EMA_SLOW': 1.0}
    prev = {'close': 1.0, 'EMA_FAST': 1.01, 'EMA_SLOW': 1.0}
    res = SignalAnalyzer().generate_signals(make_df(cur, prev))
    assert res['buy_score'] == res['sell_score'] == 0.4, res
    assert res['signal'] == 'NEUTRAL' and res['direction'] is None


def test_existing_bullish_trend_gives_zero_votes():
    # fast stays above slow and MACD stays above its signal on both candles
    cur = {'EMA_FAST': 1.01, 'EMA_SLOW': 1.0, 'MACD': 0.1, 'MACD_SIGNAL': 0.0}
    prev = {'EMA_FAST': 1.02, 'EMA_SLOW': 1.0, 'MACD': 0.2, 'MACD_SIGNAL': 0.0}
    res = SignalAnalyzer().generate_signals(make_df(cur, prev))
    assert res['buy_score'] == 0 and res['sell_score'] == 0, res
    assert res['components']['EMA'] == 'NO CROSSOVER'
    assert res['components']['MACD'] == 'NO CROSSOVER'
    assert res['signal'] == 'NEUTRAL' and res['direction'] is None


def test_existing_bearish_trend_gives_zero_votes():
    cur = {'EMA_FAST': 0.99, 'EMA_SLOW': 1.0, 'MACD': -0.1, 'MACD_SIGNAL': 0.0}
    prev = {'EMA_FAST': 0.98, 'EMA_SLOW': 1.0, 'MACD': -0.2, 'MACD_SIGNAL': 0.0}
    res = SignalAnalyzer().generate_signals(make_df(cur, prev))
    assert res['buy_score'] == 0 and res['sell_score'] == 0, res
    assert res['components']['EMA'] == 'NO CROSSOVER'
    assert res['components']['MACD'] == 'NO CROSSOVER'
    assert res['signal'] == 'NEUTRAL' and res['direction'] is None


def test_fresh_ema_crossovers_vote_full_weight():
    bull = SignalAnalyzer().generate_signals(make_df(
        {'EMA_FAST': 1.01, 'EMA_SLOW': 1.0}, {'EMA_FAST': 0.99, 'EMA_SLOW': 1.0}))
    assert bull['components']['EMA'] == 'BULLISH CROSSOVER'
    assert bull['buy_score'] == 0.3 and bull['sell_score'] == 0
    assert bull['signal'] == 'WEAK_BUY' and bull['direction'] is None  # 1 vote only

    bear = SignalAnalyzer().generate_signals(make_df(
        {'EMA_FAST': 0.99, 'EMA_SLOW': 1.0}, {'EMA_FAST': 1.01, 'EMA_SLOW': 1.0}))
    assert bear['components']['EMA'] == 'BEARISH CROSSOVER'
    assert bear['sell_score'] == 0.3 and bear['buy_score'] == 0
    assert bear['signal'] == 'WEAK_SELL' and bear['direction'] is None


def test_fresh_macd_crossovers_vote_full_weight():
    bull = SignalAnalyzer().generate_signals(make_df(
        {'MACD': 0.1, 'MACD_SIGNAL': 0.0}, {'MACD': -0.1, 'MACD_SIGNAL': 0.0}))
    assert bull['components']['MACD'] == 'BULLISH CROSSOVER'
    assert bull['buy_score'] == 0.2 and bull['sell_score'] == 0

    bear = SignalAnalyzer().generate_signals(make_df(
        {'MACD': -0.1, 'MACD_SIGNAL': 0.0}, {'MACD': 0.1, 'MACD_SIGNAL': 0.0}))
    assert bear['components']['MACD'] == 'BEARISH CROSSOVER'
    assert bear['sell_score'] == 0.2 and bear['buy_score'] == 0


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


# ---------------------------------------------------------------------------
# Validation and ATR tests
# ---------------------------------------------------------------------------

def candle(o=1.0, h=1.1, l=0.9, c=1.05):
    return {'open': o, 'high': h, 'low': l, 'close': c}


def series_with(index, **overrides):
    """120 valid candles with one field of one candle overridden."""
    candles = make_candles()
    candles[index] = {**candles[index], **overrides}
    return candles


def test_missing_close_column_is_an_error():
    df = make_df().drop(columns=['close'])
    res = SignalAnalyzer().generate_signals(df)
    assert res['signal'] == 'ERROR' and res['direction'] is None
    assert 'close' in res['components']['error']
    assert SCHEMA - {'asset', 'timestamp'} <= set(res)  # schema unchanged


def test_missing_close_in_input_candles_is_rejected():
    candles = [{k: v for k, v in c.items() if k != 'close'} for c in make_candles()]
    a = SignalAnalyzer()
    assert a.prepare_data(candles) is None
    assert a.analyze(candles)['signal'] == 'ERROR'


def test_infinity_in_any_ohlc_field_is_rejected():
    a = SignalAnalyzer()
    for field in ('open', 'high', 'low', 'close'):
        for bad in (float('inf'), float('-inf')):
            candles = series_with(60, **{field: bad})
            assert a.prepare_data(candles) is None, (field, bad)
            res = a.analyze(candles)
            assert res['signal'] == 'ERROR', (field, bad, res)
            assert res['direction'] is None


def test_nan_and_unparseable_values_are_rejected_not_dropped():
    a = SignalAnalyzer()
    for field in ('open', 'high', 'low', 'close'):
        for bad in (float('nan'), None, 'abc'):
            candles = series_with(60, **{field: bad})
            assert a.prepare_data(candles) is None, (field, bad)
            assert a.analyze(candles)['signal'] == 'ERROR', (field, bad)


def test_invalid_ohlc_relationships_are_rejected():
    a = SignalAnalyzer()
    bad_candles = {
        'high below low': candle(o=1.0, h=0.8, l=0.9, c=0.85),
        'high below open': candle(o=1.2, h=1.1, l=0.9, c=1.0),
        'high below close': candle(o=1.0, h=1.1, l=0.9, c=1.2),
        'low above open': candle(o=0.8, h=1.1, l=0.9, c=1.0),
        'low above close': candle(o=1.0, h=1.1, l=0.9, c=0.8),
    }
    for name, bad in bad_candles.items():
        candles = make_candles()
        candles[60] = {**candles[60], **bad}
        assert a.prepare_data(candles) is None, name
        assert a.analyze(candles)['signal'] == 'ERROR', name


def test_zero_and_negative_prices_are_rejected():
    a = SignalAnalyzer()
    for field in ('open', 'high', 'low', 'close'):
        for bad in (0.0, -1.0):
            candles = series_with(60, **{field: bad})
            assert a.prepare_data(candles) is None, (field, bad)
            assert a.analyze(candles)['signal'] == 'ERROR', (field, bad)


def test_valid_candles_still_work():
    a = SignalAnalyzer()
    candles = make_candles()
    df = a.prepare_data(candles)
    assert df is not None and len(df) == len(candles)
    res = a.analyze(candles, 'EURUSD_otc')
    assert res['signal'] in {'BUY', 'SELL', 'WEAK_BUY', 'WEAK_SELL', 'NEUTRAL'}, res
    assert SCHEMA <= set(res)
    assert res['atr'] is not None and res['atr'] > 0
    # numeric strings are still accepted
    as_text = [{k: (str(v) if k != 'time' else v) for k, v in c.items()} for c in candles]
    assert a.prepare_data(as_text) is not None
    # flat (doji) candles where open == high == low == close are valid
    assert a.prepare_data([candle(1.0, 1.0, 1.0, 1.0)] * 3) is not None


def test_atr_matches_hand_calculated_wilder_reference():
    # period 3. True ranges (first candle uses high - low; the rest use prev close):
    #   c1 H12 L9           -> TR 3
    #   c2 H14 L9  (pc 11)  -> max(5, |14-11|=3, |9-11|=2)  = 5
    #   c3 H15 L11 (pc 12)  -> max(4, |15-12|=3, |11-12|=1) = 4
    #   c4 H18 L16 (pc 14)  -> max(2, |18-14|=4, |16-14|=2) = 4   (gap up)
    #   c5 H21 L15 (pc 17)  -> max(6, |21-17|=4, |15-17|=2) = 6
    # Wilder: ATR[2] = (3+5+4)/3 = 4.0 (simple-average seed)
    #         ATR[3] = (4.0*2 + 4)/3 = 4.0
    #         ATR[4] = (4.0*2 + 6)/3 = 14/3
    # (an EMA seeded from the first TR would give 3.7778 at index 2, not 4.0)
    candles = [
        candle(10, 12, 9, 11),
        candle(11, 14, 9, 12),
        candle(12, 15, 11, 14),
        candle(17, 18, 16, 17),
        candle(17, 21, 15, 19),
    ]
    a = SignalAnalyzer(atr_period=3)
    df = a.calculate_atr(a.prepare_data(candles))
    atr = df['ATR'].tolist()
    print('ATR reference values:', atr)
    assert np.isnan(atr[0]) and np.isnan(atr[1])
    assert abs(atr[2] - 4.0) < 1e-12
    assert abs(atr[3] - 4.0) < 1e-12
    assert abs(atr[4] - 14 / 3) < 1e-12


def test_atr_needs_a_full_period_before_first_value():
    a = SignalAnalyzer(atr_period=3)
    df = a.calculate_atr(a.prepare_data([candle(), candle()]))
    assert df['ATR'].isna().all()


# ---------------------------------------------------------------------------
# Constructor validation tests
# ---------------------------------------------------------------------------

INT_PARAMS = ('rsi_period', 'ema_fast', 'ema_slow', 'macd_fast', 'macd_slow',
              'macd_signal', 'bb_period', 'atr_period', 'volume_period')


def assert_rejected(param, value):
    """Constructing with param=value must raise a ValueError naming the param."""
    try:
        SignalAnalyzer(**{param: value})
    except ValueError as e:
        assert param in str(e), f'message does not name {param}: {e}'
        return
    raise AssertionError(f'{param}={value!r} was accepted')


def test_zero_and_negative_periods_rejected():
    for param in INT_PARAMS + ('history_size',):
        for bad in (0, -1, -14):
            assert_rejected(param, bad)


def test_float_and_string_periods_rejected():
    for param in INT_PARAMS + ('history_size',):
        for bad in (14.0, 2.5, '14', 'abc', None):
            assert_rejected(param, bad)


def test_boolean_periods_rejected():
    for param in INT_PARAMS + ('history_size',):
        for bad in (True, False):
            assert_rejected(param, bad)
    assert_rejected('bb_std', True)


def test_invalid_ema_ordering_rejected():
    for fast, slow in ((21, 21), (30, 21)):
        try:
            SignalAnalyzer(ema_fast=fast, ema_slow=slow)
        except ValueError as e:
            assert 'ema_fast' in str(e) and 'ema_slow' in str(e), str(e)
        else:
            raise AssertionError(f'ema_fast={fast}, ema_slow={slow} was accepted')


def test_invalid_macd_ordering_rejected():
    for fast, slow in ((26, 26), (40, 26)):
        try:
            SignalAnalyzer(macd_fast=fast, macd_slow=slow)
        except ValueError as e:
            assert 'macd_fast' in str(e) and 'macd_slow' in str(e), str(e)
        else:
            raise AssertionError(f'macd_fast={fast}, macd_slow={slow} was accepted')


def test_invalid_bb_std_rejected():
    for bad in (0, 0.0, -1, -0.5, float('nan'), float('inf'), float('-inf'),
                np.nan, np.inf, '2', None):
        assert_rejected('bb_std', bad)


def test_invalid_history_size_rejected():
    for bad in (0, -5, 2.5, 5.0, '5', None, True):
        assert_rejected('history_size', bad)


def test_valid_custom_configuration():
    a = SignalAnalyzer(rsi_period=7, ema_fast=5, ema_slow=13, macd_fast=8,
                       macd_slow=17, macd_signal=5, bb_period=10, bb_std=2.5,
                       atr_period=7, volume_period=10, history_size=50)
    assert (a.rsi_period, a.ema_fast, a.ema_slow) == (7, 5, 13)
    assert (a.macd_fast, a.macd_slow, a.macd_signal) == (8, 17, 5)
    assert (a.bb_period, a.bb_std, a.atr_period, a.volume_period) == (10, 2.5, 7, 10)
    assert a.signals_history.maxlen == 50
    assert a.min_candles == 23  # max(13, 17 + 5, 10, 8, 8) + 1
    res = a.analyze(make_candles(n=80), 'EURUSD_otc')
    assert res['signal'] != 'ERROR' and SCHEMA <= set(res), res
    # numpy integers / floats and an integer bb_std are valid
    b = SignalAnalyzer(rsi_period=np.int64(14), bb_std=np.float64(2.0), history_size=np.int32(10))
    assert b.rsi_period == 14 and isinstance(b.rsi_period, int) and b.signals_history.maxlen == 10
    assert SignalAnalyzer(bb_std=3).bb_std == 3


def test_default_configuration_still_works():
    a = SignalAnalyzer()
    assert (a.rsi_period, a.ema_fast, a.ema_slow) == (14, 9, 21)
    assert (a.macd_fast, a.macd_slow, a.macd_signal) == (12, 26, 9)
    assert (a.bb_period, a.bb_std, a.atr_period, a.volume_period) == (20, 2, 14, 20)
    assert a.signals_history.maxlen == 500
    assert a.min_candles == 36
    res = a.analyze(make_candles(), 'EURUSD_otc')
    assert res['signal'] != 'ERROR' and SCHEMA <= set(res), res


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
