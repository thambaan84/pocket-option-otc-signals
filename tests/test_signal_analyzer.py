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


# ---------------------------------------------------------------------------
# Independent indicator reference tests
#
# Every expected value below was derived by hand from the textbook definition
# (arithmetic shown in the comments) on small, fixed, deterministic data. None
# of it is computed by calling the code under test or a copy of it. These
# tests pin down the CURRENT behaviour; they do not claim the indicators have
# any predictive value.
# ---------------------------------------------------------------------------

def candles_from_closes(closes, volumes=None):
    """Valid OHLC candles around fixed closes (high/low = close +/- 0.5)."""
    out = []
    for i, c in enumerate(closes):
        row = {'time': i, 'open': c, 'high': c + 0.5, 'low': c - 0.5, 'close': c}
        if volumes is not None:
            row['volume'] = volumes[i]
        out.append(row)
    return out


def run_indicators(analyzer, candles):
    df = analyzer.prepare_data(candles)
    assert df is not None
    for step in (analyzer.calculate_rsi, analyzer.calculate_ema,
                 analyzer.calculate_macd, analyzer.calculate_bollinger_bands,
                 analyzer.calculate_atr, analyzer.calculate_volume_signal):
        df = step(df)
    return df


def assert_series(series, expected, tol=1e-9, label=''):
    """expected: list of floats, with None meaning 'must be NaN'."""
    got = series.tolist()
    assert len(got) == len(expected), (label, got)
    for i, (g, e) in enumerate(zip(got, expected)):
        if e is None:
            assert np.isnan(g), (label, i, g, 'expected NaN')
        else:
            assert not np.isnan(g) and abs(g - e) <= tol, (label, i, g, e)


def test_ema_matches_hand_calculated_reference():
    # closes 2, 4, 8, 6, 10. EMA = alpha*close + (1-alpha)*previous EMA with
    # alpha = 2/(span+1), seeded with the first close; the first span-1 rows are NaN.
    # span 2 (alpha 2/3): y1 = 2/3*4 + 1/3*2      = 10/3
    #                     y2 = 2/3*8 + 1/3*10/3   = 58/9
    #                     y3 = 2/3*6 + 1/3*58/9   = 166/27
    #                     y4 = 2/3*10 + 1/3*166/27 = 706/81
    # span 3 (alpha 1/2): 2, 3, 11/2, 23/4, 63/8
    a = SignalAnalyzer(ema_fast=2, ema_slow=3)
    df = run_indicators(a, candles_from_closes([2, 4, 8, 6, 10]))
    assert_series(df['EMA_FAST'], [None, 10 / 3, 58 / 9, 166 / 27, 706 / 81], label='EMA_FAST')
    assert_series(df['EMA_SLOW'], [None, None, 11 / 2, 23 / 4, 63 / 8], label='EMA_SLOW')


def test_rsi_matches_hand_calculated_recursive_ewm_reference():
    # Pins the CURRENT implementation. It is a recursive exponentially weighted
    # average, NOT the conventional SMA-seeded Wilder RSI (which would start from
    # the simple mean of the first `period` changes).
    # avg_gain / avg_loss here are pandas ewm(adjust=False, alpha=1/period) means:
    # they start from the FIRST observed price change (idx1; idx0 has no change),
    # then recurse as new = (1 - alpha)*old + alpha*x with alpha = 1/3 for period 3.
    # They exist from idx1 on, but min_periods=period hides them (NaN) until
    # `period` changes have been observed, so the first visible RSI is idx3.
    # closes 10, 12, 11, 14, 13, 11
    # change      -    +2   -1   +3   -1   -2
    # avg_gain    -    2    4/3  17/9 34/27 68/81   (seed = first gain, then 2/3*old + 1/3*gain)
    # avg_loss    -    0    1/3  2/9  13/27 80/81   (seed = first loss, then 2/3*old + 1/3*loss)
    # RSI = 100 * avg_gain / (avg_gain + avg_loss); idx0-2 are NaN (warm-up)
    #   idx3: 100*17/19 = 1700/19    idx4: 100*34/47 = 3400/47    idx5: 100*68/148 = 1700/37
    # (For contrast, SMA seeding at idx3 would give avg_gain 5/3, avg_loss 1/3,
    #  RSI = 100*5/6 = 83.33, not the 89.47 expected here.)
    a = SignalAnalyzer(rsi_period=3)
    df = run_indicators(a, candles_from_closes([10, 12, 11, 14, 13, 11]))
    assert_series(df['RSI'], [None, None, None, 1700 / 19, 3400 / 47, 1700 / 37], label='RSI')


def test_rsi_flat_market_is_50():
    # DOCUMENTED EXISTING BEHAVIOUR: when every change is zero, avg_gain and
    # avg_loss are both 0, RS is undefined, and _rsi deliberately returns the
    # neutral value 50 (not NaN, not 0 and not 100). Warm-up is unchanged: the
    # first `period` rows are still NaN.
    a = SignalAnalyzer(rsi_period=3)
    df = run_indicators(a, candles_from_closes([10] * 8))
    assert_series(df['RSI'], [None, None, None, 50.0, 50.0, 50.0, 50.0, 50.0],
                  tol=0, label='RSI flat')


def test_rsi_keeps_last_value_when_market_turns_flat_after_movement():
    # DOCUMENTED EXISTING BEHAVIOUR: after a flat stretch that FOLLOWS movement,
    # avg_gain and avg_loss both decay by the same factor (2/3 per flat bar), so
    # their ratio, and therefore RSI, stays at its last value. It does not
    # return to 50. Example: 10, 12, 11, 14, 13, 11, then 11, 11.
    #   idx6: avg_gain 136/243, avg_loss 160/243 -> 100*136/296 = 1700/37 (unchanged)
    a = SignalAnalyzer(rsi_period=3)
    df = run_indicators(a, candles_from_closes([10, 12, 11, 14, 13, 11, 11, 11]))
    last = 1700 / 37
    assert_series(df['RSI'], [None, None, None, 1700 / 19, 3400 / 47, last, last, last],
                  label='RSI stale after flat')


def test_rsi_rising_is_100_and_falling_is_0():
    # no losses -> 100 ; no gains -> RS = 0 -> 100 - 100/(1+0) = 0
    a = SignalAnalyzer(rsi_period=3)
    up = run_indicators(a, candles_from_closes([10, 11, 12, 13, 14, 15]))
    assert_series(up['RSI'], [None, None, None, 100.0, 100.0, 100.0], tol=0, label='RSI up')
    down = run_indicators(a, candles_from_closes([15, 14, 13, 12, 11, 10]))
    assert_series(down['RSI'], [None, None, None, 0.0, 0.0, 0.0], tol=0, label='RSI down')


def test_macd_matches_hand_calculated_reference():
    # fast 2, slow 3, signal 2 on closes 2, 4, 8, 6, 10 (EMA values as in the EMA test)
    # MACD = EMA2 - EMA3   (NaN until both exist, i.e. from idx2)
    #   idx2: 58/9 - 11/2    = 17/18
    #   idx3: 166/27 - 23/4  = 43/108
    #   idx4: 706/81 - 63/8  = 545/648
    # SIGNAL = EMA(span 2, alpha 2/3) of MACD, needs 2 MACD values -> from idx3
    #   idx3: 2/3*43/108 + 1/3*17/18    = 47/81
    #   idx4: 2/3*545/648 + 1/3*47/81   = 733/972
    # DIFF = MACD - SIGNAL
    #   idx3: 43/108 - 47/81 = -59/324     idx4: 545/648 - 733/972 = 169/1944
    a = SignalAnalyzer(macd_fast=2, macd_slow=3, macd_signal=2)
    df = run_indicators(a, candles_from_closes([2, 4, 8, 6, 10]))
    assert_series(df['MACD'], [None, None, 17 / 18, 43 / 108, 545 / 648], label='MACD')
    assert_series(df['MACD_SIGNAL'], [None, None, None, 47 / 81, 733 / 972], label='MACD_SIGNAL')
    assert_series(df['MACD_DIFF'], [None, None, None, -59 / 324, 169 / 1944], label='MACD_DIFF')


def test_bollinger_bands_match_hand_calculated_reference():
    # period 4, closes 2, 2, 6, 6, 10, 10. Population standard deviation (ddof=0).
    #   idx3 window 2,2,6,6   : mean 4, variance 16/4 = 4 , std 2
    #   idx4 window 2,6,6,10  : mean 6, variance 32/4 = 8 , std 2*sqrt(2)
    #   idx5 window 6,6,10,10 : mean 8, variance 16/4 = 4 , std 2
    # bands = mean +/- k*std. (A sample std, ddof=1, would give 2.309 at idx3.)
    r2 = np.sqrt(2)
    a = SignalAnalyzer(bb_period=4, bb_std=2)
    df = run_indicators(a, candles_from_closes([2, 2, 6, 6, 10, 10]))
    assert_series(df['BB_MID'], [None, None, None, 4, 6, 8], label='BB_MID')
    assert_series(df['BB_HIGH'], [None, None, None, 8, 6 + 4 * r2, 12], label='BB_HIGH k=2')
    assert_series(df['BB_LOW'], [None, None, None, 0, 6 - 4 * r2, 4], label='BB_LOW k=2')
    # custom multiplier k = 1.5
    b = SignalAnalyzer(bb_period=4, bb_std=1.5)
    df = run_indicators(b, candles_from_closes([2, 2, 6, 6, 10, 10]))
    assert_series(df['BB_HIGH'], [None, None, None, 7, 6 + 3 * r2, 11], label='BB_HIGH k=1.5')
    assert_series(df['BB_LOW'], [None, None, None, 1, 6 - 3 * r2, 5], label='BB_LOW k=1.5')


def test_volume_ratio_matches_hand_calculated_reference():
    # period 3, volumes 100, 200, 300, 600, 300 ; ratio = volume / rolling mean
    #   idx2: mean 200            -> 300/200 = 1.5
    #   idx3: mean 1100/3         -> 600*3/1100 = 18/11
    #   idx4: mean 400            -> 300/400 = 0.75
    a = SignalAnalyzer(volume_period=3)
    df = run_indicators(a, candles_from_closes([10, 11, 12, 13, 14], [100, 200, 300, 600, 300]))
    assert_series(df['VOLUME_RATIO'], [None, None, 1.5, 18 / 11, 0.75], label='VOLUME_RATIO')


def test_volume_ratio_zero_volume_window_is_nan():
    # volumes 0, 0, 0, 5, 10, period 3
    #   idx2: window all zero -> 0/0 -> NaN
    #   idx3: mean 5/3 -> 5/(5/3) = 3      idx4: mean 5 -> 10/5 = 2
    a = SignalAnalyzer(volume_period=3)
    df = run_indicators(a, candles_from_closes([10, 11, 12, 13, 14], [0, 0, 0, 5, 10]))
    assert_series(df['VOLUME_RATIO'], [None, None, None, 3.0, 2.0], label='VOLUME_RATIO zero')


def test_volume_ratio_zero_mean_guard_gives_nan_not_inf():
    # The implementation replaces a window mean of exactly 0 with NaN. 0/0 is
    # already NaN, so only a NON-zero volume over a zero mean exercises that
    # guard, which needs signed volumes: -5, 0, 5, 10, 20 with period 3
    #   idx2: window -5,0,5 -> mean 0, volume 5 -> would be inf; guarded to NaN
    #   idx3: window 0,5,10 -> mean 5   -> 10/5 = 2
    #   idx4: window 5,10,20 -> mean 35/3 -> 20*3/35 = 12/7
    a = SignalAnalyzer(volume_period=3)
    df = run_indicators(a, candles_from_closes([10, 11, 12, 13, 14], [-5, 0, 5, 10, 20]))
    assert_series(df['VOLUME_RATIO'], [None, None, None, 2.0, 12 / 7], label='VOLUME_RATIO guard')
    assert not np.isinf(df['VOLUME_RATIO'].to_numpy(dtype=float)).any()


def test_volume_ratio_is_all_nan_without_usable_volume():
    a = SignalAnalyzer(volume_period=3)
    closes = [10, 11, 12, 13, 14]
    for volumes in (None, [100] * 5, [0] * 5):  # missing, constant, all zero
        df = run_indicators(a, candles_from_closes(closes, volumes))
        assert df['VOLUME_RATIO'].isna().all(), volumes


def test_default_period_warmup_and_nan_placement():
    # With default periods the first valid row (0-based index) of each column is:
    #   RSI 14 | EMA_FAST 8 (span 9) | EMA_SLOW 20 (span 21) | MACD 25 (slow 26)
    #   MACD_SIGNAL / MACD_DIFF 33 (25 + signal 9 - 1) | BB 19 | ATR 13 | VOLUME_RATIO 19
    # Everything before is NaN, everything from there on is a number.
    closes = [1.0 + 0.01 * i for i in range(60)]
    volumes = [100 + (i % 7) * 10 for i in range(60)]
    df = run_indicators(SignalAnalyzer(), candles_from_closes(closes, volumes))
    first_valid = {'RSI': 14, 'EMA_FAST': 8, 'EMA_SLOW': 20, 'MACD': 25,
                   'MACD_SIGNAL': 33, 'MACD_DIFF': 33, 'BB_MID': 19, 'BB_HIGH': 19,
                   'BB_LOW': 19, 'ATR': 13, 'VOLUME_RATIO': 19}
    for col, first in first_valid.items():
        assert df[col].iloc[:first].isna().all(), (col, 'NaN expected before', first)
        assert df[col].iloc[first:].notna().all(), (col, 'valid expected from', first)


def test_min_candles_gives_valid_last_two_rows():
    cols = ['RSI', 'EMA_FAST', 'EMA_SLOW', 'MACD', 'MACD_SIGNAL', 'BB_LOW', 'BB_HIGH', 'ATR']
    a = SignalAnalyzer()
    closes = [1.0 + 0.01 * i for i in range(a.min_candles)]
    df = run_indicators(a, candles_from_closes(closes))
    assert a.min_candles == 36
    assert df[cols].iloc[-2:].notna().all().all()
    # one MACD-signal warm-up short: 34 candles leave the previous row NaN
    short = run_indicators(a, candles_from_closes(closes[:34]))
    assert np.isnan(short['MACD_SIGNAL'].iloc[-2]) and not np.isnan(short['MACD_SIGNAL'].iloc[-1])


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
