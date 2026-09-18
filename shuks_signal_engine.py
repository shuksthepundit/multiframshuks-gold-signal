#!/usr/bin/env python3
"""
SHUKS MULTI-TIMEFRAME SIGNAL ENGINE  (Gold / CFDs)
==================================================
Flow
  1. Load 5-minute candles (MetaTrader5, CSV, or demo) and resample to
     5m,10m,15m,30m,45m,1h,2h,3h,4h,1d  (only CLOSED candles are used).
  2. Compute SMA1/SMA3 (high & low), MACD, Williams %R, ADX/+DI/-DI, wicks,
     and market structure (swings, BOS, CHoCH) on every timeframe.
  3. Run your rules (numbered R1..R17 below). Each rule adds "evidence":
        trigger -> says WHICH DIRECTION to trade now
        level   -> says WHERE to enter (price zones)
        bias    -> MACD directional lean
  4. Gate: need >=3 of 4 confirmations on 5m/15m/30m/1h (4 of 4 = "BIG MOVE").
  5. Choose the entry zone: rule 7 (left-side check), rule 10 (pullback grid),
     rule 14 (highest zone for sells, lowest for buys), then set SL / TP.

Usage
  python shuks_signal_engine.py --demo
  python shuks_signal_engine.py --csv XAUUSD_M5.csv
  python shuks_signal_engine.py --mt5 XAUUSD --bars 40000
  python shuks_signal_engine.py --mt5 XAUUSD,EURUSD,US30 --auto-scale
  python shuks_signal_engine.py --mt5 XAUUSD --watch 60 --out signals.csv

Educational tool - not financial advice. Test on a demo account first.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TF_MIN = {"5m": 5, "10m": 10, "15m": 15, "30m": 30, "45m": 45,
          "1h": 60, "2h": 120, "3h": 180, "4h": 240, "1d": 1440}


def tf_rule(tf: str) -> str:
    return "1D" if tf == "1d" else f"{TF_MIN[tf]}min"


# =============================================================================
# CONFIG  - every number from your rules lives here (gold price units)
# =============================================================================
@dataclass
class Config:
    symbol: str = "XAUUSD"
    base_min: int = 5

    # timeframe groups
    all_tfs: List[str] = field(default_factory=lambda: list(TF_MIN))
    r1_htf: List[str] = field(default_factory=lambda: ["1h", "2h", "4h"])
    r1_ltf: List[str] = field(default_factory=lambda: ["10m", "15m", "30m", "45m"])
    macd_tfs: List[str] = field(default_factory=lambda: ["1h", "2h", "3h", "4h"])
    conv_tfs: List[str] = field(default_factory=lambda: ["1h", "2h", "3h", "4h", "1d"])
    mtf_tfs: List[str] = field(default_factory=lambda: ["5m", "15m", "30m", "1h"])
    struct_tfs: List[str] = field(default_factory=lambda: ["15m", "30m", "1h", "4h"])
    wr_tfs: List[str] = field(default_factory=lambda: ["5m", "15m", "30m", "1h"])
    scalp_tfs: List[str] = field(default_factory=lambda: ["5m", "15m", "1h"])
    range_tfs: List[str] = field(default_factory=lambda: ["15m", "30m", "1h", "2h", "4h"])
    grid_tfs: List[str] = field(default_factory=lambda: ["1h", "2h", "4h", "1d"])

    # scaling of all price offsets (1.0 = gold). --auto-scale sets it from D1 ATR
    scale: float = 1.0
    auto_scale: bool = False
    gold_ref_daily_atr: float = 66.0      # your example day: 4400-4334 = 66

    # ---- your numbers ----
    ext_offset: float = 42.0              # R1: +42 above highs (sell) / -42 below lows (buy)
    prog_bars: int = 3                    # R1: three candles with progressive highs/lows
    range_add: float = 17.0               # R12.2 / R16
    sell_back: float = 10.0               # R13: sell range = X-10 .. X
    buy_back: float = 28.5                # R13: buy range = (X-28.5-10) .. (X-28.5)
    discount: float = 4.0                 # R13: "good discount" shift
    pullback_offsets: List[int] = field(default_factory=lambda: [7, 14, 21, 35, 42, 49, 56, 63, 70, 77])
    ladder_div: float = 4.0               # R16/17: ((range+17)+17)/4
    ladder_steps: int = 4
    sessions: int = 2                     # "after two sessions" (daily candles)
    left_bars: Tuple[int, int, int] = (7, 8, 9)   # R8/R12.1 candles to the left

    # ---- indicators ----
    adx_len: int = 14
    adx_min: float = 25.0
    wr_len: int = 14
    wr_sell: float = -20.0
    wr_buy: float = -80.0

    # ---- rule behaviour ----
    r1_max_age: int = 40                  # bars: pattern must be this fresh
    r1_reconfirm_tol: float = 15.0
    conv_max_age: int = 60
    region_bars: int = 20
    range_bars: int = 24
    struct_window: int = 600
    swing_n: int = 3
    event_age: int = 6                    # BOS/CHoCH must be this recent (bars)
    wick_ratio: float = 0.50              # R5: rejection wick / candle range
    cross_recent: int = 3                 # "transition" = cross within last N bars

    # ---- R7 left side check (15m candles that already printed) ----
    left_lookback: int = 1000
    left_min_touches: int = 6
    left_tol: float = 3.0

    # ---- R10 grid / fib confluence ----
    grid_tol: float = 2.0

    # ---- decision ----
    min_mtf: int = 3                      # 3 confirmations = valid, 4 = BIG MOVE
    mtf_weight: float = 1.0
    bias_weight: float = 0.5
    min_score: float = 4.0
    min_margin: float = 1.5
    min_confluence: int = 2
    max_dist: float = 45.0                # zone must be within this of price
    zone_tol: float = 8.0                 # levels this close count as confluent

    # ---- risk (YOUR RULES DON'T SPECIFY THESE - my defaults, change freely) ----
    sl_steps: float = 0.75
    tp_steps: Tuple[float, float, float] = (1.0, 2.0, 3.0)


# =============================================================================
# DATA
# =============================================================================
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next(c for c in ("time", "datetime", "date", "timestamp") if c in df.columns)
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.set_index(tcol).sort_index()
    if "volume" not in df.columns:
        df["volume"] = df["tick_volume"] if "tick_volume" in df.columns else 0.0
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def load_mt5(symbol: str, bars: int) -> pd.DataFrame:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        sys.exit("pip install MetaTrader5  (Windows, MT5 terminal must be running)")
    if not mt5.initialize():
        sys.exit(f"MT5 initialize failed: {mt5.last_error()}")
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        sys.exit(f"No data for {symbol}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").rename(columns={"tick_volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]].astype(float).iloc[:-1]  # drop forming bar


def demo_data(seed: int = 7, n: int = 30000, start_price: float = 4350.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-05-01", periods=n, freq="5min")
    vol = 1.6 * (1 + 0.6 * np.sin(np.arange(n) / 700.0)) * (1 + 0.3 * rng.standard_normal(n).clip(-2, 2))
    drift = 0.02 * np.sin(np.arange(n) / 1500.0)
    ret = drift + vol * rng.standard_normal(n)
    close = start_price + np.cumsum(ret)
    open_ = np.r_[start_price, close[:-1]]
    hi = np.maximum(open_, close) + np.abs(rng.standard_normal(n)) * vol * 0.6
    lo = np.minimum(open_, close) - np.abs(rng.standard_normal(n)) * vol * 0.6
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                         "volume": rng.integers(50, 500, n).astype(float)}, index=idx)


def resample(base: pd.DataFrame, tf: str, base_min: int) -> pd.DataFrame:
    if TF_MIN[tf] == base_min:
        df = base.copy()
    else:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df = base.resample(tf_rule(tf)).agg(agg).dropna(subset=["open"])
    bar_end = df.index[-1] + pd.Timedelta(minutes=TF_MIN[tf])
    data_end = base.index[-1] + pd.Timedelta(minutes=base_min)
    if bar_end > data_end:               # last higher-TF candle still forming -> ignore
        df = df.iloc[:-1]
    return df


# =============================================================================
# INDICATORS
# =============================================================================
def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df.copy()
    # SMA1 = the raw high/low itself, SMA3 = 3-bar average of the high/low
    d["s1_low"], d["s3_low"] = d["low"], d["low"].rolling(3).mean()
    d["s1_high"], d["s3_high"] = d["high"], d["high"].rolling(3).mean()
    d["x_up_low"] = (d.s1_low > d.s3_low) & (d.s1_low.shift(1) <= d.s3_low.shift(1))    # SMA1 low crosses ABOVE SMA3 low
    d["x_dn_high"] = (d.s1_high < d.s3_high) & (d.s1_high.shift(1) >= d.s3_high.shift(1))  # SMA1 high crosses BELOW SMA3 high

    ef, es = d.close.ewm(span=12, adjust=False).mean(), d.close.ewm(span=26, adjust=False).mean()
    d["macd"] = ef - es
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    d["macd_x_up"] = (d.macd > d.macd_sig) & (d.macd.shift(1) <= d.macd_sig.shift(1))
    d["macd_x_dn"] = (d.macd < d.macd_sig) & (d.macd.shift(1) >= d.macd_sig.shift(1))

    n = cfg.wr_len
    hh, ll = d.high.rolling(n).max(), d.low.rolling(n).min()
    d["wr"] = -100 * (hh - d.close) / (hh - ll).replace(0, np.nan)

    n = cfg.adx_len
    up, dn = d.high.diff(), -d.low.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    d["atr"] = atr
    d["pdi"] = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    d["mdi"] = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (d.pdi - d.mdi).abs() / (d.pdi + d.mdi).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1 / n, adjust=False).mean()

    rng = (d.high - d.low).replace(0, np.nan)
    d["up_wick"] = (d.high - d[["open", "close"]].max(axis=1)) / rng
    d["lo_wick"] = (d[["open", "close"]].min(axis=1) - d.low) / rng
    return d


def find_structure(d: pd.DataFrame, n: int = 3) -> dict:
    """Swing points (fractals), BOS and CHoCH. Only confirmed pivots are used."""
    hi, lo, cl = d.high.values, d.low.values, d.close.values
    sh = sl = None
    trend, events = 0, []
    for t in range(len(d)):
        p = t - n
        if p >= n:
            if hi[p] == hi[p - n:p + n + 1].max():
                sh = [p, hi[p], False]
            if lo[p] == lo[p - n:p + n + 1].min():
                sl = [p, lo[p], False]
        if sh and not sh[2] and cl[t] > sh[1]:
            events.append(dict(i=t, kind="BOS" if trend >= 0 else "CHOCH", dir=1, level=sh[1]))
            sh[2], trend = True, 1
        if sl and not sl[2] and cl[t] < sl[1]:
            events.append(dict(i=t, kind="BOS" if trend <= 0 else "CHOCH", dir=-1, level=sl[1]))
            sl[2], trend = True, -1
    return dict(events=events, sh=sh[1] if sh else None, sl=sl[1] if sl else None, trend=trend)


# =============================================================================
# OUTPUT TYPES
# =============================================================================
@dataclass
class Evidence:
    rule: int
    tf: str
    side: str            # buy | sell
    kind: str            # trigger | level | bias
    note: str
    weight: float = 1.0
    zone: Optional[Tuple[float, float]] = None


@dataclass
class Signal:
    time: str
    symbol: str
    side: str                       # BUY | SELL | WAIT
    order_type: str = ""            # LIMIT | ACTIVE (price inside zone) | MARKET
    entry: float = 0.0
    entry_zone: Tuple[float, float] = (0.0, 0.0)
    discount_zone: Tuple[float, float] = (0.0, 0.0)
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    confirmations: List[str] = field(default_factory=list)
    big_move: bool = False
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    exit_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# ENGINE
# =============================================================================
class Engine:
    def __init__(self, base: pd.DataFrame, cfg: Config):
        self.cfg, self.base = cfg, base
        self.tf = {n: add_indicators(resample(base, n, cfg.base_min), cfg) for n in cfg.all_tfs}
        self.price = float(base.close.iloc[-1])
        if cfg.auto_scale:
            atr_d = float(self.tf["1d"].atr.iloc[-1])
            cfg.scale = atr_d / cfg.gold_ref_daily_atr
        self.ev: List[Evidence] = []
        self.ctx: dict = {}
        self._struct: Dict[str, dict] = {}

    # ---- helpers ----
    def k(self, x: float) -> float:
        return x * self.cfg.scale

    def struct(self, tf: str) -> dict:
        if tf not in self._struct:
            self._struct[tf] = find_structure(self.tf[tf].tail(self.cfg.struct_window), self.cfg.swing_n)
        return self._struct[tf]

    def zones_from_level(self, X: float, orient: str):
        """R13: sell = X-10..X ; buy = X-28.5-10..X-28.5 (mirrored for a down projection)."""
        b, s = self.k(self.cfg.sell_back), self.k(self.cfg.buy_back)
        if orient == "up":
            return (X - b, X), (X - s - b, X - s)
        return (X + s, X + s + b), (X, X + b)

    def shift_disc(self, zone, side):
        dd = self.k(self.cfg.discount) * (1 if side == "sell" else -1)   # R13 "good discount"
        return (zone[0] + dd, zone[1] + dd)

    def add_zone_pair(self, rule, tf, X, orient, note, w=0.4):
        sell_z, buy_z = self.zones_from_level(X, orient)
        self.ev.append(Evidence(rule, tf, "sell", "level", f"{note} -> X={X:.2f}", w, sell_z))
        self.ev.append(Evidence(rule, tf, "buy", "level", f"{note} -> X={X:.2f}", w, buy_z))

    @staticmethod
    def recent(series: pd.Series, n: int) -> bool:
        return bool(series.tail(n).any())

    # ---- R1: first intersection SMA1/SMA3 + 3 progressive candles +/- 42 ----
    def r1_first_intersection(self):
        cfg, found = self.cfg, []
        for tf in cfg.r1_htf + cfg.r1_ltf:
            d, m = self.tf[tf], cfg.prog_bars
            H, L = d.high.values, d.low.values
            for i in np.flatnonzero(d.x_up_low.values)[::-1]:            # SMA1 low up through SMA3 low
                if len(d) - 1 - (i + m - 1) > cfg.r1_max_age:
                    break
                if i + m - 1 < len(d) and all(H[i + j] < H[i + j + 1] for j in range(m - 1)):
                    lvl = H[i:i + m].max() + self.k(cfg.ext_offset)      # +42 above highest -> SELL
                    found.append(Evidence(1, tf, "sell", "level", f"SMA1-low x-up SMA3-low + {m} rising highs, +{cfg.ext_offset:g}",
                                          0.5, (lvl - self.k(1.5), lvl + self.k(1.5))))
                    break
            for i in np.flatnonzero(d.x_dn_high.values)[::-1]:           # SMA1 high down through SMA3 high
                if len(d) - 1 - (i + m - 1) > cfg.r1_max_age:
                    break
                if i + m - 1 < len(d) and all(L[i + j] > L[i + j + 1] for j in range(m - 1)):
                    lvl = L[i:i + m].min() - self.k(cfg.ext_offset)      # -42 below lowest -> BUY
                    found.append(Evidence(1, tf, "buy", "level", f"SMA1-high x-dn SMA3-high + {m} falling lows, -{cfg.ext_offset:g}",
                                          0.5, (lvl - self.k(1.5), lvl + self.k(1.5))))
                    break
        # re-confirm the 1h/2h/4h price on 10/15/30/45m
        tol = self.k(cfg.r1_reconfirm_tol)
        for e in found:
            if e.tf in cfg.r1_htf:
                mid = sum(e.zone) / 2
                agree = [o.tf for o in found if o.tf in cfg.r1_ltf and o.side == e.side and abs(sum(o.zone) / 2 - mid) <= tol]
                if agree:
                    e.weight += 0.3 * len(agree)
                    e.note += f" | re-confirmed on {','.join(agree)}"
        self.ev += found

    # ---- R2: MACD 1h-4h -> bias and next likely range region ----
    def r2_macd_region(self):
        cfg, votes, lows, highs = self.cfg, [], [], []
        for tf in cfg.macd_tfs:
            d = self.tf[tf]
            r, p = d.iloc[-1], d.iloc[-2]
            bias = 1 if (r.macd > r.macd_sig and r.macd_hist >= p.macd_hist) else -1 if (r.macd < r.macd_sig and r.macd_hist <= p.macd_hist) else 0
            w = d.tail(cfg.region_bars)
            lows.append(w.low.min()); highs.append(w.high.max()); votes.append(bias)
            if bias:
                self.ev.append(Evidence(2, tf, "buy" if bias > 0 else "sell", "bias",
                                        f"MACD {'bull' if bias > 0 else 'bear'} (hist {'rising' if bias > 0 else 'falling'})", cfg.bias_weight))
        net, lo, hi = sum(votes), float(np.median(lows)), float(np.median(highs))
        R = hi - lo
        region = (hi - 0.25 * R, hi + 0.5 * R) if net >= 3 else (lo - 0.5 * R, lo + 0.25 * R) if net <= -3 else (lo, hi)
        self.ctx["macd_net"], self.ctx["region"] = net, region

    # ---- R3, R9, R11: BOS / CHoCH (fade), rejection + hold at new S/R ----
    def r3_r9_structure(self):
        cfg, tol = self.cfg, self.k(3)
        for tf in cfg.struct_tfs:
            d = self.tf[tf].tail(cfg.struct_window)
            st = self.struct(tf)
            self.ctx.setdefault("trend", {})[tf] = st["trend"]
            if not st["events"]:
                continue
            e = st["events"][-1]
            age = len(d) - 1 - e["i"]
            if age > cfg.event_age:
                continue
            L, dr, last = e["level"], e["dir"], d.iloc[-1]
            after = d.iloc[e["i"] + 1:]
            opp = "sell" if dr == 1 else "buy"
            same = "buy" if dr == 1 else "sell"
            w = 1.2 if e["kind"] == "CHOCH" else 1.0
            # R3: enter the OPPOSITE trade after a BOS/CHoCH
            self.ev.append(Evidence(3, tf, opp, "trigger", f"{e['kind']} {'up' if dr == 1 else 'down'} at {L:.2f} -> fade", w))
            self.ev.append(Evidence(3, tf, opp, "level", f"{e['kind']} level {L:.2f}", 0.4, (L - tol, L + tol)))
            # R9: did participants reject the new S/R and did price hold beyond it?
            if len(after):
                if dr == 1:
                    failed = bool((after.close < L).any())
                    held_rej = bool(((after.low <= L + tol) & (after.close > L) & (after.lo_wick >= 0.4)).any())
                    sma_ok = last.s1_low > last.s3_low
                else:
                    failed = bool((after.close > L).any())
                    held_rej = bool(((after.high >= L - tol) & (after.close < L) & (after.up_wick >= 0.4)).any())
                    sma_ok = last.s1_high < last.s3_high
                if failed:
                    self.ev.append(Evidence(9, tf, opp, "trigger", f"price could NOT hold beyond {L:.2f} (failed break)", 1.5))
                elif held_rej and sma_ok:
                    self.ev.append(Evidence(9, tf, same, "trigger", f"rejection wick + hold at new S/R {L:.2f}, SMA1/3 aligned", 1.2))

    # ---- R4: Williams %R extreme + SMA1/SMA3 transition ----
    def r4_wr_transition(self):
        cfg = self.cfg
        for tf in cfg.wr_tfs:
            d = self.tf[tf]
            wr = d.wr.tail(cfg.cross_recent)
            if wr.max() >= cfg.wr_sell and self.recent(d.x_dn_high, cfg.cross_recent):
                self.ev.append(Evidence(4, tf, "sell", "trigger", f"%R>={cfg.wr_sell:g} & SMA1-high crossed below SMA3-high", 1.0))
            if wr.min() <= cfg.wr_buy and self.recent(d.x_up_low, cfg.cross_recent):
                self.ev.append(Evidence(4, tf, "buy", "trigger", f"%R<={cfg.wr_buy:g} & SMA1-low crossed above SMA3-low", 1.0))

    # ---- R5: two candles, same price rejection (scalping) ----
    def r5_rejection(self):
        cfg = self.cfg
        for tf in cfg.scalp_tfs:
            t = self.tf[tf].tail(2)
            if (t.up_wick >= cfg.wick_ratio).all():
                z = t.high.max()
                self.ev.append(Evidence(5, tf, "sell", "trigger", "2 candles with upper-wick rejection", 0.8, (z - self.k(3), z)))
            if (t.lo_wick >= cfg.wick_ratio).all():
                z = t.low.min()
                self.ev.append(Evidence(5, tf, "buy", "trigger", "2 candles with lower-wick rejection", 0.8, (z, z + self.k(3))))

    # ---- R6: 1h open-price transition + ADX/DI > 25 + Williams %R ----
    def r6_open_trend_adx(self):
        cfg, d = self.cfg, self.tf["1h"]
        o, r = d.open.values[-3:], d.iloc[-1]
        up, dn = o[0] < o[1] < o[2], o[0] > o[1] > o[2]
        wr = d.wr.tail(cfg.cross_recent)
        if dn and r.mdi > cfg.adx_min and wr.max() >= cfg.wr_sell:
            self.ev.append(Evidence(6, "1h", "sell", "trigger", f"1h opens falling, -DI {r.mdi:.0f}>25, %R {wr.max():.0f}", 1.2))
        if up and r.pdi > cfg.adx_min and wr.min() <= cfg.wr_buy:
            self.ev.append(Evidence(6, "1h", "buy", "trigger", f"1h opens rising, +DI {r.pdi:.0f}>25, %R {wr.min():.0f}", 1.2))
        self.ctx["open_trend_1h"] = "UP" if up else "DOWN" if dn else "MIXED"

    # ---- R8 / R12.1: MACD convergence + candle-range subtotal ----
    def r8_convergence(self):
        cfg, cands = self.cfg, []
        for tf in cfg.conv_tfs:
            d = self.tf[tf]
            xs = np.flatnonzero((d.macd_x_up | d.macd_x_dn).values)
            if not len(xs):
                continue
            c = int(xs[-1])
            if len(d) - 1 - c > cfg.conv_max_age or c < max(cfg.left_bars):
                continue
            rc = d.high.iloc[c] - d.low.iloc[c]
            longest = max(d.high.iloc[c - j] - d.low.iloc[c - j] for j in cfg.left_bars)
            sub = float(rc + longest)
            up = bool(d.macd_x_up.iloc[c])
            X = float(d.low.iloc[c] + sub) if up else float(d.high.iloc[c] - sub)
            cands.append((sub, tf, X, "up" if up else "down", rc, longest))
        for sub, tf, X, o, rc, lg in cands:
            self.add_zone_pair(8, tf, X, o, f"MACD convergence: candle {rc:.1f} + longest(7/8/9 left) {lg:.1f} = {sub:.1f}")
        if cands:
            best = max(cands)                     # "pick the highest from all those time frames"
            self.ctx["best_convergence"] = dict(tf=best[1], subtotal=round(best[0], 2), X=round(best[2], 2), orient=best[3])
            for e in self.ev:
                if e.rule == 8 and e.tf == best[1]:
                    e.weight += 0.3; e.note += " [largest subtotal]"

    # ---- R12.2: (high-low)+17 per timeframe/day -> X and trend ----
    def r12_range_plus17(self):
        cfg = self.cfg
        wins = {"day": self.base.tail(max(1, 1440 // cfg.base_min))}
        for tf in cfg.range_tfs:
            wins[tf] = self.tf[tf].tail(cfg.range_bars)
        trends = {}
        for name, w in wins.items():
            hi, lo = float(w.high.max()), float(w.low.min())
            R = hi - lo
            up = self.price >= lo + R / 2
            X = lo + R + self.k(cfg.range_add) if up else hi - R - self.k(cfg.range_add)
            trends[name] = "UP" if up else "DOWN"
            self.add_zone_pair(12, name, float(X), "up" if up else "down", f"range {R:.1f}+{cfg.range_add:g}, trend {trends[name]}")
        self.ctx["range_trend"] = trends

    # ---- R16 / R17: two-session range -> step -> enhanced ladder ----
    def r16_ladder(self):
        cfg, d = self.cfg, self.tf["1d"]
        w = d.tail(cfg.sessions)
        hi, lo = float(w.high.max()), float(w.low.min())
        R = hi - lo
        step = ((R + self.k(cfg.range_add)) + self.k(cfg.range_add)) / cfg.ladder_div
        levels = [hi - i * step for i in range(1, cfg.ladder_steps + 1)] + [lo + i * step for i in range(1, cfg.ladder_steps + 1)]
        self.ctx["two_session"] = dict(high=round(hi, 2), low=round(lo, 2), range=round(R, 2), step=round(step, 2))
        self.step = step
        for lv in sorted(set(round(x, 2) for x in levels)):
            side = "sell" if lv > self.price else "buy"
            self.ev.append(Evidence(17, "1d", side, "level", f"ladder level (step {step:.1f})", 0.3, (lv - self.k(1), lv + self.k(1))))

    # ---- R7: left-side check (did price print there before?) ----
    def left_touches(self, zone) -> int:
        d = self.tf["15m"].tail(self.cfg.left_lookback)
        tol = self.k(self.cfg.left_tol)
        return int(((d.low <= zone[1] + tol) & (d.high >= zone[0] - tol)).sum())

    # ---- R10: pullback grid (+/-7,14,...,77 from swing points) + fib ----
    def grid_hits(self, price: float) -> List[str]:
        cfg, hits, tol = self.cfg, [], self.k(self.cfg.grid_tol)
        for tf in cfg.grid_tfs:
            st = self.struct(tf)
            for nm, anchor in (("swing-high", st["sh"]), ("swing-low", st["sl"])):
                if anchor is None:
                    continue
                for off in cfg.pullback_offsets:
                    for sg in (1, -1):
                        if abs(anchor + sg * self.k(off) - price) <= tol:
                            hits.append(f"{tf} {nm} {sg * off:+d}")
            if st["sh"] and st["sl"]:
                rg = st["sh"] - st["sl"]
                for r in (0.382, 0.5, 0.618):
                    if abs(st["sl"] + r * rg - price) <= tol:
                        hits.append(f"{tf} fib {r}")
        return hits

    # ---- multi-timeframe confirmation (5m / 15m / 30m / 1h) ----
    def mtf_confirm(self, side: str) -> List[str]:
        out = []
        for tf in self.cfg.mtf_tfs:
            d = self.tf[tf]
            r, p = d.iloc[-1], d.iloc[-2]
            if side == "sell" and r.s1_high < r.s3_high and r.macd_hist < p.macd_hist:
                out.append(tf)
            if side == "buy" and r.s1_low > r.s3_low and r.macd_hist > p.macd_hist:
                out.append(tf)
        return out

    # ---- DECISION ----
    def run(self) -> Signal:
        self.step = self.k(25.0)
        for fn in (self.r16_ladder, self.r1_first_intersection, self.r2_macd_region, self.r3_r9_structure,
                   self.r4_wr_transition, self.r5_rejection, self.r6_open_trend_adx,
                   self.r8_convergence, self.r12_range_plus17):
            fn()
        return self.decide()

    def decide(self) -> Signal:
        cfg, price = self.cfg, self.price
        now = str(self.base.index[-1])
        score, trig, why = {"buy": 0.0, "sell": 0.0}, {"buy": 0, "sell": 0}, {"buy": [], "sell": []}
        conf = {s: self.mtf_confirm(s) for s in ("buy", "sell")}
        for e in self.ev:
            if e.kind in ("trigger", "bias"):
                score[e.side] += e.weight
                why[e.side].append(f"R{e.rule} [{e.tf}] {e.note}")
                trig[e.side] += e.kind == "trigger"
        for s in score:
            score[s] += cfg.mtf_weight * len(conf[s])
        side = "buy" if score["buy"] >= score["sell"] else "sell"
        other = "sell" if side == "buy" else "buy"
        margin = score[side] - score[other]

        sig = Signal(time=now, symbol=cfg.symbol, side="WAIT", score=round(score[side], 2),
                     confirmations=conf[side], reasons=why[side][:12])
        wait = []
        if len(conf[side]) < cfg.min_mtf:
            wait.append(f"only {len(conf[side])}/4 confirmations on 5m/15m/30m/1h (need {cfg.min_mtf})")
        if trig[side] == 0:
            wait.append("no trigger rule (R3/R4/R5/R6/R9) fired")
        if score[side] < cfg.min_score:
            wait.append(f"score {score[side]:.1f} < {cfg.min_score}")
        if margin < cfg.min_margin:
            wait.append(f"buy/sell scores too close ({score['buy']:.1f} vs {score['sell']:.1f})")
        if wait:
            sig.warnings = ["No trade: " + "; ".join(wait)]
            return sig

        # ---- choose entry zone: R7 left-side, R10 confluence, R14 highest sell / lowest buy ----
        tol = self.k(cfg.zone_tol)
        lv = [e for e in self.ev if e.kind == "level" and e.side == side and e.zone]
        survivors = []
        for e in lv:
            mid = sum(e.zone) / 2
            if side == "sell" and (e.zone[1] < price - self.k(3) or mid - price > self.k(cfg.max_dist)):
                continue
            if side == "buy" and (e.zone[0] > price + self.k(3) or price - mid > self.k(cfg.max_dist)):
                continue
            rules = {o.rule for o in lv if abs(sum(o.zone) / 2 - mid) <= tol}
            grid = self.grid_hits(mid)
            confl = len(rules) + (1 if grid else 0)
            if confl < cfg.min_confluence:
                continue
            touches = self.left_touches(e.zone)
            if touches < cfg.left_min_touches:
                continue
            survivors.append((mid, e, confl, grid, touches))
        step = self.step
        if survivors:
            pick = max(survivors, key=lambda x: x[0]) if side == "sell" else min(survivors, key=lambda x: x[0])
            mid, e, confl, grid, touches = pick
            zone = (round(e.zone[0], 2), round(e.zone[1], 2))
            inside = zone[0] - self.k(2) <= price <= zone[1] + self.k(2)
            sig.order_type = "ACTIVE" if inside else "LIMIT"
            entry = price if inside else (zone[0] + zone[1]) / 2
            sig.reasons.append(f"ZONE: R{e.rule}[{e.tf}] {e.note}; confluence={confl}; left-side touches={touches}"
                               + (f"; grid/fib: {', '.join(grid[:3])}" if grid else ""))
            disc = self.shift_disc(zone, side)
            sig.discount_zone = (round(min(disc), 2), round(max(disc), 2))
        else:
            zone = (round(price, 2), round(price, 2))
            entry = price
            sig.order_type = "MARKET"
            t = self.left_touches((price, price))
            sig.warnings.append("No level zone passed confluence + left-side checks; using market price.")
            if t < cfg.left_min_touches:
                sig.side, sig.order_type = "WAIT", ""
                sig.warnings.append(f"Left-side check failed at {price:.2f} (only {t} prior touches) - thin area, wait for a retest.")
                return sig
        sig.side = side.upper()
        sig.entry_zone, sig.entry = zone, round(entry, 2)
        top, bot = max(entry, zone[1]), min(entry, zone[0])
        if side == "sell":
            sig.stop_loss = round(top + cfg.sl_steps * step, 2)
            tps = [round(entry - m * step, 2) for m in cfg.tp_steps]
        else:
            sig.stop_loss = round(bot - cfg.sl_steps * step, 2)
            tps = [round(entry + m * step, 2) for m in cfg.tp_steps]
        sig.tp1, sig.tp2, sig.tp3 = tps
        sig.big_move = len(conf[side]) == 4
        hs = "s1_high < s3_high" if side == "sell" else "s1_low > s3_low"
        sig.exit_rules = [
            f"TP ladder steps of {step:.1f}: {tps[0]} / {tps[1]} / {tps[2]} (SL {sig.stop_loss}).",
            f"R3 exit: if SMA1 crosses SMA3 back AGAINST the trade ({'SMA1-high above SMA3-high' if side == 'sell' else 'SMA1-low below SMA3-low'}) "
            f"on 15m within the next 3 candles, close the trade.",
            "Close/trim if the 5m/15m/30m/1h confirmations drop below 3.",
        ]
        if sig.big_move:
            sig.reasons.insert(0, "BIG MOVE: 5m, 15m, 30m and 1h all confirm")
        return sig


# =============================================================================
# REPORT
# =============================================================================
def report(eng: Engine, sig: Signal) -> str:
    c = eng.ctx
    L = [f"=== {eng.cfg.symbol} | {sig.time} | price {eng.price:.2f} | offset scale x{eng.cfg.scale:.2f} ===",
         f"Structure trend (+1 up/-1 down): {c.get('trend', {})}",
         f"1h open-price trend (R6): {c.get('open_trend_1h')}   |   R12.2 day/TF trend: {c.get('range_trend')}",
         f"MACD 1h-4h net bias (R2): {c.get('macd_net')}   next range region: {c['region'][0]:.2f} - {c['region'][1]:.2f}"]
        
    if "best_convergence" in c:
        L.append(f"Best MACD convergence entry (R8): {c['best_convergence']}")
    L.append(f"Two-session range / ladder step (R16/17): {c.get('two_session')}")
    L.append(f"Confirmations  BUY: {eng.mtf_confirm('buy')}   SELL: {eng.mtf_confirm('sell')}")
    L.append("-" * 70)
    if sig.side == "WAIT":
        L.append("SIGNAL: WAIT")
    else:
        L += [f"SIGNAL: {sig.side} {sig.order_type}" + ("   ** BIG MOVE **" if sig.big_move else ""),
              f"  entry zone     : {sig.entry_zone[0]} - {sig.entry_zone[1]}   (entry {sig.entry})",
              f"  discount zone  : {sig.discount_zone[0]} - {sig.discount_zone[1]}" if sig.discount_zone != (0.0, 0.0) else "",
              f"  stop loss      : {sig.stop_loss}",
              f"  take profit    : {sig.tp1} / {sig.tp2} / {sig.tp3}",
              f"  confirmations  : {', '.join(sig.confirmations)}   score {sig.score}"]
    for r in sig.reasons:
        L.append(f"  + {r}")
    for r in sig.exit_rules:
        L.append(f"  > {r}")
    for w in sig.warnings:
        L.append(f"  ! {w}")
    return "\n".join(x for x in L if x != "")


def analyse(base: pd.DataFrame, cfg: Config):
    eng = Engine(base, cfg)
    sig = eng.run()
    return eng, sig


def main():
    ap = argparse.ArgumentParser(description="Shuks multi-timeframe signal engine")
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--csv", help="CSV of 5-minute candles (time,open,high,low,close,volume)")
    ap.add_argument("--mt5", help="MT5 symbol(s), comma-separated, e.g. XAUUSD,EURUSD")
    ap.add_argument("--bars", type=int, default=40000)
    ap.add_argument("--auto-scale", action="store_true", help="scale the gold offsets to other symbols via D1 ATR")
    ap.add_argument("--watch", type=int, default=0, help="re-run every N seconds")
    ap.add_argument("--out", help="append signals to this CSV/JSON-lines file")
    a = ap.parse_args()

    def once():
        jobs = []
        if a.demo:
            jobs.append(("XAUUSD-DEMO", demo_data(a.seed)))
        if a.csv:
            jobs.append((a.csv.split("/")[-1].split(".")[0].upper(), load_csv(a.csv)))
        if a.mt5:
            for s in a.mt5.split(","):
                jobs.append((s.strip(), load_mt5(s.strip(), a.bars)))
        if not jobs:
            ap.error("choose --demo, --csv or --mt5")
        for sym, df in jobs:
            cfg = Config(symbol=sym, auto_scale=a.auto_scale or not sym.upper().startswith("XAU"))
            eng, sig = analyse(df, cfg)
            print(report(eng, sig), "\n")
            if a.out and sig.side != "WAIT":
                with open(a.out, "a") as f:
                    f.write(json.dumps(asdict(sig)) + "\n")

    once()
    while a.watch:
        time.sleep(a.watch)
        print("\033[2J\033[H", end="")
        once()


if __name__ == "__main__":
    main()
