from telegram.helpers import escape_markdown
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import StochasticOscillator
import pandas as pd

def format_price(price: float) -> str:
    return f"${price:,.2f}".replace(",", "[comma]").replace(".", "[dot]")

def bollinger_strategy(df, current_price):
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_lower = bb.bollinger_lband().iloc[-1]

    buy = bb_lower * 0.95
    sell = bb_upper * 1.05

    msg = (
        f"📊 *Bollinger Bands*\n"
        f"🟢 Buy: `{format_price(buy)}`\n"
        f"🔴 Sell: `{format_price(sell)}`"
    )
    return msg, buy, sell

def ema_strategy(df, current_price):
    ema_12 = df['close'].ewm(span=12, adjust=False).mean().iloc[-1]
    ema_26 = df['close'].ewm(span=26, adjust=False).mean().iloc[-1]

    trend = "Bullish" if ema_12 > ema_26 else "Bearish"
    buy = current_price * (0.98 if trend == "Bullish" else 0.93)
    sell = current_price * (1.05 if trend == "Bullish" else 1.01)

    trend_md = escape_markdown(trend, version=2)

    msg = (
        f"📊 *EMA Crossover*\n"
        f"📈 Trend: `{trend_md}`\n"
        f"🟢 Buy: `{format_price(buy)}`\n"
        f"🔴 Sell: `{format_price(sell)}`"
    )
    return msg, buy, sell

def rsi_strategy(df, current_price):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]

    if rsi < 30:
        zone = "Oversold"
        buy = current_price * 0.97
        sell = current_price * 1.06
    elif rsi > 70:
        zone = "Overbought"
        buy = current_price * 0.92
        sell = current_price * 1.01
    else:
        zone = "Neutral"
        buy = current_price * 0.95
        sell = current_price * 1.05

    rsi_header = escape_markdown(f"RSI ({rsi:.1f}) - {zone}", version=2)

    msg = (
        f"📊 *{rsi_header}*\n"
        f"🟢 Buy: `{format_price(buy)}`\n"
        f"🔴 Sell: `{format_price(sell)}`"
    )
    return msg, buy, sell

def macd_strategy(df, current_price):
    macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd.macd().iloc[-1]
    signal_line = macd.macd_signal().iloc[-1]
    histogram = macd.macd_diff().iloc[-1]
    
    # Determine market phase
    if macd_line > signal_line and histogram > 0:
        trend = "Bullish"
        buy = current_price * 0.98
        sell = current_price * 1.05
    elif macd_line < signal_line and histogram < 0:
        trend = "Bearish"
        buy = current_price * 0.93
        sell = current_price * 1.01
    else:
        trend = "Neutral"
        buy = current_price * 0.95
        sell = current_price * 1.03

    msg = (
        f"📊 *MACD Crossover*\n"
        f"📈 Trend: `{escape_markdown(trend, version=2)}`\n"
        f"🟢 Buy: `{format_price(buy)}`\n"
        f"🔴 Sell: `{format_price(sell)}`"
    )
    return msg, buy, sell

def stochastic_strategy(df, current_price):
    stoch = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14,
        smooth_window=3
    )
    k_line = stoch.stoch().iloc[-1]
    d_line = stoch.stoch_signal().iloc[-1]
    
    # Generate signals based on oscillator position
    if k_line < 20 and d_line < 20:
        zone = "Oversold"
        buy = current_price * 0.97
        sell = current_price * 1.06
    elif k_line > 80 and d_line > 80:
        zone = "Overbought"
        buy = current_price * 0.92
        sell = current_price * 1.01
    else:
        zone = "Neutral"
        buy = current_price * 0.95
        sell = current_price * 1.04

    # Format values for display
    k_val = escape_markdown(f"K: {k_line:.1f}", version=2)
    d_val = escape_markdown(f"D: {d_line:.1f}", version=2)
    zone_md = escape_markdown(zone, version=2)

    msg = (
        f"📊 *Stochastic Oscillator*\n"
        f"⚖️ Zone: `{zone_md}`\n"
        f"📊 Values: `{k_val}, {d_val}`\n"
        f"🟢 Buy: `{format_price(buy)}`\n"
        f"🔴 Sell: `{format_price(sell)}`"
    )
    return msg, buy, sell
