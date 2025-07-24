import websockets
import json
import datetime
import httpx
import asyncio
import logging
import pandas as pd
from telegram import Update, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from telegram.helpers import escape_markdown
from telegram.constants import ChatAction
from binance.client import Client
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from config import TELEGRAM_BOT_TOKEN, BINANCE_API_KEY, BINANCE_API_SECRET, CHAT_ID, NEWS_API_KEY
from strategy import bollinger_strategy, ema_strategy, rsi_strategy, macd_strategy, stochastic_strategy, format_price
import collections

client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def escape_markdown_v2(text):
    escape_chars = '_*[]()~`>#+-=|{}.!'
    return ''.join('\\' + char if char in escape_chars else char for char in text)

async def monitor_multiple_volume_ws(app, symbols, interval="1m"):
    streams = [f"{s.lower()}@kline_{interval}" for s in symbols]
    uri = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

    # Use a rolling deque per symbol
    volume_history = {s: collections.deque(maxlen=30) for s in symbols}

    async with websockets.connect(uri) as ws:
        await app.bot.send_message(CHAT_ID, f"📡 Monitoring: {', '.join(symbols)}")

        async for message in ws:
            try:
                msg = json.loads(message)
                stream = msg["stream"]
                data = msg["data"]
                kline = data["k"]
                symbol = kline["s"]  # This is uppercased already
                is_closed = kline["x"]

                if not is_closed:
                    continue

                vol = float(kline["v"])
                history = volume_history[symbol]
                history.append(vol)

                if len(history) < 30:
                    continue

                avg_vol = sum(history) / len(history)
                vol_ratio = vol / avg_vol

                if vol_ratio > 2.5:
                    alert = (
                        f"*⚠️ Volume Spike Detected*\n"
                        f"{escape_markdown(symbol, 2)}\n"
                        f"Last Vol: {vol:,.2f}\n"
                        f"Avg Vol (30m): {avg_vol:,.2f}\n"
                        f"Ratio: {vol_ratio:.2f}"
                    )
                    await app.bot.send_message(CHAT_ID, alert, parse_mode="MarkdownV2")

            except Exception as e:
                logger.warning(f"WebSocket error: {e}")

def get_top_symbols(base_asset='USDT', top_n=10):
    info = client.get_exchange_info()
    symbols = [
        s['symbol'] for s in info['symbols']
        if s['status'] == 'TRADING' and s['quoteAsset'] == base_asset
    ]
    tickers = client.get_ticker()
    volume_map = {t['symbol']: float(t['quoteVolume']) for t in tickers if t['symbol'] in symbols}
    sorted_symbols = sorted(volume_map, key=volume_map.get, reverse=True)
    return sorted_symbols[:top_n]

SYMBOL_LIST = get_top_symbols()

def fetch_data(symbol='BTCUSDT', interval='1d', limit=90):
    klines = client.get_klines(symbol=symbol.upper(), interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def analyze_single_timeframe(df):
    """Analyze one timeframe and return (signal_score, reasons, last_close, signal_text)"""
    df['50_ma'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['200_ma'] = EMAIndicator(df['close'], window=200).ema_indicator()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()

    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    bb = BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    reasons = []
    buy_score = 0
    sell_score = 0

    if last['50_ma'] > last['200_ma']:
        reasons.append("📈 Trend: Bullish (50 MA > 200 MA)")
        buy_score += 1
    else:
        reasons.append("📉 Trend: Bearish (50 MA < 200 MA)")
        sell_score += 1

    if last['rsi'] < 30:
        reasons.append("🟢 RSI < 30 (Oversold)")
        buy_score += 1
    elif last['rsi'] > 70:
        reasons.append("🔴 RSI > 70 (Overbought)")
        sell_score += 1

    if last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']:
        reasons.append("🟢 MACD Bullish Crossover")
        buy_score += 1
    elif last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']:
        reasons.append("🔴 MACD Bearish Crossover")
        sell_score += 1

    if last['close'] < last['bb_lower']:
        reasons.append("🟢 Price below Bollinger Band (Oversold)")
        buy_score += 1
    elif last['close'] > last['bb_upper']:
        reasons.append("🔴 Price above Bollinger Band (Overbought)")
        sell_score += 1

    score = buy_score - sell_score
    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    return score, reasons, last['close'], signal

def combine_signals(daily, four_hour):
    """Combine scores from two timeframes for stronger signals."""
    score = daily[0] + four_hour[0]  # simple sum of scores
    reasons = daily[1] + four_hour[1]
    price = daily[2]
    signal = None

    if score >= 3:
        signal = "🟢 STRONG BUY"
    elif score == 2:
        signal = "🟢 BUY"
    elif score == 1:
        signal = "🟡 WEAK BUY"
    elif score == 0:
        signal = "🟡 HOLD"
    elif score == -1:
        signal = "🟠 WEAK SELL"
    elif score == -2:
        signal = "🔴 SELL"
    else:
        signal = "🔴 STRONG SELL"

    return signal, reasons, price

async def suggest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != CHAT_ID:
        await update.message.reply_text("🚫 You are not authorized to use this bot.")
        return

    await update.message.chat.send_action(action=ChatAction.TYPING)
    try:
        results = []
        for symbol in SYMBOL_LIST:
            try:
                df_daily = fetch_data(symbol, '1d')
                df_4h = fetch_data(symbol, '4h', limit=60)
                daily_analysis = analyze_single_timeframe(df_daily)
                four_hour_analysis = analyze_single_timeframe(df_4h)
                signal, _, price = combine_signals(daily_analysis, four_hour_analysis)
                results.append((symbol, signal, price))
            except Exception as e:
                logger.warning(f"Skipping {symbol} due to error: {e}")

        results.sort(key=lambda x: x[2], reverse=True)

        table_lines = []
        for symbol, signal, _ in results:
            table_lines.append(f"{symbol.ljust(12)}: {signal}")

        # Send preformatted block (monospaced, no escaping needed inside)
        message = "*🧠 Suggested Coins to Watch:*\n\n"
        message += "```text\n"
        message += "\n".join(table_lines)
        message += "\n```\n"
        message += "_DYOR_"

        await update.message.reply_text(message, parse_mode="MarkdownV2")

    except Exception as e:
        logger.error(f"Error in suggest command: {e}")
        await update.message.reply_text("❌ Failed to generate suggestions.")

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != CHAT_ID:
        await update.message.reply_text("🚫 Unauthorized.")
        return

    coin = context.args[0].upper() if context.args else None

    if coin:
        url = (
            f"https://cryptopanic.com/api/developer/v2/posts/?auth_token={NEWS_API_KEY}"
            f"&currencies={coin}&kind=news&public=true"
        )
        header_msg = f"*📰 Latest news for {escape_markdown(coin, 2)}:*\n\n"
    else:
        url = (
            f"https://cryptopanic.com/api/developer/v2/posts/?auth_token={NEWS_API_KEY}"
            f"&kind=news&public=true&filter=hot"
        )
        header_msg = "*🔥 Hot Crypto News:*\n\n"

    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        if r.status_code != 200:
            await update.message.reply_text(f"Failed to fetch news (status {r.status_code})")
            return
        data = r.json()

    if "results" not in data or not data["results"]:
        await update.message.reply_text("No news found.")
        return

    msg = header_msg
    for post in data["results"][:5]:
        title = escape_markdown(post.get("title", "No title"), 2)
        post_url = post.get("url", "")
        source = escape_markdown(post.get("source", {}).get("title", "Unknown source"), 2)

        published_at = post.get("published_at")
        if published_at:
            dt = datetime.datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            published_str = escape_markdown(dt.strftime("%Y-%m-%d %H:%M UTC"), 2)
        else:
            published_str = "Unknown date"

        # Escape literal dash in markdown with backslash \-
        # Wrap the entire italic block content with escape_markdown already done
        # but since dash is literal, we add a backslash before it explicitly.
        msg += f"• [{title}]({post_url})\n  _{source} \\- {published_str}_\n\n"

    await update.message.reply_text(msg, parse_mode="MarkdownV2", disable_web_page_preview=True)

async def targets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != CHAT_ID:
        await update.message.reply_text("🚫 You are not authorized to use this bot.")
        return

    # Send typing action asynchronously
    await update.message.chat.send_action(action=ChatAction.TYPING)

    # Parse symbol
    try:
        symbol = context.args[0].upper() + "USDT"
    except IndexError:
        await update.message.reply_text("❌ Usage: /targets BTC")
        return

    try:
        # Fetch current price
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker["price"])

        # Shared data
        df = fetch_data(symbol, "1d", limit=50)

        # Run strategies and collect targets
        strategies = [
            ("Bollinger Bands", bollinger_strategy),
            ("EMA Crossover", ema_strategy),
            ("RSI", rsi_strategy),
            ("MACD", macd_strategy),
            ("Stochastic", stochastic_strategy)
        ]
        
        buy_targets = []
        sell_targets = []
        strategy_messages = []
        
        for name, strategy_func in strategies:
            msg, buy, sell = strategy_func(df, current_price)
            strategy_messages.append(msg)
            buy_targets.append(buy)
            sell_targets.append(sell)
            
        # Calculate average targets
        avg_buy = sum(buy_targets) / len(buy_targets)
        avg_sell = sum(sell_targets) / len(sell_targets)
        
        # Format summary
        summary = (
            f"*📊 Summary for {escape_markdown(symbol, version=2)}*\n\n"
            f"💰 Current Price: `{format_price(current_price)}`\n\n"
            f"🔹 *Recommended Targets*\n"
            f"🟢 Buy: `{format_price(avg_buy)}`\n"
            f"🔴 Sell: `{format_price(avg_sell)}`\n\n"
            f"*Based on {len(strategies)} strategies*\n\n"
        )
        
        # Build detailed strategy section
        details = "*🔍 Strategy Details*\n\n" + "\n\n".join(strategy_messages)
        
        # Combine all parts
        msg = summary + details
        
        # Replace placeholder formatting
        msg = msg.replace("[dot]", ".").replace("[comma]", ",")

        await update.message.reply_text(msg, parse_mode="MarkdownV2")

    except Exception as e:
        logger.error(f"Error in targets for {symbol}: {e}")
        error_msg = escape_markdown(f"❌ Failed to get targets for {symbol}\n{str(e)}", version=2)
        await update.message.reply_text(error_msg, parse_mode="MarkdownV2")

async def trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != CHAT_ID:
        await update.message.reply_text("🚫 Unauthorized.")
        return

    try:
        # Parse arguments
        args = context.args
        if len(args) < 2:
            await update.message.reply_text("❌ Usage: /trade <symbol> <timeframe>\nExample: /trade btc 15m")
            return
            
        symbol = args[0].upper() + "USDT"
        timeframe = args[1].lower()
        
        # Validate timeframe
        valid_timeframes = ['15m', '30m', '1h']
        if timeframe not in valid_timeframes:
            await update.message.reply_text(f"❌ Invalid timeframe. Choose from: {', '.join(valid_timeframes)}")
            return

        await update.message.chat.send_action(action=ChatAction.TYPING)
        
        # Fetch data based on timeframe
        limit = 100  # Enough for indicators
        df = fetch_data(symbol, timeframe, limit)
        
        if df.empty:
            await update.message.reply_text(f"❌ No data found for {symbol} on {timeframe} timeframe")
            return
            
        # Calculate indicators
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['ema20'] = EMAIndicator(df['close'], window=20).ema_indicator()
        
        # Get current values
        current = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = current['close']
        
        # Generate signals
        signals = []
        
        # Bollinger Bands strategy
        if current_price < current['bb_lower']:
            signals.append("🟢 Price below lower Bollinger Band (potential reversal)")
        elif current_price > current['bb_upper']:
            signals.append("🔴 Price above upper Bollinger Band (potential reversal)")
            
        # RSI strategy
        if current['rsi'] < 30:
            signals.append("🟢 RSI < 30 (oversold)")
        elif current['rsi'] > 70:
            signals.append("🔴 RSI > 70 (overbought)")
            
        # Stochastic strategy
        if current['stoch_k'] < 20 and current['stoch_d'] < 20:
            signals.append("🟢 Stochastic in oversold territory")
        elif current['stoch_k'] > 80 and current['stoch_d'] > 80:
            signals.append("🔴 Stochastic in overbought territory")
        if current['stoch_k'] > current['stoch_d'] and prev['stoch_k'] <= prev['stoch_d']:
            signals.append("🟢 Stochastic bullish crossover")
        elif current['stoch_k'] < current['stoch_d'] and prev['stoch_k'] >= prev['stoch_d']:
            signals.append("🔴 Stochastic bearish crossover")
            
        # EMA strategy
        if current_price > current['ema20']:
            signals.append("🟢 Price above 20 EMA (bullish bias)")
        else:
            signals.append("🔴 Price below 20 EMA (bearish bias)")
            
        # Determine overall bias
        buy_signals = sum(1 for s in signals if '🟢' in s)
        sell_signals = sum(1 for s in signals if '🔴' in s)
        
        if buy_signals > sell_signals:
            bias = "BULLISH"
            emoji = "🟢"
        elif sell_signals > buy_signals:
            bias = "BEARISH"
            emoji = "🔴"
        else:
            bias = "NEUTRAL"
            emoji = "🟡"
            
        # Calculate entry/exit targets (1% risk/reward ratio)
        volatility = (df['high'].iloc[-10:].max() - df['low'].iloc[-10:].min()) / current_price
        risk_percentage = max(0.5, min(2.0, volatility * 100))  # Dynamic risk based on volatility
        
        if bias == "BULLISH":
            entry = current_price * 0.995  # Slightly below current price
            stop_loss = entry * (1 - risk_percentage/100)
            take_profit1 = entry * (1 + risk_percentage/100)
            take_profit2 = entry * (1 + risk_percentage*2/100)
        elif bias == "BEARISH":
            entry = current_price * 1.005  # Slightly above current price
            stop_loss = entry * (1 + risk_percentage/100)
            take_profit1 = entry * (1 - risk_percentage/100)
            take_profit2 = entry * (1 - risk_percentage*2/100)
        else:  # Neutral
            entry = current_price
            stop_loss = entry * 0.99
            take_profit1 = entry * 1.01
            take_profit2 = entry * 1.02
            
        symbol_esc = escape_markdown(symbol, version=2)
        timeframe_esc = escape_markdown(timeframe.upper(), version=2)
        bias_esc = escape_markdown(bias, version=2)
        
        # Format prices with actual decimal points
        def format_price_fixed(price):
            """Format price with actual decimal point and commas"""
            return f"{price:,.2f}".replace(",", "\\,")  # Only escape commas
        
        current_price_str = format_price_fixed(current_price)
        entry_str = format_price_fixed(entry)
        stop_loss_str = format_price_fixed(stop_loss)
        take_profit1_str = format_price_fixed(take_profit1)
        take_profit2_str = format_price_fixed(take_profit2)
        
        # Build the message with proper MarkdownV2 formatting
        msg = (
            f"{emoji} *{bias_esc} BIAS TRADE SETUP FOR {symbol_esc} \\({timeframe_esc}\\)*\n\n"
            f"💰 Current Price: `{current_price_str}`\n\n"
            "🎯 *ENTRY*: " f"`{entry_str}`\n"
            "❌ *STOP LOSS*: " f"`{stop_loss_str}`\n"
            "✅ *TAKE PROFIT 1*: " f"`{take_profit1_str}`\n"
            "🚀 *TAKE PROFIT 2*: " f"`{take_profit2_str}`\n\n"
            f"*SIGNALS \\({len(signals)}\\):*\n"
        )

        # Add signals
        for i, signal in enumerate(signals, 1):
            signal_esc = escape_markdown(signal, version=2)
            msg += f"{i}\\. {signal_esc}\n"
            
        # Add risk management note
        msg += "\n_Risk management\\: 1\\:2 risk\\-reward ratio\\. Adjust based on your risk tolerance\\._"

        await update.message.reply_text(msg, parse_mode="MarkdownV2")
        
    except Exception as e:
        logger.error(f"Error in trade command: {e}")
        error_msg = escape_markdown(f"❌ Error generating trade: {str(e)}", version=2)
        await update.message.reply_text(error_msg, parse_mode="MarkdownV2")


def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("suggest", suggest))
    app.add_handler(CommandHandler("news", news))
    app.add_handler(CommandHandler("target", targets))
    app.add_handler(CommandHandler("trade", trade))
    commands = [
        BotCommand("suggest", "Suggest top coins to watch"),
        BotCommand("news", "Fetch recent articles with the coin name or symbol keyword"),
        BotCommand("target", "Get buy/sell targets for a coin"),
        BotCommand("trade", "Get trade setup for specific timeframe"),
    ]
    async def post_init(app):
        await set_commands(app)  # call the helper
        coins_to_watch = ["BTCUSDT", "WCTUSDT", "SFPUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]
        asyncio.create_task(monitor_multiple_volume_ws(app, coins_to_watch))

    async def set_commands(app):
        await app.bot.set_my_commands(commands)

    app.post_init = post_init  # <- single entry point
    print("✅ Bot is running")
    app.run_polling()

if __name__ == "__main__":
    main()
