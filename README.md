## Run
create a file called `config.py` then copy paste this code and put ur own
```
BINANCE_API_KEY = 'your binance api key'
BINANCE_API_SECRET = 'your binance api secret'
TELEGRAM_BOT_TOKEN = 'your telegram bot token'
CHAT_ID = your_telegram_id
NEWS_API_KEY= 'your crypto panic news api key'
```

## Creeate a systemd service to run bot continuesly
```
[Unit]
Description=Crypto Telegram Bot
After=network.target

[Service]
User=your_username
WorkingDirectory=/path/to/your/project
ExecStart=/usr/bin/python3 /path/to/your/project/telegram_bot.py
Restart=always
Environment="PYTHONPATH=/path/to/your/project"

[Install]
WantedBy=multi-user.target
```
- enable
```
sudo systemctl daemon-reload
sudo systemctl enable btc_trading_bot
sudo systemctl start btc_trading_bot
```
