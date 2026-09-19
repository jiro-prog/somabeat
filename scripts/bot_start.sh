#!/bin/bash
# Discord botを起動する（二重起動防止付き）
cd /home/jiro/integrated-system

EXISTING=$(pgrep -f "python.*discord_bot\.bot" 2>/dev/null)
if [ -n "$EXISTING" ]; then
    echo "ERROR: Bot is already running (PIDs: $EXISTING). Run bot_stop.sh first."
    exit 1
fi

nohup .venv/bin/python -m discord_bot.bot > /dev/null 2>&1 &
PID=$!
sleep 2

if kill -0 $PID 2>/dev/null; then
    echo "Bot started: PID $PID"
else
    echo "ERROR: Bot failed to start."
    exit 1
fi
