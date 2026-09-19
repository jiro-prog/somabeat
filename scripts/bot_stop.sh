#!/bin/bash
# Discord botを確実に停止する
PIDS=$(pgrep -f "python.*discord_bot\.bot" 2>/dev/null)
if [ -z "$PIDS" ]; then
    echo "Bot is not running."
    exit 0
fi
echo "Stopping bot (PIDs: $PIDS)..."
pkill -f "python.*discord_bot\.bot"
sleep 2
REMAINING=$(pgrep -f "python.*discord_bot\.bot" 2>/dev/null)
if [ -n "$REMAINING" ]; then
    echo "Force killing remaining processes: $REMAINING"
    pkill -9 -f "python.*discord_bot\.bot"
    sleep 1
fi
echo "Bot stopped."
