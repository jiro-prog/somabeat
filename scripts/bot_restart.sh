#!/bin/bash
# Discord botを停止→起動する
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/bot_stop.sh" && "$SCRIPT_DIR/bot_start.sh"
