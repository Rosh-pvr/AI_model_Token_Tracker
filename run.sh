#!/bin/bash
echo "🎯 Multi-Provider Token Tracker"
echo "1. CLI version"
echo "2. Web dashboard"
echo "3. Setup"
read -p "Choice (1-3): " choice

case $choice in
    1) python3 token_tracker.py ;;
    2) python3 dashboard_app.py ;;
    3) python3 setup.py ;;
    *) echo "Invalid" ;;
esac
