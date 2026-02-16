#!/usr/bin/env python
"""
Simple WebSocket listener - displays all incoming messages.
This will show you exactly what the server is receiving and sending.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
import websockets


async def listen():
    """Connect to WebSocket and display all messages."""
    url = "ws://127.0.0.1:2020"
    
    try:
        print(f"Connecting to {url}...")
        async with websockets.connect(url) as websocket:
            print("✓ Connected! Listening for messages...\n")
            
            try:
                async for message in websocket:
                    print("=" * 70)
                    print("📨 RECEIVED MESSAGE:")
                    print("=" * 70)
                    try:
                        data = json.loads(message)
                        print(json.dumps(data, indent=2))
                    except:
                        print(message)
                    print()
                    
            except KeyboardInterrupt:
                print("\n\nListener stopped by user.")
                
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure WebSocket server is running:")
        print("  Run: start_system.bat")


if __name__ == "__main__":
    print("=" * 70)
    print("WebSocket Message Listener")
    print("=" * 70)
    print("Waiting for messages from the demo or test scripts...")
    print("Press Ctrl+C to stop.\n")
    
    try:
        asyncio.run(listen())
    except KeyboardInterrupt:
        print("\nExiting...")
