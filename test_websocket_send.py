#!/usr/bin/env python
"""
Test WebSocket connection and send a sample capture result.
Use this to verify the WebSocket server is working before scanning tags.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import asyncio
import websockets
from datetime import datetime


async def test_websocket():
    """Send a test message to WebSocket server."""
    url = "ws://127.0.0.1:2020"
    
    try:
        print(f"Connecting to {url}...")
        async with websockets.connect(url) as websocket:
            print("✓ Connected!")
            
            # Create a sample capture result
            test_payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "tags": ["TEST001"],
                "tag": "TEST001",
                "success": True,
                "number": "99999",
                "attempts": 1,
                "raw_text": "TEST DATA",
                "primary_image_url": "http://127.0.0.1:3000/images/TEST001_primary_test.jpg",
                "secondary_image_url": "http://127.0.0.1:3000/images/TEST001_secondary_test.jpg",
                "label_image_url": "http://127.0.0.1:3000/images/TEST001_label_test.jpg",
                "message": "Test capture - WebSocket working!",
                "errors": []
            }
            
            print(f"\nSending test payload...")
            await websocket.send(json.dumps(test_payload))
            print("✓ Test message sent!")
            
            # Wait for response
            print("Waiting for server response...")
            response = await websocket.recv()
            print(f"✓ Server responded: {response}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure:")
        print("  1. WebSocket server is running (start_system.bat)")
        print("  2. Port 2020 is not blocked by firewall")
        print("  3. No other services are using port 2020")


if __name__ == "__main__":
    print("=" * 60)
    print("WebSocket Connection Test")
    print("=" * 60)
    print()
    
    try:
        asyncio.run(test_websocket())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
