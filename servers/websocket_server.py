#!/usr/bin/env python
"""
WebSocket server listener for capture results.
Listens on ws://0.0.0.0:2020 and logs all incoming payloads.
"""

import sys
import os

# Add parent directory to path so we can import lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import logging
from datetime import datetime

import websockets

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Store all connected clients
connected_clients = set()


async def handle_client(websocket):
    """Handle incoming websocket connection."""
    client_addr = websocket.remote_address
    connected_clients.add(websocket)
    logger.info(f"Client connected: {client_addr} (total clients: {len(connected_clients)})")
    
    try:
        async for message in websocket:
            try:
                payload = json.loads(message)
                logger.info(f"Received from {client_addr}:")
                logger.info(json.dumps(payload, indent=2))
                
                # Broadcast this message to ALL connected clients
                logger.info(f"Broadcasting to {len(connected_clients)} clients...")
                
                # Create a broadcast message
                broadcast_msg = json.dumps({
                    "type": "data",
                    "data": payload,
                    "received_from": str(client_addr),
                    "received_at": datetime.utcnow().isoformat()
                })
                
                # Send to all connected clients
                disconnected = set()
                for client in connected_clients:
                    try:
                        await client.send(broadcast_msg)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected.add(client)
                
                # Remove disconnected clients
                for client in disconnected:
                    connected_clients.discard(client)
                    
                logger.info(f"✓ Broadcast complete (failed: {len(disconnected)})")
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from {client_addr}: {e}")
                logger.error(f"Raw message: {message}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_addr}")
    except Exception as e:
        logger.error(f"Error handling client {client_addr}: {e}")
    finally:
        connected_clients.discard(websocket)
        logger.info(f"Client removed: {client_addr} (remaining: {len(connected_clients)})")


async def main():
    """Start the WebSocket server."""
    host = "0.0.0.0"
    port = 2020
    logger.info(f"Starting WebSocket server on ws://0.0.0.0:{port} (connect with ws://127.0.0.1:{port})")
    logger.info(f"Accessible from: ws://127.0.0.1:{port}")
    logger.info(f"Accessible from network: ws://<your-ip>:{port}")
    
    server = await websockets.serve(handle_client, host, port)
    logger.info(f"WebSocket server is listening on ws://0.0.0.0:{port}")
    logger.info("Server will run indefinitely. Press Ctrl+C to stop.")
    logger.info("✓ Broadcasting enabled: messages from any client sent to all clients.\n")
    
    try:
        # Run forever - this will keep the server alive
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    finally:
        server.close()
        await server.wait_closed()
        logger.info("Server stopped.")


if __name__ == "__main__":
    asyncio.run(main())
