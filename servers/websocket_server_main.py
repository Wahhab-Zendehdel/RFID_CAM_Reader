#!/usr/bin/env python
"""Entry point for running websocket_server as a module."""

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


async def handle_client(websocket):
    """Handle incoming websocket connection."""
    client_addr = websocket.remote_address
    logger.info(f"Client connected: {client_addr}")
    try:
        async for message in websocket:
            try:
                payload = json.loads(message)
                logger.info(f"Received from {client_addr}:")
                logger.info(json.dumps(payload, indent=2))
                # Echo back confirmation
                await websocket.send(json.dumps({"status": "received", "timestamp": datetime.utcnow().isoformat()}))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from {client_addr}: {e}")
                logger.error(f"Raw message: {message}")
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_addr}")
    except Exception as e:
        logger.error(f"Error handling client {client_addr}: {e}")


async def main():
    """Start the WebSocket server."""
    host = "0.0.0.0"
    port = 2020
    logger.info(f"Starting WebSocket server on 0.0.0.0:{port} (connect with ws://127.0.0.1:{port})")
    logger.info(f"Accessible from: ws://127.0.0.1:{port}")
    logger.info(f"Accessible from network: ws://<your-ip>:{port}")
    
    server = await websockets.serve(handle_client, host, port)
    logger.info(f"WebSocket server is listening on ws://0.0.0.0:{port}")
    logger.info("Server will run indefinitely. Press Ctrl+C to stop.")
    
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
