#!/usr/bin/env python
"""
Simple HTTP server to serve captured images.
Serves images from the 'images/' directory on port 3000.
"""

import sys
import os

# Add parent directory to path so we can import lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class ImageRequestHandler(SimpleHTTPRequestHandler):
    """Handle HTTP requests for images."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
            <html>
            <head><title>Bascol Image Server</title></head>
            <body>
                <h1>Bascol Image Server</h1>
                <p>Image server is running on port 3000</p>
                <p>Images are stored in: ./images/</p>
            </body>
            </html>
            """)
            return
        
        # Serve files from images/ directory
        if self.path.startswith('/images/'):
            self.path = '.' + self.path
        
        return super().do_GET()
    
    def log_message(self, format, *args):
        """Custom logging."""
        logger.info(format % args)


def main():
    """Start the image server."""
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    host = '0.0.0.0'
    port = 3000
    
    server_address = (host, port)
    httpd = HTTPServer(server_address, ImageRequestHandler)
    
    logger.info(f"Image server starting on http://{host}:{port}")
    logger.info(f"Serving images from ./images/")
    logger.info(f"Images will be accessible at http://{host}:{port}/images/<filename>")
    logger.info("Server will run indefinitely. Press Ctrl+C to stop.")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Image server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        httpd.server_close()
        logger.info("Image server stopped.")


if __name__ == "__main__":
    main()
