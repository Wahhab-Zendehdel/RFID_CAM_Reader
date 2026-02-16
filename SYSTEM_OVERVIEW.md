# Bascol WebSocket + Image Server Integration - COMPLETE

## ✅ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      BASCOL CAPTURE SYSTEM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  RFID Reader (192.168.1.2:6000)                                 │
│       ↓ (Detects tag)                                           │
│  Bascol Demo (demo_run.py)                                      │
│       ├─ Waits for RFID tag                                     │
│       ├─ Attempts up to 10x camera capture                      │
│       ├─ Captures primary camera image                          │
│       ├─ Captures secondary camera image                        │
│       ├─ Detects label via OCR (TrOCR)                          │
│       ├─ Collects multiple tags during capture                  │
│       └─ Saves images to ./images/                              │
│            ↓                                                     │
│       Image Server (image_server.py)                            │
│       ├─ Serves images via HTTP                                 │
│       ├─ Port: 3000                                             │
│       ├─ URL Format: http://127.0.0.1:3000/images/TAG*.jpg     │
│       └─ Available to: WebSocket client, external viewers       │
│            ↓                                                     │
│       WebSocket Client sends result                             │
│       ├─ All tags collected                                     │
│       ├─ Image URLs (not base64)                                │
│       ├─ Success/failure status                                 │
│       ├─ All errors & attempts                                  │
│       └─ OCR detected number                                    │
│            ↓                                                     │
│       WebSocket Server (websocket_server.py)                    │
│       ├─ Listens on ws://127.0.0.1:2020                        │
│       ├─ Logs all payloads (pretty-printed JSON)               │
│       └─ Echoes back confirmation                               │
│            ↓                                                     │
│       Your Application (REST API, Dashboard, etc.)              │
│       ├─ Receives WebSocket payload                             │
│       ├─ Fetches images from Image Server                       │
│       ├─ Processes result (save to DB, etc.)                    │
│       └─ Returns response to client                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Windows (Recommended)
```bash
start_system.bat
```
Opens 3 windows automatically:
1. Image Server (http://127.0.0.1:3000)
2. WebSocket Server (ws://127.0.0.1:2020)
3. Bascol Demo (listens for RFID tags)

### Manual (Any OS)
```bash
# Terminal 1
python image_server.py

# Terminal 2
python websocket_server.py

# Terminal 3
python demo_run.py
```

## 📊 Workflow When RFID Tag Detected

```
1. RFID tag detected (e.g., "TAG001")
   ↓
2. Try up to 10 times:
   - Capture primary camera frame
   - Check for label detection (OCR)
   - If detected, break loop
   - If 10 attempts reached, save last frame
   ↓
3. Capture secondary camera image (backup)
   ↓
4. Save all captured images to disk:
   - images/TAG001_primary_1769581721123.jpg
   - images/TAG001_secondary_1769581721125.jpg
   - images/TAG001_label_1769581721127.jpg
   ↓
5. Collect any additional tags detected during capture
   ↓
6. Build result payload:
   {
     "tags": ["TAG001", "TAG002"],          # All tags found
     "success": true/false,                 # Did we detect?
     "number": "12345",                     # Detected label text
     "errors": [...],                       # Detailed error messages
     "attempts": 5,                         # Number of attempts
     "primary_image_url": "http://...",     # Image URLs
     "secondary_image_url": "http://...",
     "label_image_url": "http://...",
     ... (other metadata)
   }
   ↓
7. Send via WebSocket to ws://127.0.0.1:2020
   ↓
8. Server receives, logs, echoes confirmation
   ↓
9. (Optional) Your backend can:
   - Fetch images from Image Server
   - Save result to database
   - Trigger downstream processing
   - Update UI/Dashboard
```

## 📋 Files Overview

### Core Files
- `demo_run.py` - Main loop: waits for tags, processes captures, sends results
- `bascol_station.py` - Capture logic (10 attempts, error handling, tag collection)
- `common_camera.py` - Camera streaming & frame capture
- `common_rfid.py` - RFID tag reading & debouncing
- `common_label_ocr.py` - OCR/label detection (TrOCR model)

### Server Files
- `image_server.py` - HTTP server on port 3000 (serves ./images/)
- `websocket_server.py` - WebSocket server on port 2020 (receives results)
- `test_websocket.py` - Test script to verify connection

### Configuration
- `start_system.bat` - One-click start on Windows
- `README_WEBSOCKET.md` - Full documentation
- `requirements.txt` - Python dependencies

## 🖼️ Image URLs

### Format
```
http://{IMAGE_SERVER_URL}/images/{TAG}_{TYPE}_{TIMESTAMP}.jpg
```

### Examples
```
http://127.0.0.1:3000/images/TAG001_primary_1769581721123.jpg
http://127.0.0.1:3000/images/TAG001_secondary_1769581721125.jpg
http://127.0.0.1:3000/images/TAG001_label_1769581721127.jpg
```

### Custom Image Server URL
```bash
$env:IMAGE_SERVER_URL="http://192.168.1.100:3000"
python demo_run.py
```

## 📡 WebSocket Payload Example

### Success Case
```json
{
  "timestamp": "2026-01-28T10:50:00.123456",
  "tags": ["TAG001", "TAG002"],
  "tag": "TAG001",
  "tag_source": "SHORT14",
  "tag_timestamp_iso": "2026-01-28T10:50:00Z",
  "started_ts": 1769581721.123,
  "finished_ts": 1769581726.456,
  "success": true,
  "number": "12345",
  "raw_text": "12345",
  "message": "Successfully detected label",
  "errors": [],
  "attempts": 3,
  "primary_image_url": "http://127.0.0.1:3000/images/TAG001_primary_1769581721123.jpg",
  "secondary_image_url": "http://127.0.0.1:3000/images/TAG001_secondary_1769581721125.jpg",
  "label_image_url": "http://127.0.0.1:3000/images/TAG001_label_1769581721127.jpg"
}
```

### Failure Case (After 10 Attempts)
```json
{
  "timestamp": "2026-01-28T10:50:00.123456",
  "tags": ["TAG001"],
  "tag": "TAG001",
  "tag_source": "SHORT14",
  "tag_timestamp_iso": "2026-01-28T10:50:00Z",
  "started_ts": 1769581721.123,
  "finished_ts": 1769581726.456,
  "success": false,
  "number": "",
  "raw_text": "",
  "message": "Max attempts reached (10)",
  "errors": [
    "Max attempts reached (10)",
    "Primary frame not available (attempt 1)",
    "Label detection failed: No label found (attempt 2)",
    "...",
    "Primary camera error: Stream timeout",
    "Secondary frame not available",
    "RFID error: Connection lost"
  ],
  "attempts": 10,
  "primary_image_url": "http://127.0.0.1:3000/images/TAG001_primary_1769581726123.jpg",
  "secondary_image_url": "",
  "label_image_url": ""
}
```

## 🔧 Configuration

### Camera Settings (config.json)
```json
{
  "camera": {
    "host": "192.168.1.3",
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "retry_seconds": 1.0,
    "stale_seconds": 30.0
  },
  "secondary_camera": {
    "host": "192.168.1.201"
  }
}
```

### RFID Settings (demo_run.py)
```python
BascolStation(
    primary_cam="192.168.1.3",
    secondary_cam="192.168.1.201",
    rfid_host="192.168.1.2",      # RFID reader IP
    rfid_port=6000,                # RFID reader port
)
```

### Capture Settings (config.json)
```json
{
  "capture": {
    "retry_interval_seconds": 0.4,
    "timeout_seconds": 30.0,
    "target_digits": 5,
    "max_attempts": 10              # Maximum retry attempts
  }
}
```

### Server URLs (Environment Variables)
```bash
# WebSocket server endpoint
$env:WEBSOCKET_URL="ws://127.0.0.1:2020"

# Image server base URL
$env:IMAGE_SERVER_URL="http://127.0.0.1:3000"

python demo_run.py
```

## 🐛 Troubleshooting

### Images not saving
- Check `./images/` directory exists and is writable
- Check disk space
- Look for "Failed to save" messages in demo output

### Image URLs broken
- Verify Image Server is running: `python image_server.py`
- Check port 3000 is accessible: `curl http://127.0.0.1:3000/`
- Verify IMAGE_SERVER_URL environment variable if custom

### WebSocket not receiving data
- Verify WebSocket Server is running: `python websocket_server.py`
- Check port 2020 is accessible
- Look for connection errors in demo output
- Try `test_websocket.py` to verify connection

### RFID tags not detected
- Check RFID reader hardware
- Verify IP/port in `demo_run.py`
- Check network connectivity to reader
- Look for "Waiting for RFID tag..." message

### Camera frames not captured
- Check cameras are online
- Verify IP addresses in config.json
- Try `common_camera.py` test independently
- Check for timeout messages (>30sec stream timeout)

## 📊 Logs to Monitor

### Image Server Console
```
2026-01-28 10:43:19 [INFO] Image server starting on http://127.0.0.1:3000
2026-01-28 10:43:19 [INFO] Serving images from ./images/
2026-01-28 10:45:29 [INFO] GET /images/TAG001_primary_1769581721123.jpg 200 OK (125KB)
```

### WebSocket Server Console
```
2026-01-28 10:45:49 [INFO] WebSocket server is listening on ws://127.0.0.1:2020
2026-01-28 10:50:00 [INFO] Client connected: ('127.0.0.1', 15943)
2026-01-28 10:50:00 [INFO] Received from ('127.0.0.1', 15943):
{
  "tags": ["TAG001", "TAG002"],
  "success": true,
  ...
}
2026-01-28 10:50:00 [INFO] Client disconnected: ('127.0.0.1', 15943)
```

### Demo Console
```
📍 Bascol Station started. Listening for RFID tags...
⏳ Waiting for RFID tag...
======================================================================
📦 Capture Result (attempt 5):
   Tags: ['TAG001', 'TAG002']
   Success: True
   Number: 12345
   Message: Successfully detected label
======================================================================
  📸 Primary image: http://127.0.0.1:3000/images/TAG001_primary_1769581721123.jpg
  📸 Secondary image: http://127.0.0.1:3000/images/TAG001_secondary_1769581721125.jpg
  📸 Label image: http://127.0.0.1:3000/images/TAG001_label_1769581721127.jpg

🚀 Sending result via WebSocket...
  Connecting to ws://127.0.0.1:2020...
✓ Sent result to ws://127.0.0.1:2020
✓ Result sent!
```

## 🔐 Security Notes

- Image Server has no authentication - restrict network access if needed
- WebSocket has no authentication - add to your backend if required
- Images stored in local `./images/` directory - implement cleanup policy
- No HTTPS/WSS - use in trusted network or behind reverse proxy

## 📈 Performance

- **Memory**: ~500MB (TrOCR model loaded)
- **Network**: ~1-2 Mbps per capture (3 images × 200-400KB each)
- **Processing Time**: 3-5 seconds per tag (depends on camera latency)
- **Images Retention**: Indefinite - implement cleanup in your app

## 🎯 Integration Example

```python
# Your backend listening to WebSocket
import asyncio
import websockets
import json
import requests

async def listen():
    uri = "ws://127.0.0.1:2020"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            result = json.loads(message)
            
            # Download images if needed
            if result['success']:
                img = requests.get(result['primary_image_url']).content
                # Save to DB
                
            # Process result
            process_capture(result)

asyncio.run(listen())
```

---

**System Ready!** 🎉

Start with `start_system.bat` and watch the demo window for results.
