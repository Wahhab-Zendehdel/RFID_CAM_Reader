# Bascol Station with WebSocket Integration

## Overview
This system captures RFID tags and attempts label detection via camera with automatic 10-attempt retry logic. Results (including all tags, errors, and image URLs) are sent via WebSocket to a server listening on `ws://127.0.0.1:2020`.

Images are saved to disk and served via HTTP on `http://127.0.0.1:3000/images/`

## Quick Start

### Option 1: Batch File (Recommended on Windows)
```bash
start_system.bat
```
This opens three windows:
1. **Image Server** - Serves captured images on HTTP
2. **WebSocket Server** - Listens for results
3. **Bascol Demo** - Waits for RFID tags

### Option 2: Manual Terminal Windows

**Terminal 1 - Start Image Server:**
```bash
python image_server.py
```

**Terminal 2 - Start WebSocket Server:**
```bash
python websocket_server.py
```

**Terminal 3 - Start Demo (waits for RFID tags):**
```bash
python demo_run.py
```

## How It Works

### When an RFID tag is detected:
1. **Tag detection** - RFID reader sends tag
2. **Capture attempts** - Try up to 10 times to detect label via primary camera
3. **Secondary capture** - Capture image from secondary camera (if available)
4. **Label detection** - OCR on detected label (if success)
5. **Image saving** - Save images to `./images/` directory
6. **Result sent** - Send complete result to WebSocket server:
   - All tags collected during capture
   - Success/failure status
   - Detected number from label
   - Primary camera image URL (http://127.0.0.1:3000/images/TAG001_primary_*.jpg)
   - Secondary camera image URL
   - Label detection image URL
   - All errors encountered
   - Number of attempts made

## Configuration

### Camera Configuration
Edit `config.json`:
```json
{
  "camera": {
    "host": "192.168.1.3",
    "width": 1920,
    "height": 1080
  },
  "secondary_camera": {
    "host": "192.168.1.201"
  }
}
```

### RFID Configuration
Update in `demo_run.py`:
```python
rfid_host="192.168.1.2",  # RFID reader IP
rfid_port=6000,            # RFID reader port
```

### WebSocket Server
- Default: `ws://127.0.0.1:2020`
- Override: Set `WEBSOCKET_URL` environment variable

### Image Server
- Default: `http://127.0.0.1:3000`
- Override: Set `IMAGE_SERVER_URL` environment variable
- Images saved to: `./images/`
- Filename format: `{TAG}_{TYPE}_{TIMESTAMP}.jpg`
  - Example: `TAG001_primary_1769581721123.jpg`

```bash
# Override URLs
$env:WEBSOCKET_URL="ws://192.168.1.100:2020"
$env:IMAGE_SERVER_URL="http://192.168.1.100:3000"
python demo_run.py
```

## Result Payload Format

```json
{
  "timestamp": "2026-01-28T10:30:00.123456",
  "tags": ["TAG001", "TAG002"],
  "tag": "TAG001",
  "tag_source": "SHORT14",
  "tag_timestamp_iso": "2026-01-28T10:30:00Z",
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

## Error Scenarios

### Camera not found (After 10 attempts):
```json
{
  "success": false,
  "errors": [
    "Max attempts reached (10)",
    "Primary camera error: Failed to open stream",
    "Secondary frame not available"
  ],
  "primary_image_base64": "<empty or last frame>"
}
```

### No label detected (After 10 attempts):
```json
{
  "success": false,
  "errors": [
    "Max attempts reached (10)",
    "Primary frame not available (attempt 1)",
    "... attempt messages ..."
  ]
}
```

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `websockets>=12.0` - WebSocket server
- `websocket-client>=1.6.0` - WebSocket client
- `torch>=2.0.0` - TrOCR model
- `transformers>=4.30.0` - Model loading
- `opencv-python>=4.8.0` - Camera capture

## Testing

### Test WebSocket Connection:
```bash
# Start server first
python websocket_server.py

# In another terminal:
python test_websocket.py
```

This sends a dummy payload to verify the connection works.

## Troubleshooting

### "Connection refused" on WebSocket
- Make sure `websocket_server.py` is running first
- Check firewall settings
- Verify `ws://127.0.0.1:2020` is accessible

### Camera timeouts
- Check network connectivity to camera hosts
- Verify IP addresses in config.json
- Check camera is not in use by another process

### No result sent on tag detection
- Check WebSocket server is running and listening
- Look at server logs for "Client connected" message
- Verify camera hardware is responding
- Check RFID reader is sending tags correctly

### Logs

### Image Server
Shows:
- Server start message
- HTTP requests for images
- File serving status

### WebSocket Server
Shows:
- Client connections
- Received payloads (pretty-printed JSON)
- Connection errors

### Bascol Demo
Shows:
- Tag detection ("⏳ Waiting for RFID tag...")
- Captured image URLs
- Capture attempts
- Result sent confirmation ("✓ Result sent!")
- Errors if any

## License
Internal project - Siman
