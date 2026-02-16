# Bascol Capture System - Setup Complete ✓

## Folder Structure

```
final/
├── servers/
│   ├── websocket_server.py    # WebSocket server (ws://127.0.0.1:2020)
│   └── image_server.py         # Image HTTP server (http://127.0.0.1:3000)
│
├── demo/
│   └── main.py                 # Main demo app (listens for RFID tags)
│
├── lib/
│   ├── bascol_station.py       # Station logic
│   ├── sangshekan_station.py   # Alternative station
│   ├── common_camera.py        # Camera handling
│   ├── common_config.py        # Config loading
│   ├── common_label_ocr.py     # TrOCR label detection
│   ├── common_rfid.py          # RFID reading
│   └── models.py               # Data models
│
├── images/                     # Auto-created capture storage
├── start_system.bat            # One-click startup
└── requirements.txt            # Python dependencies
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start System
```bash
start_system.bat
```

This launches 3 services in separate windows:
- **WebSocket Server** → ws://127.0.0.1:2020
- **Image Server** → http://127.0.0.1:3000
- **Demo App** → Listens for RFID tags

### 3. Connect with Hoppscotch

1. Open Hoppscotch (https://hoppscotch.io)
2. Create new WebSocket request
3. URL: `ws://127.0.0.1:2020`
4. Click "Connect"
5. Scan RFID tag in reader
6. Watch messages appear in Hoppscotch

## How It Works

```
RFID Tag Detected
    ↓
Demo App (demo/main.py)
    ├─ Captures primary image (camera 1)
    ├─ Captures secondary image (camera 2)
    ├─ Extracts label text (TrOCR)
    ├─ Saves images to ./images/
    └─ Sends JSON to WebSocket
        ↓
    Hoppscotch / Any WebSocket Client
    receives:
    {
        "timestamp": "2026-01-28T10:30:45.123456",
        "tags": ["TAG001"],
        "success": true,
        "number": "12345",
        "attempts": 3,
        "primary_image_url": "http://127.0.0.1:3000/images/TAG001_primary_1769581721123.jpg",
        "secondary_image_url": "http://127.0.0.1:3000/images/TAG001_secondary_1769581721124.jpg",
        "label_image_url": "http://127.0.0.1:3000/images/TAG001_label_1769581721125.jpg"
    }
```

## Server Capabilities

### WebSocket Server (servers/websocket_server.py)
- Listens on `ws://127.0.0.1:2020`
- Receives capture results in real-time
- Echoes confirmation back to client
- Logs all payloads

### Image Server (servers/image_server.py)
- Serves captured images via HTTP
- Access: `http://127.0.0.1:3000/images/`
- Images saved with timestamps for uniqueness

### Demo App (demo/main.py)
- Waits for RFID tag events
- Runs 10 capture attempts per tag
- Collects error details if captures fail
- Sends results to WebSocket

## Environment Variables (Optional)

```bash
# Override default WebSocket address
$env:WEBSOCKET_URL = "ws://192.168.1.100:2020"

# Override image server address
$env:IMAGE_SERVER_URL = "http://192.168.1.100:3000"

python demo/main.py
```

## Troubleshooting

**"Port already in use"**
- Kill process: `netstat -ano | findstr :2020`
- Or change port in servers/websocket_server.py

**"ImportError: No module named lib"**
- Run from the `final/` directory
- Check PYTHONPATH includes current directory

**"Cannot find cameras"**
- Edit demo/main.py and set correct camera addresses
- Default: `primary_cam="192.168.1.3"`, `secondary_cam="192.168.1.201"`

**WebSocket client can't connect**
- Verify `servers/websocket_server.py` is running
- Check firewall allows port 2020
- Try telnet: `telnet 127.0.0.1 2020`

## Files Moved

- `bascol_station.py` → `lib/bascol_station.py`
- `sangshekan_station.py` → `lib/sangshekan_station.py`
- `common_*.py` → `lib/common_*.py`
- `models.py` → `lib/models.py`
- `websocket_server.py` → `servers/websocket_server.py`
- `image_server.py` → `servers/image_server.py`
- `demo_run.py` → `demo/main.py`

## Next Steps

To extend functionality:
1. Add more capture logic to `lib/bascol_station.py`
2. Add more server endpoints to `servers/` folder
3. Add test scripts to `tests/` folder
4. Store configuration in `config/` folder

All imports automatically updated for new structure ✓
