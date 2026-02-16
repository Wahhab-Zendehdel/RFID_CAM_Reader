## Quick Reference - WebSocket Capture System

### Start Everything
```bash
.\start_system.bat
```

### Connect with Hoppscotch
1. Open https://hoppscotch.io
2. Create **WebSocket** request
3. URL: `ws://127.0.0.1:2020`
4. Click "Connect"

### Scan Tag
Scan RFID tag in reader → See results in Hoppscotch instantly

### What You'll See
```json
{
  "tags": ["TAG123"],
  "success": true,
  "number": "45678",
  "primary_image_url": "http://127.0.0.1:3000/images/TAG123_primary_1234567890.jpg",
  "secondary_image_url": "http://127.0.0.1:3000/images/TAG123_secondary_1234567891.jpg",
  "label_image_url": "http://127.0.0.1:3000/images/TAG123_label_1234567892.jpg"
}
```

### Folder Structure
```
final/
├── servers/          # WebSocket & Image servers
├── demo/             # Main app (main.py)
├── lib/              # Core logic (bascol_station, cameras, RFID, etc)
└── images/           # Captured images
```

### Ports
- **WebSocket**: 2020
- **Images**: 3000

### That's it!
System runs indefinitely. Press Ctrl+C to stop.
