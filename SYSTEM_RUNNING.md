# System Running Successfully! ✓

## What Just Happened
The batch script successfully:
1. ✓ Started WebSocket Server (ws://127.0.0.1:2020) in a new window
2. ✓ Started Image Server (http://127.0.0.1:3000) in a new window  
3. ✓ Started Demo App listening for RFID tags in the main window
4. ✓ All 3 services running simultaneously

## Next Steps - Connect with Hoppscotch

### Option 1: Use Hoppscotch (Recommended)
1. Go to https://hoppscotch.io
2. Select **WebSocket** as request type
3. Enter URL: `ws://127.0.0.1:2020`
4. Click **Connect**
5. You'll see "Connected" message
6. Scan an RFID tag in the reader
7. Capture results will appear instantly in Hoppscotch!

### Option 2: Use WebSocket Client (VS Code Extension)
1. Install "WebSocket Client" extension in VS Code
2. Connect to: `ws://127.0.0.1:2020`
3. Scan tags and watch results come through

## What You'll See in WebSocket

When you scan a tag, you'll receive JSON like:
```json
{
  "timestamp": "2026-01-28T10:30:45.123456",
  "tags": ["TAG001"],
  "tag": "TAG001",
  "success": true,
  "number": "12345",
  "attempts": 3,
  "raw_text": "detected label text",
  "primary_image_url": "http://127.0.0.1:3000/images/TAG001_primary_1769581721123.jpg",
  "secondary_image_url": "http://127.0.0.1:3000/images/TAG001_secondary_1769581721124.jpg",
  "label_image_url": "http://127.0.0.1:3000/images/TAG001_label_1769581721125.jpg",
  "message": "Capture successful"
}
```

## Image URLs Work Too!

You can click on any image URL to view:
- http://127.0.0.1:3000/images/TAG001_primary_1769581721123.jpg
- http://127.0.0.1:3000/images/TAG001_secondary_1769581721124.jpg
- http://127.0.0.1:3000/images/TAG001_label_1769581721125.jpg

## Running Again

Simply run: `.\start_system.bat`

All servers start in new windows. Click on any window to see logs/debug info.

## Stopping

Press Ctrl+C in any window to stop that service, or close the window.

## Troubleshooting

**"Cannot connect to WebSocket"**
- Check that WebSocket server window shows "listening"
- Verify firewall allows port 2020

**"No image URLs in results"**
- Check Image Server window is running
- Check ./images/ folder exists

**"Cannot get RFID tags"**
- Verify RFID reader is connected on 192.168.1.2:6000
- Check cameras are accessible (192.168.1.3, 192.168.1.201)
- Edit camera addresses in demo/main.py if different
