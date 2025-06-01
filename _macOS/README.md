# AgenticSeek Native macOS App

This directory contains a native macOS application wrapper for AgenticSeek, providing a true native app experience without needing to use a web browser.

## Features

🖥️ **Native macOS Experience**
- True native macOS application with proper window management
- Menu bar integration with service status indicators
- Native keyboard shortcuts and macOS integrations
- Automatic dark/light mode support

🌐 **Embedded Web Frontend**
- Seamlessly embeds the React frontend using WebKit
- JavaScript bridge for native integrations (notifications, clipboard, etc.)
- Automatic service detection and management

⚙️ **Service Management**
- Automatic Docker service startup and monitoring
- Real-time service status indicators
- Easy service restart functionality
- Background service health checks

🎯 **Privacy Focused**
- Runs completely locally with sandboxing
- Network access only to localhost services
- No external data transmission unless explicitly configured

## Requirements

- macOS 14.0 or later (Sonoma+)
- Xcode 15.4 or later (for building)
- Docker Desktop (for AgenticSeek services)

## Quick Start

### 1. Build the App

```bash
cd _macOS
./build.sh
```

The build script will:
- Clean previous builds
- Build the native macOS app
- Create a `build/` directory with `AgenticSeek.app`
- Provide setup instructions

### 2. Run the App

1. Double-click `build/AgenticSeek.app` to launch
2. The app will automatically check for running services
3. If services aren't running, it will attempt to start them
4. Once ready, you'll see the AgenticSeek interface

## Architecture

### App Components

- **AgenticSeekApp.swift**: Main app entry point with menu bar integration
- **ContentView.swift**: Main window with WebView and loading states
- **WebViewManager.swift**: WebKit management with JavaScript bridge
- **ServiceManager.swift**: Docker service monitoring and management
- **MenuBarManager.swift**: Menu bar functionality

### Native Features

#### JavaScript Bridge
The app provides a JavaScript bridge for React frontend integration:

```javascript
// Available in the React frontend
window.AgenticSeekNative.showNotification(title, message);
window.AgenticSeekNative.openExternal(url);
window.AgenticSeekNative.copyToClipboard(text);
```

#### Keyboard Shortcuts
- `⌘ + N`: New chat
- `⌘ + ⇧ + K`: Clear conversation
- Built-in WebView shortcuts (⌘+R refresh, etc.)

#### Menu Bar Integration
- Real-time service status indicators
- Quick access to restart services
- Show/hide main window
- Quit application

## Development

### Project Structure

```
_macOS/
├── AgenticSeek.xcodeproj/           # Xcode project
├── AgenticSeek/                     # App source code
│   ├── AgenticSeekApp.swift        # Main app
│   ├── ContentView.swift           # Main window
│   ├── WebViewManager.swift        # WebKit integration
│   ├── ServiceManager.swift        # Service management
│   ├── MenuBarManager.swift        # Menu bar functionality
│   ├── Assets.xcassets/            # App icons and assets
│   ├── AgenticSeek.entitlements    # Security permissions
│   └── Info.plist                  # App configuration
├── build.sh                        # Build script
└── README.md                       # This file
```

### Building for Development

1. Open `AgenticSeek.xcodeproj` in Xcode
2. Select the "AgenticSeek" scheme
3. Build and run with `⌘ + R`

### Entitlements

The app requires these entitlements:
- `com.apple.security.network.client`: Connect to localhost services
- `com.apple.security.files.user-selected.read-write`: File access for uploads
- `com.apple.security.automation.apple-events`: AppleScript automation
- `com.apple.security.device.audio-input`: Microphone for voice features

## Service Integration

### Automatic Service Management

The app automatically manages AgenticSeek services:

1. **Health Checking**: Monitors localhost:3000 (frontend) and localhost:8000 (backend)
2. **Auto-Start**: Runs `docker-compose up -d` if services aren't running
3. **Status Display**: Real-time indicators in menu bar and settings
4. **Recovery**: Automatic restart functionality

### Manual Service Control

If you prefer manual control:

```bash
# Start services manually
cd /path/to/agenticseek
docker-compose up -d

# Stop services
docker-compose down

# Check status
docker-compose ps
```

## Troubleshooting

### App Won't Launch
- Check macOS version (requires 14.0+)
- Verify app isn't quarantined: `xattr -dr com.apple.quarantine AgenticSeek.app`

### Services Won't Start
- Ensure Docker Desktop is running
- Check available ports (3000, 8000, 6379)
- Review Docker logs: `docker-compose logs`

### WebView Issues
- Clear WebKit cache: Remove `~/Library/Caches/com.agenticseek.app`
- Check network connectivity to localhost
- Verify React frontend is building correctly

### Build Issues
- Update Xcode to latest version
- Clean derived data: `rm -rf ~/Library/Developer/Xcode/DerivedData`
- Check code signing settings

## Customization

### Changing Service URLs

Edit `WebViewManager.swift` and `ServiceManager.swift` to change default ports:

```swift
// Default URLs
let frontendURL = URL(string: "http://localhost:3000")!
let backendURL = URL(string: "http://localhost:8000")!
```

### Adding Features

The WebView bridge can be extended for new native integrations:

```swift
// Add to WebViewManager
contentController.add(self, name: "newFeature")

// Handle in userContentController
case "newFeature":
    // Handle new feature
```

## Distribution

### For Personal Use
- Build with the provided script
- Copy `AgenticSeek.app` to Applications folder

### For Team Distribution
- Configure code signing in Xcode
- Use Developer ID for distribution outside App Store
- Consider notarization for public distribution

## Contributing

1. Fork the repository
2. Create feature branches for improvements
3. Test thoroughly on different macOS versions
4. Submit pull requests with clear descriptions

## License

This native macOS wrapper follows the same license as the main AgenticSeek project.

---

**Enjoy your native AgenticSeek experience on macOS! 🚀**