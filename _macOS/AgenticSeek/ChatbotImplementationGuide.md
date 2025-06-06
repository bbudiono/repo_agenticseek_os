# Persistent macOS Chatbot UI/UX Implementation Guide

## Overview

This implementation provides a comprehensive persistent chatbot interface for macOS applications using SwiftUI. The chatbot features smart tagging, autocompletion, and seamless integration with your existing AgenticSeek backend.

## Features Implemented

### ✅ Core UI Components
- **Persistent Sidebar**: Resizable right-hand chatbot panel (280-500px width)
- **Chat Display**: Message bubbles with user/AI distinction, avatars, and timestamps
- **Input Field**: Multi-line text input with auto-resize and placeholder text
- **Smart Controls**: Send button, clear conversation, stop generation

### ✅ Smart Tagging & Autocompletion
- **Trigger Character**: Type `@` to activate smart suggestions
- **Multiple Categories**: Files, folders, app elements, knowledge base, commands, contacts, bookmarks
- **Real-time Filtering**: Suggestions filter as you type after `@`
- **Keyboard Navigation**: Arrow keys and Enter for selection
- **Visual Indicators**: Icons, badges, and descriptions for each suggestion type

### ✅ Advanced Features
- **Code Snippets**: Syntax-highlighted code blocks with copy functionality
- **File Attachments**: File preview integration with QuickLook
- **Accessibility**: Full VoiceOver support and keyboard navigation
- **Error Handling**: Comprehensive error states and recovery
- **Performance**: Debounced search, lazy loading, and efficient rendering

### ✅ Backend Integration Points
- **Mock Service**: Fully functional mock backend for development
- **Real Service**: Integration points for your AgenticSeek backend
- **WebSocket Support**: Ready for real-time communication
- **API Endpoints**: Structured for chat and autocompletion services

## File Structure

```
_macOS/AgenticSeek/
├── ChatbotInterface.swift          # Main chatbot UI components
├── ChatbotModels.swift            # Data models and view models
├── EnhancedContentView.swift      # Integration with existing app
└── ChatbotImplementationGuide.md  # This documentation
```

## Architecture

### Core Components

1. **ChatbotInterface**: Main container with resizable panel
2. **ChatbotPanel**: Complete chatbot interface with header, messages, and input
3. **ChatMessageDisplay**: Scrollable message history with smart navigation
4. **ChatInputField**: Advanced text input with autocompletion integration
5. **AutoCompleteSuggestionsList**: Smart suggestion interface

### Data Models

1. **ChatMessage**: Message structure with content, metadata, and attachments
2. **AutoCompleteSuggestion**: Suggestion data with type, description, and metadata
3. **MessageAttachment**: File attachment with preview capabilities

### View Models

1. **ChatViewModel**: Manages chat state, messages, and backend communication
2. **AutoCompleteManager**: Handles suggestion fetching and display logic

## Integration Guide

### Step 1: Basic Integration

Replace your existing `ContentView` with `EnhancedContentView`:

```swift
@main
struct YourApp: App {
    var body: some Scene {
        WindowGroup {
            EnhancedContentView()
                .frame(minWidth: 1200, minHeight: 800)
        }
    }
}
```

### Step 2: Backend Service Configuration

#### Option A: Use Mock Service (Development)
```swift
let chatViewModel = ChatViewModel(backendService: MockChatbotBackendService())
```

#### Option B: Use Real Backend (Production)
```swift
let backendService = AgenticSeekBackendService()
let chatViewModel = ChatViewModel(backendService: backendService)
```

### Step 3: Custom Backend Implementation

Implement the `ChatbotBackendService` protocol:

```swift
class YourBackendService: ChatbotBackendService {
    func sendMessage(_ message: String) async throws -> String {
        // Your API call to send message
    }
    
    func fetchAutoCompleteSuggestions(query: String, type: AutoCompleteType) async throws -> [AutoCompleteSuggestion] {
        // Your API call to fetch suggestions
    }
    
    // ... other required methods
}
```

## Backend API Specification

### Chat Endpoint

**POST** `/api/chat`

Request:
```json
{
    "message": "User message text",
    "user_id": "unique_user_identifier",
    "conversation_id": "session_identifier"
}
```

Response:
```json
{
    "response": "AI response text",
    "conversation_id": "session_identifier",
    "message_id": "unique_message_id"
}
```

### Autocompletion Endpoint

**GET** `/api/autocomplete?query=text&type=file`

Response:
```json
[
    {
        "displayText": "document.pdf",
        "insertionText": "@Documents/document.pdf",
        "type": "file",
        "description": "PDF document in Documents folder",
        "badge": "PDF",
        "metadata": {
            "path": "/Users/username/Documents/document.pdf",
            "size": "1.2 MB",
            "modified": "2024-06-04T10:30:00Z"
        }
    }
]
```

### Stop Generation Endpoint

**POST** `/api/stop`

Request:
```json
{
    "conversation_id": "session_identifier"
}
```

## Autocompletion Categories

### 1. Files (`AutoCompleteType.file`)
- Local files and documents
- Recent files
- Workspace files
- File metadata (size, type, modification date)

### 2. Folders (`AutoCompleteType.folder`)
- Directory structure
- Common folders (Documents, Downloads, Desktop)
- Project folders
- Workspace directories

### 3. App Elements (`AutoCompleteType.appElement`)
- Application views and screens
- Settings panels
- Menu items
- Feature toggles

### 4. Knowledge Base (`AutoCompleteType.ragItem`)
- RAG system indexed content
- Documentation
- Knowledge articles
- FAQ items

### 5. Commands (`AutoCompleteType.command`)
- Application commands
- System commands
- Custom workflows
- Shortcuts

### 6. Contacts (`AutoCompleteType.contact`)
- Address book contacts
- Team members
- Recent contacts
- Contact groups

### 7. Bookmarks (`AutoCompleteType.bookmark`)
- Browser bookmarks
- Saved links
- Frequently visited sites
- Resource links

## Customization Options

### Visual Customization

1. **Colors**: Modify `DesignSystem.swift` for color scheme
2. **Typography**: Adjust font sizes and weights
3. **Spacing**: Update padding and margins
4. **Icons**: Replace SF Symbols with custom icons

### Behavioral Customization

1. **Trigger Character**: Change from `@` to another character
2. **Suggestion Limits**: Adjust maximum number of suggestions
3. **Debounce Timing**: Modify search delay
4. **Panel Width**: Adjust min/max width constraints

### Example Customizations

```swift
// Custom trigger character
func processInput(_ text: String, cursorPosition: Int) {
    guard let triggerIndex = text.lastIndex(of: "#") else { // Changed from @
        clearSuggestions()
        return
    }
    // ... rest of implementation
}

// Custom suggestion limit
.suggestions = Array(allSuggestions.prefix(15)) // Changed from 10

// Custom debounce timing
try? await Task.sleep(nanoseconds: 500_000_000) // Changed from 300ms
```

## Performance Considerations

### Optimization Strategies

1. **Lazy Loading**: Messages and suggestions loaded on-demand
2. **Debounced Search**: 300ms delay prevents excessive API calls
3. **Result Caching**: Recent suggestions cached for faster response
4. **Efficient Rendering**: LazyVStack for large message lists
5. **Memory Management**: Automatic cleanup of old messages

### Memory Usage

- Message history: Limited to recent 100 messages by default
- Suggestion cache: Cleared after 5 minutes of inactivity
- Image attachments: Lazy loaded and cached efficiently

## Accessibility Features

### VoiceOver Support
- All UI elements properly labeled
- Message content readable by screen reader
- Suggestion navigation with keyboard
- Status announcements for connection state

### Keyboard Navigation
- Tab navigation through all interactive elements
- Enter to send messages
- Arrow keys for suggestion selection
- Escape to dismiss suggestions

### Visual Accessibility
- High contrast support
- Dynamic type support
- Color-independent status indicators
- Focus indicators for keyboard users

## Testing

### Unit Tests
```swift
// Test message sending
func testSendMessage() async {
    let viewModel = ChatViewModel(backendService: MockChatbotBackendService())
    viewModel.currentMessage = "Test message"
    await viewModel.sendMessage()
    
    XCTAssertEqual(viewModel.messages.count, 2) // User + AI response
    XCTAssertTrue(viewModel.messages.first?.isFromUser == true)
}

// Test autocompletion
func testAutoCompletion() async {
    let manager = AutoCompleteManager(backendService: MockChatbotBackendService())
    await manager.processInput("@doc", cursorPosition: 4)
    
    XCTAssertTrue(manager.isActive)
    XCTAssertGreaterThan(manager.suggestions.count, 0)
}
```

### UI Tests
```swift
// Test chatbot visibility
func testChatbotToggle() {
    let app = XCUIApplication()
    app.launch()
    
    let chatbotButton = app.buttons["Show chatbot panel"]
    chatbotButton.tap()
    
    XCTAssertTrue(app.otherElements["Chatbot interface panel"].exists)
}
```

## Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Check backend URL configuration
   - Verify API endpoints are accessible
   - Test with mock service first

2. **Autocompletion Not Working**
   - Ensure `@` character triggers search
   - Check API response format
   - Verify suggestion parsing logic

3. **Performance Issues**
   - Reduce suggestion limit
   - Increase debounce delay
   - Implement result pagination

4. **UI Layout Problems**
   - Check minimum window size
   - Verify constraint conflicts
   - Test with different screen sizes

### Debug Mode

Enable debug logging:

```swift
let chatViewModel = ChatViewModel(backendService: YourBackendService())
chatViewModel.enableDebugLogging = true
```

## Future Enhancements

### Planned Features
1. **Message Threading**: Group related messages
2. **Rich Text Support**: Markdown rendering
3. **Voice Integration**: Voice message support
4. **Collaboration**: Multi-user chat sessions
5. **Plugin System**: Extensible suggestion providers

### Advanced Integrations
1. **Calendar Integration**: Schedule suggestions
2. **Email Integration**: Contact and message suggestions
3. **Project Management**: Task and milestone references
4. **Version Control**: Git branch and commit suggestions

## Conclusion

This chatbot implementation provides a solid foundation for AI-powered user interaction in macOS applications. The modular architecture allows for easy customization and extension, while the comprehensive backend integration points ensure smooth operation with your existing systems.

For questions or support, refer to the inline documentation in each Swift file or contact the development team.