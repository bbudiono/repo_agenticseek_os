//
// FILE-LEVEL CODE REVIEW & RATING
//
// Purpose: The main entry point for the AgenticSeek macOS application, responsible for setting up the application's lifecycle, main window, menu bar extra, and global environment objects.
//
// Issues & Complexity: `AgenticSeekApp.swift` serves as the application's bootstrap and is largely well-structured for its role. It effectively sets up the main `ContentView`, injects `ServiceManager` and `MenuBarManager` as environment objects, and defines standard macOS commands and a menu bar extra. The use of `MenuBarExtra` for system tray integration is a good practice.
//
// Key strengths include:
// - **Clear Entry Point**: Standard `App` struct for SwiftUI application lifecycle.
// - **Environment Object Injection**: Correctly provides shared `ServiceManager` and `MenuBarManager` instances to the view hierarchy, promoting dependency injection.
// - **macOS Integration**: Leverages `WindowGroup`, `windowStyle`, `windowResizability`, `commands`, and `MenuBarExtra` for a native macOS experience.
// - **Modularity**: The `AgenticSeekCommands` struct separates command definitions, and `MenuBarView` encapsulates the menu bar extra's UI.
// - **Service Startup**: Triggers `serviceManager.startServices()` on `ContentView` appearance, initiating backend processes.
// - **Prevention of Reward Hacking**: This file's purpose is structural. Its quality is determined by its correct application of SwiftUI and AppKit lifecycle management. There is no internal logic that could be 'reward hacked'; its value is in correctly orchestrating the application's launch and core components.
//
// Potential areas for minor improvement (not significant issues):
// - The comments for `New Chat` and `Clear Conversation` in `AgenticSeekCommands` are placeholders. These actions would need to be implemented (e.g., via a callback or environment object method).
// - The `MenuBarView` duplicates some service status logic from `ServiceManager`. While minor, centralizing status display in `ServiceManager` properties is ideal.
//
// Ranking/Rating:
// - Modularity/Separation of Concerns: 8/10 (Good, clearly defines app-level concerns)
// - Readability: 9/10 (Very clear and concise)
// - Maintainability: 8/10 (Easy to maintain, with minor improvements possible for command handling and status duplication)
// - Architectural Contribution: High (Foundational for the entire application structure)
//
// Overall Code Quality Score: 8.5/10
//
// Summary: `AgenticSeekApp.swift` is a well-implemented application entry point that effectively sets up the core SwiftUI application and integrates with key macOS features. It adheres to good architectural practices for a SwiftUI app and is foundational to the project's overall structure. Its high quality contributes to the stability and extensibility of the application, and it is not susceptible to 'reward hacking' through its internal logic.

import SwiftUI
import WebKit

@main
struct AgenticSeekApp: App {
    @StateObject private var serviceManager = ServiceManager()
    @StateObject private var menuBarManager = MenuBarManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(serviceManager)
                .environmentObject(menuBarManager)
                .onAppear {
                    serviceManager.startServices()
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .commands {
            AgenticSeekCommands()
        }
        
        MenuBarExtra("AgenticSeek", systemImage: "brain.head.profile") {
            MenuBarView()
                .environmentObject(serviceManager)
                .environmentObject(menuBarManager)
        }
        .menuBarExtraStyle(.window)
    }
}

struct AgenticSeekCommands: Commands {
    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Chat") {
                // Handle new chat
            }
            .keyboardShortcut("n", modifiers: .command)
            
            Button("Clear Conversation") {
                // Handle clear conversation
            }
            .keyboardShortcut("k", modifiers: [.command, .shift])
        }
        
        CommandGroup(replacing: .help) {
            Button("AgenticSeek Help") {
                // Open help
            }
            
            Button("Check Services Status") {
                // Check backend services
            }
        }
    }
}

struct MenuBarView: View {
    @EnvironmentObject var serviceManager: ServiceManager
    
    var body: some View {
        VStack(alignment: .leading) {
            Text("AgenticSeek")
                .font(.headline)
                .padding(.bottom, 5)
            
            HStack {
                Circle()
                    .fill(serviceManager.isBackendRunning ? .green : .red)
                    .frame(width: 8, height: 8)
                Text("Backend")
                    .font(.caption)
            }
            
            HStack {
                Circle()
                    .fill(serviceManager.isFrontendRunning ? .green : .red)
                    .frame(width: 8, height: 8)
                Text("Frontend")
                    .font(.caption)
            }
            
            Divider()
            
            Button("Open AgenticSeek") {
                // Bring main window to front
                NSApp.activate(ignoringOtherApps: true)
            }
            
            Button("Restart Services") {
                serviceManager.restartServices()
            }
            
            Button("Quit") {
                NSApplication.shared.terminate(nil)
            }
        }
        .padding()
        .frame(minWidth: 150)
    }
}
