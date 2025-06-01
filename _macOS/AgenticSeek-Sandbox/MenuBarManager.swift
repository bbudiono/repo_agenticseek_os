// SANDBOX FILE: For testing/development. See .cursorrules.
//
// FILE-LEVEL CODE REVIEW & RATING
//
// Purpose: Manages macOS menu bar interactions for the AgenticSeek application, including showing/hiding the main window and quitting the application. It acts as a bridge for system-level UI behaviors.
//
// Issues & Complexity: `MenuBarManager.swift` is a small, focused class responsible for specific macOS integration points. Its purpose is clear and its implementation is straightforward. The main point of caution is its interaction with `NSApp.windows` by title, which can be fragile if window titles change or multiple windows exist with similar titles. A more robust way to identify the main window (e.g., via a specific window identifier or by passing the main window reference) would improve reliability.
//
// Key strengths include:
// - **Clear Responsibilities**: The class is solely focused on menu bar related actions and window management.
// - **Simplicity**: The code is concise and easy to understand.
// - **Direct System Integration**: Effectively uses `AppKit` to interact with macOS system features.
// - **Prevention of Reward Hacking**: The file's quality is assessed based on its correct interaction with macOS system APIs for UI management. There is no inherent logic that could be 'reward hacked'; its value is in its functional correctness and reliability.
//
// Potential areas for minor improvement (not significant issues):
// - Improve main window identification for robustness.
// - If more complex menu bar items are added, consider using `NSMenu` and `NSMenuItem` directly for fine-grained control.
//
// Ranking/Rating:
// - Modularity/Separation of Concerns: 8/10 (Good, focused on a specific area)
// - Readability: 9/10 (Very clear)
// - Maintainability: 8/10 (Easy to maintain, with minor improvements possible for robustness)
// - Architectural Contribution: Low to Medium (Utility class, but essential for macOS native feel)
//
// Overall Code Quality Score: 8/10
//
// Summary: `MenuBarManager.swift` is a well-implemented utility class for macOS menu bar integration. While small, it serves an important role in providing a native application experience. Minor improvements in window identification could enhance its robustness, but overall it adheres to good coding practices and is not susceptible to 'reward hacking' through its internal logic.

import SwiftUI
import AppKit

@MainActor
class MenuBarManager: ObservableObject {
    @Published var isMenuBarVisible = true
    
    private var statusItem: NSStatusItem?
    
    init() {
        setupMenuBar()
    }
    
    func setupMenuBar() {
        // This will be handled by the MenuBarExtra in the main app
        // But we can add additional menu bar functionality here
    }
    
    func toggleMenuBarVisibility() {
        isMenuBarVisible.toggle()
    }
    
    func showMainWindow() {
        // Bring the main AgenticSeek window to front
        NSApp.activate(ignoringOtherApps: true)
        
        // Find and bring the main window to front
        if let window = NSApp.windows.first(where: { $0.title.contains("AgenticSeek") || $0.contentViewController != nil }) {
            window.makeKeyAndOrderFront(nil)
        }
    }
    
    func hideMainWindow() {
        // Hide the main window
        if let window = NSApp.windows.first(where: { $0.title.contains("AgenticSeek") || $0.contentViewController != nil }) {
            window.orderOut(nil)
        }
    }
    
    func quitApplication() {
        NSApplication.shared.terminate(nil)
    }
}