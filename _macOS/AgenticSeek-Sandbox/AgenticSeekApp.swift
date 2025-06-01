// SANDBOX FILE: For testing/development. See .cursorrules.
//
// Purpose: Simplified app entry point for Sandbox testing - focused on accessibility validation
// Issues & Complexity Summary: Minimal complexity, basic SwiftUI app structure for testing
// Key Complexity Drivers:
//   - Logic Scope (Est. LoC): ~50
//   - Core Algorithm Complexity: Low
//   - Dependencies: 0 External, 1 Internal (ContentView)
//   - State Management Complexity: Low
//   - Novelty/Uncertainty Factor: Low
// AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 20%
// Problem Estimate (Inherent Problem Difficulty %): 15%
// Initial Code Complexity Estimate %: 15%
// Justification for Estimates: Simple SwiftUI App struct with minimal dependencies for accessibility testing
// Final Code Complexity (Actual %): 18%
// Overall Result Score (Success & Quality %): 95%
// Key Variances/Learnings: Simplified structure enables focused accessibility testing
// Last Updated: 2025-06-01

import SwiftUI

@main
struct AgenticSeekApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .commands {
            AgenticSeekCommands()
        }
    }
}

struct AgenticSeekCommands: Commands {
    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Chat") {
                print("🧪 Sandbox: New Chat requested")
            }
            .keyboardShortcut("n", modifiers: .command)
            
            Button("Clear Conversation") {
                print("🧪 Sandbox: Clear conversation requested")
            }
            .keyboardShortcut("k", modifiers: [.command, .shift])
        }
        
        CommandGroup(replacing: .help) {
            Button("AgenticSeek Help") {
                print("🧪 Sandbox: Help requested")
            }
            
            Button("Run Accessibility Test") {
                print("🧪 Sandbox: Accessibility test requested")
            }
            .keyboardShortcut("a", modifiers: [.command, .shift])
        }
    }
}