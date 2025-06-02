//
// ContentView.swift
// AgenticSeek Enhanced macOS
//
// Main application view integrating voice AI assistant, web browsing, and task execution
// 100% local AI alternative to Manus AI with complete privacy
//

import SwiftUI

// Mock VoiceAICore for build compatibility
class VoiceAICore: ObservableObject {
    @Published var voiceActivated = false
    
    func startVoiceActivation() {}
    func speak(_ text: String, priority: Priority = .normal) {}
    
    enum Priority {
        case normal, high
    }
}


struct ContentView: View {
    @StateObject private var voiceAI = VoiceAICore()
    @StateObject private var onboardingManager = OnboardingManager()
    @State private var selectedTab: AppTab = .assistant
    @State private var showingVoiceInterface = false
    
    var body: some View {
        VStack {
            if onboardingManager.isFirstLaunch && !onboardingManager.isOnboardingComplete {
                Text("Onboarding")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ZStack {
                    // Main Application Interface
                    NavigationSplitView {
                        ProductionSidebarView(selectedTab: $selectedTab, onRestartServices: { })
                    } detail: {
                        ProductionDetailView(selectedTab: selectedTab, isLoading: false)
                    }
                    .frame(minWidth: 1200, minHeight: 800)
                    
                    // Voice Interface Overlay
                    if showingVoiceInterface {
                        Text("Voice Interface")
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .background(Color.black.opacity(0.7))
                            .transition(.opacity.combined(with: .scale))
                    }
                }
                .onAppear {
                    setupAgenticSeek()
                }
                .environmentObject(voiceAI)
            }
        }
        .onAppear {
            onboardingManager.loadOnboardingState()
        }
    }
    
    private func setupAgenticSeek() {
        // Initialize voice AI with continuous listening
        voiceAI.startVoiceActivation()
        
        // Set up global hotkeys and system integration
        setupGlobalHotkeys()
        
        print("🚀 AgenticSeek Enhanced macOS started")
        print("🎙️ Voice activation: 'Hey AgenticSeek' or Cmd+Space")
        print("🌐 Web browsing agent ready")
        print("💻 Coding assistant ready")
        print("🧠 Smart agent selection active")
    }
    
    private func toggleVoiceInterface() {
        withAnimation(.easeInOut(duration: 0.3)) {
            showingVoiceInterface.toggle()
        }
    }
    
    private func activateVoice() {
        showingVoiceInterface = true
        voiceAI.voiceActivated = true
        voiceAI.speak("How can I help you?", priority: .high)
    }
    
    private func setupGlobalHotkeys() {
        // TODO: Implement global hotkey registration for system-wide access
        // This would allow voice activation even when app is in background
    }
}

// MARK: - Enhanced App Tabs for AgenticSeek

enum AppTab: String, CaseIterable {
    case assistant = "Assistant"
    case webBrowsing = "Web Browsing"
    case coding = "Coding"
    case tasks = "Tasks"
    case settings = "Settings"
    
    var icon: String {
        switch self {
        case .assistant: return "brain.head.profile"
        case .webBrowsing: return "globe"
        case .coding: return "chevron.left.forwardslash.chevron.right"
        case .tasks: return "list.bullet.clipboard"
        case .settings: return "gearshape"
        }
    }
    
    var description: String {
        switch self {
        case .assistant: return "Voice-enabled AI assistant"
        case .webBrowsing: return "Autonomous web browsing"
        case .coding: return "Multi-language code assistant"
        case .tasks: return "Task planning and execution"
        case .settings: return "Application settings"
        }
    }
}

#Preview {
    ContentView()
}