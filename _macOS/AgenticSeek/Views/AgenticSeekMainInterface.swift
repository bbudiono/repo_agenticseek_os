//
// AgenticSeekMainInterface.swift
// AgenticSeek Enhanced macOS
//
// Main interface components for the voice-enabled AI assistant
// Sidebar, main view, and voice overlay implementation
//

import SwiftUI

// MARK: - AgenticSeek Sidebar

struct AgenticSeekSidebar: View {
    @Binding var selectedTab: AppTab
    @ObservedObject var voiceAI: VoiceAICore
    
    var body: some View {
        VStack(spacing: 0) {
            // Header with voice status
            VStack(spacing: 12) {
                HStack {
                    Image(systemName: "brain.head.profile")
                        .font(.title2)
                        .foregroundStyle(.accent)
                    
                    Text("AgenticSeek")
                        .font(.title2)
                        .fontWeight(.semibold)
                    
                    Spacer()
                }
                
                VoiceStatusIndicator(voiceAI: voiceAI)
            }
            .padding(.horizontal, 16)
            .padding(.top, 20)
            .padding(.bottom, 16)
            
            Divider()
            
            // Navigation Tabs
            VStack(spacing: 4) {
                ForEach(AppTab.allCases, id: \.self) { tab in
                    SidebarTabButton(
                        tab: tab,
                        isSelected: selectedTab == tab,
                        action: { selectedTab = tab }
                    )
                }
            }
            .padding(.horizontal, 12)
            .padding(.top, 16)
            
            Spacer()
            
            // Quick Actions
            VStack(spacing: 8) {
                QuickActionButton(
                    icon: "mic.fill",
                    title: "Voice Command",
                    subtitle: "Cmd+Space",
                    isActive: voiceAI.voiceActivated,
                    action: { voiceAI.voiceActivated.toggle() }
                )
                
                QuickActionButton(
                    icon: "globe",
                    title: "Browse Web",
                    subtitle: "Autonomous",
                    action: { /* TODO: Quick web browse */ }
                )
                
                QuickActionButton(
                    icon: "terminal",
                    title: "Code Assistant",
                    subtitle: "Multi-language",
                    action: { /* TODO: Quick code assist */ }
                )
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 20)
        }
        .frame(width: 260)
        .background(.regularMaterial, in: Rectangle())
    }
}

// MARK: - Voice Status Indicator

struct VoiceStatusIndicator: View {
    @ObservedObject var voiceAI: VoiceAICore
    
    var body: some View {
        HStack(spacing: 8) {
            // Status Icon
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
                .scaleEffect(voiceAI.isListening ? 1.5 : 1.0)
                .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true), value: voiceAI.isListening)
            
            Text(voiceAI.agentStatus.displayText)
                .font(.caption)
                .foregroundStyle(.secondary)
            
            Spacer()
            
            if voiceAI.isListening {
                Image(systemName: "waveform")
                    .font(.caption)
                    .foregroundStyle(.accent)
                    .symbolEffect(.pulse)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
    }
    
    private var statusColor: Color {
        switch voiceAI.agentStatus {
        case .idle:
            return .secondary
        case .listening:
            return .blue
        case .analyzing:
            return .orange
        case .executing:
            return .green
        case .completed:
            return .green
        case .error:
            return .red
        }
    }
}

// MARK: - Sidebar Tab Button

struct SidebarTabButton: View {
    let tab: AppTab
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: tab.icon)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(isSelected ? .accent : .secondary)
                    .frame(width: 20)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(tab.rawValue)
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(isSelected ? .primary : .secondary)
                    
                    Text(tab.description)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(isSelected ? .accent.opacity(0.1) : .clear)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Quick Action Button

struct QuickActionButton: View {
    let icon: String
    let title: String
    let subtitle: String
    var isActive: Bool = false
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(isActive ? .accent : .secondary)
                    .frame(width: 18)
                
                VStack(alignment: .leading, spacing: 1) {
                    Text(title)
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundStyle(isActive ? .accent : .primary)
                    
                    Text(subtitle)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                
                Spacer()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isActive ? .accent.opacity(0.1) : .quaternary.opacity(0.3))
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - AgenticSeek Main View

struct AgenticSeekMainView: View {
    let selectedTab: AppTab
    @ObservedObject var voiceAI: VoiceAICore
    
    var body: some View {
        ZStack {
            switch selectedTab {
            case .assistant:
                VoiceAssistantView(voiceAI: voiceAI)
            case .webBrowsing:
                WebBrowsingView()
            case .coding:
                CodingAssistantView()
            case .tasks:
                TaskPlanningView()
            case .settings:
                SettingsView()
            }
        }
        .navigationTitle(selectedTab.rawValue)
        .navigationBarTitleDisplayMode(.large)
    }
}

// MARK: - Voice Interface Overlay

struct VoiceInterfaceOverlay: View {
    @ObservedObject var voiceAI: VoiceAICore
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        ZStack {
            // Background blur
            Rectangle()
                .fill(.ultraThinMaterial)
                .ignoresSafeArea()
                .onTapGesture {
                    dismiss()
                }
            
            // Voice interface card
            VStack(spacing: 24) {
                // Voice visualization
                VoiceVisualizationView(isListening: voiceAI.isListening, isProcessing: voiceAI.isProcessing)
                
                // Status and transcription
                VStack(spacing: 12) {
                    Text(voiceAI.agentStatus.displayText)
                        .font(.title2)
                        .fontWeight(.semibold)
                    
                    if !voiceAI.currentTranscription.isEmpty {
                        Text(voiceAI.currentTranscription)
                            .font(.body)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    
                    if !voiceAI.currentTask.isEmpty {
                        Text(voiceAI.currentTask)
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                }
                
                // Control buttons
                HStack(spacing: 16) {
                    Button("Cancel") {
                        voiceAI.stopVoiceActivation()
                        dismiss()
                    }
                    .keyboardShortcut(.escape)
                    
                    Button(voiceAI.isListening ? "Stop Listening" : "Start Listening") {
                        if voiceAI.isListening {
                            voiceAI.stopVoiceActivation()
                        } else {
                            voiceAI.startVoiceActivation()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding(32)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
            .shadow(radius: 20)
        }
    }
}

// MARK: - Voice Visualization

struct VoiceVisualizationView: View {
    let isListening: Bool
    let isProcessing: Bool
    
    var body: some View {
        ZStack {
            // Outer ring
            Circle()
                .stroke(
                    .accent.opacity(0.3),
                    style: StrokeStyle(lineWidth: 2, dash: [10, 5])
                )
                .frame(width: 120, height: 120)
                .rotationEffect(.degrees(isListening ? 360 : 0))
                .animation(.linear(duration: 8).repeatForever(autoreverses: false), value: isListening)
            
            // Inner circle
            Circle()
                .fill(.accent)
                .frame(width: 80, height: 80)
                .scaleEffect(isListening ? 1.2 : (isProcessing ? 0.9 : 1.0))
                .opacity(isListening ? 0.8 : (isProcessing ? 0.6 : 1.0))
                .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isListening)
            
            // Center icon
            Image(systemName: isListening ? "mic.fill" : (isProcessing ? "brain.head.profile" : "mic"))
                .font(.title)
                .foregroundStyle(.white)
        }
    }
}

// MARK: - Tab Content Views (Placeholders)

struct VoiceAssistantView: View {
    @ObservedObject var voiceAI: VoiceAICore
    
    var body: some View {
        VStack {
            Text("Voice-Enabled AI Assistant")
                .font(.title)
            Text("Say 'Hey AgenticSeek' or press Cmd+Space to activate")
                .foregroundStyle(.secondary)
            
            if !voiceAI.lastResponse.isEmpty {
                VStack(alignment: .leading) {
                    Text("Last Response:")
                        .font(.headline)
                    ScrollView {
                        Text(voiceAI.lastResponse)
                            .padding()
                            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                    }
                }
                .padding()
            }
        }
        .padding()
    }
}

struct WebBrowsingView: View {
    var body: some View {
        VStack {
            Text("Autonomous Web Browsing")
                .font(.title)
            Text("AI-powered web navigation and data extraction")
                .foregroundStyle(.secondary)
            
            // TODO: Implement web browsing interface
        }
        .padding()
    }
}

struct CodingAssistantView: View {
    var body: some View {
        VStack {
            Text("Multi-Language Coding Assistant")
                .font(.title)
            Text("Code generation, debugging, and execution")
                .foregroundStyle(.secondary)
            
            // TODO: Implement coding interface
        }
        .padding()
    }
}

struct TaskPlanningView: View {
    var body: some View {
        VStack {
            Text("Task Planning & Execution")
                .font(.title)
            Text("Complex task breakdown and automated execution")
                .foregroundStyle(.secondary)
            
            // TODO: Implement task planning interface
        }
        .padding()
    }
}

struct SettingsView: View {
    var body: some View {
        VStack {
            Text("AgenticSeek Settings")
                .font(.title)
            Text("Configure your local AI assistant")
                .foregroundStyle(.secondary)
            
            // TODO: Implement settings interface
        }
        .padding()
    }
}
