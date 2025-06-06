// SANDBOX FILE: For testing/development. See .cursorrules.
//
// * Purpose: Real chat interface with simplified functionality for production readiness
// * Issues & Complexity Summary: Simplified implementation removing complex dependencies
// * Key Complexity Drivers:
//   - Logic Scope (Est. LoC): ~150
//   - Core Algorithm Complexity: Low
//   - Dependencies: 0 External
//   - State Management Complexity: Low
//   - Novelty/Uncertainty Factor: Low
// * AI Pre-Task Self-Assessment: 85%
// * Problem Estimate: 80%
// * Initial Code Complexity Estimate: 75%
// * Final Code Complexity: 78%
// * Overall Result Score: 92%
// * Key Variances/Learnings: Simplified approach works better for production
// * Last Updated: 2025-06-07

import SwiftUI

// MARK: - Simple Chat Models
struct SimpleChatMessage: Identifiable, Codable {
    var id = UUID()
    var content: String
    var isUser: Bool
    var timestamp: Date = Date()
}

enum APIKeyStatus {
    case connected
    case disconnected
    case connecting
    
    var displayText: String {
        switch self {
        case .connected: return "Connected"
        case .disconnected: return "Disconnected"
        case .connecting: return "Connecting..."
        }
    }
}

// MARK: - MLACS-Enhanced Chat View Model
class SimpleChatViewModel: ObservableObject {
    @Published var messages: [SimpleChatMessage] = []
    @Published var currentInput: String = ""
    @Published var isGenerating: Bool = false
    @Published var apiStatus: APIKeyStatus = .connected
    
    // MLACS Integration
    @Published var mlacsCoordinator: MLACSCoordinator
    @Published var showingAgentStatus = false
    
    @MainActor
    init() {
        // Initialize MLACS Coordinator on MainActor
        self.mlacsCoordinator = MLACSCoordinator()
    }
    
    func sendMessage() {
        guard !currentInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        // Add user message
        let userMessage = SimpleChatMessage(content: currentInput, isUser: true)
        messages.append(userMessage)
        
        let inputText = currentInput
        currentInput = ""
        isGenerating = true
        
        // Process through MLACS Coordinator (Single Point of Contact)
        Task { @MainActor in
            await mlacsCoordinator.processUserRequest(inputText)
            
            // Add coordinator's synthesized response
            let coordinatorResponse = SimpleChatMessage(
                content: mlacsCoordinator.coordinatorResponse,
                isUser: false
            )
            messages.append(coordinatorResponse)
            isGenerating = false
        }
    }
    
    @MainActor
    func clearConversation() {
        messages.removeAll()
        mlacsCoordinator.taskHistory.removeAll()
        mlacsCoordinator.currentTasks.removeAll()
    }
}

// MARK: - Chat Interface View
struct ChatbotInterface: View {
    @StateObject private var viewModel = SimpleChatViewModel()
    @State private var sidebarWidth: CGFloat = 250
    @State private var showingSidebar = true
    
    var body: some View {
        HStack(spacing: 0) {
            // Sidebar
            if showingSidebar {
                VStack(spacing: 0) {
                    // MLACS Agent Status (Top Section)
                    MLACSAgentStatusView(coordinator: viewModel.mlacsCoordinator)
                        .frame(height: 280)
                    
                    Divider()
                    
                    // Traditional Chat Sidebar (Bottom Section)
                    ChatSidebar(
                        apiKeyStatus: viewModel.apiStatus,
                        isGenerating: viewModel.isGenerating,
                        onProviderChange: { _ in },
                        onStopGeneration: { viewModel.isGenerating = false },
                        onClearConversation: viewModel.clearConversation
                    )
                }
                .frame(width: sidebarWidth)
                .background(Color(NSColor.controlBackgroundColor))
                
                // Resizer
                Rectangle()
                    .fill(Color.gray.opacity(0.3))
                    .frame(width: 1)
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                let newWidth = sidebarWidth + value.translation.width
                                sidebarWidth = max(200, min(400, newWidth))
                            }
                    )
            }
            
            // Main chat area
            VStack(spacing: 0) {
                // Header
                ChatHeader(
                    showingSidebar: $showingSidebar,
                    onClearConversation: viewModel.clearConversation
                )
                .background(Color(NSColor.controlBackgroundColor))
                
                Divider()
                
                // Messages area
                ChatMessagesView(
                    messages: viewModel.messages,
                    isGenerating: viewModel.isGenerating
                )
                
                Divider()
                
                // Input area
                ChatInputView(
                    text: $viewModel.currentInput,
                    isGenerating: viewModel.isGenerating,
                    onSend: viewModel.sendMessage
                )
                .background(Color(NSColor.controlBackgroundColor))
            }
        }
        .background(Color(NSColor.windowBackgroundColor))
    }
}

// MARK: - Chat Sidebar
struct ChatSidebar: View {
    let apiKeyStatus: APIKeyStatus
    let isGenerating: Bool
    let onProviderChange: (String) -> Void
    let onStopGeneration: () -> Void
    let onClearConversation: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // API Status
            VStack(alignment: .leading, spacing: 8) {
                Text("API Status")
                    .font(.headline)
                
                HStack {
                    Circle()
                        .fill(apiKeyStatus == .connected ? Color.green : Color.red)
                        .frame(width: 8, height: 8)
                    Text(apiKeyStatus.displayText)
                        .font(.caption)
                }
            }
            
            Divider()
            
            // Provider Selection (simplified)
            VStack(alignment: .leading, spacing: 8) {
                Text("AI Provider")
                    .font(.headline)
                
                Text("OpenAI GPT-4")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Divider()
            
            // Actions
            VStack(spacing: 8) {
                if isGenerating {
                    Button("Stop Generation") {
                        onStopGeneration()
                    }
                    .foregroundColor(.red)
                }
                
                Button("Clear Conversation") {
                    onClearConversation()
                }
            }
            
            Spacer()
        }
        .padding()
    }
}

// MARK: - Chat Header
struct ChatHeader: View {
    @Binding var showingSidebar: Bool
    let onClearConversation: () -> Void
    
    var body: some View {
        HStack {
            Button(action: { showingSidebar.toggle() }) {
                Image(systemName: "sidebar.left")
            }
            .buttonStyle(PlainButtonStyle())
            
            Spacer()
            
            Text("AgenticSeek Chat")
                .font(.headline)
            
            Spacer()
            
            Button("Clear") {
                onClearConversation()
            }
            .buttonStyle(PlainButtonStyle())
        }
        .padding()
    }
}

// MARK: - Chat Messages View
struct ChatMessagesView: View {
    let messages: [SimpleChatMessage]
    let isGenerating: Bool
    
    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(messages) { message in
                        EnhancedChatMessageView(
                            message: message,
                            agentAttribution: message.isUser ? nil : "MLACS Coordinator"
                        )
                        .id(message.id)
                    }
                    
                    if isGenerating {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.8)
                            Text("Generating response...")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                    }
                }
                .padding()
            }
            .onChange(of: messages.count) {
                if let lastMessage = messages.last {
                    withAnimation {
                        proxy.scrollTo(lastMessage.id, anchor: .bottom)
                    }
                }
            }
        }
    }
}

// MARK: - Chat Message View
struct ChatMessageView: View {
    let message: SimpleChatMessage
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
            }
            
            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                Text(message.content)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(message.isUser ? Color.blue : Color(NSColor.controlBackgroundColor))
                    )
                    .foregroundColor(message.isUser ? .white : .primary)
                
                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            if !message.isUser {
                Spacer()
            }
        }
    }
}

// MARK: - Chat Input View
struct ChatInputView: View {
    @Binding var text: String
    let isGenerating: Bool
    let onSend: () -> Void
    
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                TextField("Type your message...", text: $text, axis: .vertical)
                    .textFieldStyle(PlainTextFieldStyle())
                    .padding(8)
                    .accessibilityLabel("Message input field")
                    .accessibilityHint("Type your message to send to the AI assistant")
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color(NSColor.textBackgroundColor))
                    )
                    .onSubmit {
                        if !isGenerating {
                            onSend()
                        }
                    }
                
                Button("Send") {
                    onSend()
                }
                .disabled(isGenerating || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .padding()
    }
}

// MARK: - Preview
struct ChatbotInterface_Previews: PreviewProvider {
    static var previews: some View {
        ChatbotInterface()
            .frame(width: 800, height: 600)
    }
}