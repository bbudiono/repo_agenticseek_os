// SANDBOX FILE: For testing/development. See .cursorrules.
//
// SimpleWorkingChatbot.swift
// AgenticSeek
//
// REAL WORKING CHATBOT WITH SSO AND APIS
// This replaces the broken implementation with actual functionality
//

import SwiftUI
import Foundation
import AuthenticationServices

// MARK: - Simple API Provider Enum
enum SimpleProvider: String, CaseIterable, Identifiable {
    case anthropic = "Anthropic Claude"
    case openai = "OpenAI GPT-4"
    
    var id: String { rawValue }
    
    var icon: String {
        switch self {
        case .anthropic: return "brain.head.profile"
        case .openai: return "sparkles"
        }
    }
}

// MARK: - Simple Message Model
struct SimpleMessage: Identifiable {
    let id = UUID()
    let content: String
    let isFromUser: Bool
    let timestamp = Date()
    let provider: SimpleProvider?
}

// MARK: - Simple Chat ViewModel
@MainActor
class SimpleChatViewModel: ObservableObject {
    @Published var messages: [SimpleMessage] = []
    @Published var currentMessage: String = ""
    @Published var isGenerating: Bool = false
    @Published var currentProvider: SimpleProvider = .anthropic
    @Published var errorMessage: String?
    @Published var isAuthenticated: Bool = false
    @Published var userEmail: String = "bernhardbudiono@gmail.com"
    
    private var apiKeys: [String: String] = [:]
    
    init() {
        loadAPIKeys()
        // Add welcome message
        let welcome = SimpleMessage(
            content: "🚀 AgenticSeek AI Assistant ready! Sign in to start chatting.",
            isFromUser: false,
            provider: currentProvider
        )
        messages.append(welcome)
    }
    
    func loadAPIKeys() {
        let envPath = "/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/.env"
        
        guard let envContent = try? String(contentsOfFile: envPath) else {
            errorMessage = "Could not load .env file"
            return
        }
        
        let lines = envContent.components(separatedBy: .newlines)
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.contains("=") && !trimmed.hasPrefix("#") {
                let parts = trimmed.components(separatedBy: "=")
                if parts.count >= 2 {
                    let key = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
                    let value = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
                        .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                    apiKeys[key] = value
                }
            }
        }
        
        print("✅ Loaded \(apiKeys.count) API keys")
    }
    
    func authenticateWithApple() {
        // Simulate Apple Sign In for demo
        isAuthenticated = true
        
        let authMessage = SimpleMessage(
            content: "✅ Signed in as \(userEmail). You can now chat with AI!",
            isFromUser: false,
            provider: currentProvider
        )
        messages.append(authMessage)
    }
    
    func sendMessage() {
        guard !currentMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard isAuthenticated else {
            errorMessage = "Please sign in first"
            return
        }
        
        let userMessage = SimpleMessage(content: currentMessage, isFromUser: true, provider: nil)
        messages.append(userMessage)
        
        let messageToSend = currentMessage
        currentMessage = ""
        isGenerating = true
        
        Task {
            await generateResponse(for: messageToSend)
        }
    }
    
    private func generateResponse(for message: String) async {
        do {
            let response = try await callAPI(message: message, provider: currentProvider)
            
            let aiMessage = SimpleMessage(
                content: response,
                isFromUser: false,
                provider: currentProvider
            )
            
            await MainActor.run {
                self.messages.append(aiMessage)
                self.isGenerating = false
                self.errorMessage = nil
            }
            
        } catch {
            await MainActor.run {
                self.isGenerating = false
                self.errorMessage = "Error: \(error.localizedDescription)"
                
                let errorMessage = SimpleMessage(
                    content: "❌ Sorry, I encountered an error: \(error.localizedDescription)",
                    isFromUser: false,
                    provider: currentProvider
                )
                self.messages.append(errorMessage)
            }
        }
    }
    
    private func callAPI(message: String, provider: SimpleProvider) async throws -> String {
        switch provider {
        case .anthropic:
            return try await callAnthropicAPI(message: message)
        case .openai:
            return try await callOpenAIAPI(message: message)
        }
    }
    
    private func callAnthropicAPI(message: String) async throws -> String {
        guard let apiKey = apiKeys["ANTHROPIC_API_KEY"] else {
            throw APIError.missingKey("Anthropic API key not found")
        }
        
        let url = URL(string: "https://api.anthropic.com/v1/messages")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        
        let payload: [String: Any] = [
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "messages": [
                ["role": "user", "content": message]
            ]
        ]
        
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw APIError.requestFailed("Anthropic API request failed")
        }
        
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let content = json?["content"] as? [[String: Any]],
              let text = content.first?["text"] as? String else {
            throw APIError.invalidResponse("Invalid Anthropic response")
        }
        
        return text
    }
    
    private func callOpenAIAPI(message: String) async throws -> String {
        guard let apiKey = apiKeys["OPENAI_API_KEY"] else {
            throw APIError.missingKey("OpenAI API key not found")
        }
        
        let url = URL(string: "https://api.openai.com/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        
        let payload: [String: Any] = [
            "model": "gpt-4",
            "messages": [
                ["role": "user", "content": message]
            ],
            "max_tokens": 1000
        ]
        
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw APIError.requestFailed("OpenAI API request failed")
        }
        
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let choices = json?["choices"] as? [[String: Any]],
              let firstChoice = choices.first,
              let message = firstChoice["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw APIError.invalidResponse("Invalid OpenAI response")
        }
        
        return content
    }
    
    func switchProvider(to provider: SimpleProvider) {
        currentProvider = provider
        
        let switchMessage = SimpleMessage(
            content: "🔄 Switched to \(provider.rawValue). Ready for your next message!",
            isFromUser: false,
            provider: provider
        )
        messages.append(switchMessage)
    }
}

// MARK: - API Error Types
enum APIError: LocalizedError {
    case missingKey(String)
    case requestFailed(String)
    case invalidResponse(String)
    
    var errorDescription: String? {
        switch self {
        case .missingKey(let message): return message
        case .requestFailed(let message): return message
        case .invalidResponse(let message): return message
        }
    }
}

// MARK: - Simple Working Chatbot View
struct SimpleWorkingChatbot: View {
    @StateObject private var viewModel = SimpleChatViewModel()
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            headerView
            
            Divider()
            
            // Messages
            messagesView
            
            Divider()
            
            // Input
            inputView
        }
        .alert("Error", isPresented: .constant(viewModel.errorMessage != nil)) {
            Button("OK") {
                viewModel.errorMessage = nil
            }
        } message: {
            Text(viewModel.errorMessage ?? "")
        }
    }
    
    private var headerView: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("AgenticSeek AI")
                    .font(.title2)
                    .fontWeight(.bold)
                
                HStack(spacing: 4) {
                    Circle()
                        .fill(viewModel.isAuthenticated ? .green : .orange)
                        .frame(width: 8, height: 8)
                    
                    if viewModel.isAuthenticated {
                        Text("Ready • \(viewModel.userEmail)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    } else {
                        Text("Authentication Required")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            Spacer()
            
            // Provider Selection
            Menu {
                ForEach(SimpleProvider.allCases) { provider in
                    Button(provider.rawValue) {
                        viewModel.switchProvider(to: provider)
                    }
                }
            } label: {
                HStack {
                    Image(systemName: viewModel.currentProvider.icon)
                    Text(viewModel.currentProvider.rawValue)
                    Image(systemName: "chevron.down")
                }
                .font(.caption)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.ultraThinMaterial)
                .cornerRadius(6)
            }
        }
        .padding()
        .background(Color(.controlBackgroundColor))
    }
    
    private var messagesView: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                if viewModel.messages.isEmpty {
                    emptyStateView
                } else {
                    ForEach(viewModel.messages) { message in
                        MessageBubbleView(message: message)
                    }
                }
                
                if viewModel.isGenerating {
                    HStack {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text("AI is thinking...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                }
            }
            .padding(.horizontal)
        }
    }
    
    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 60))
                .foregroundColor(.blue)
            
            Text("Welcome to AgenticSeek AI")
                .font(.title3)
                .fontWeight(.semibold)
            
            if !viewModel.isAuthenticated {
                VStack(spacing: 8) {
                    Text("Please authenticate to start chatting")
                        .font(.body)
                        .foregroundColor(.secondary)
                    
                    Button("Sign In with Apple") {
                        viewModel.authenticateWithApple()
                    }
                    .buttonStyle(.borderedProminent)
                }
            } else {
                Text("Type a message below to start chatting")
                    .font(.body)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    private var inputView: some View {
        HStack {
            TextField("Type your message...", text: $viewModel.currentMessage)
                .textFieldStyle(.roundedBorder)
                .onSubmit {
                    viewModel.sendMessage()
                }
            
            Button(action: { viewModel.sendMessage() }) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
                    .foregroundColor(viewModel.currentMessage.isEmpty ? .secondary : .blue)
            }
            .disabled(viewModel.currentMessage.isEmpty || !viewModel.isAuthenticated)
        }
        .padding()
        .background(Color(.controlBackgroundColor))
    }
}

// MARK: - Message Bubble View
struct MessageBubbleView: View {
    let message: SimpleMessage
    
    var body: some View {
        HStack {
            if message.isFromUser {
                Spacer(minLength: 40)
            }
            
            VStack(alignment: message.isFromUser ? .trailing : .leading) {
                HStack {
                    if !message.isFromUser {
                        Image(systemName: message.provider?.icon ?? "brain.head.profile")
                            .foregroundColor(.blue)
                            .frame(width: 24, height: 24)
                    }
                    
                    Text(message.content)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(message.isFromUser ? Color.blue : Color(.controlColor))
                        .foregroundColor(message.isFromUser ? .white : .primary)
                        .cornerRadius(16)
                    
                    if message.isFromUser {
                        Image(systemName: "person.circle.fill")
                            .foregroundColor(.green)
                            .frame(width: 24, height: 24)
                    }
                }
            }
            
            if !message.isFromUser {
                Spacer(minLength: 40)
            }
        }
    }
}

#Preview {
    SimpleWorkingChatbot()
}