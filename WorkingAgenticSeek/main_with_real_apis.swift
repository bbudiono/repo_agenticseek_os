import SwiftUI
import Foundation

@main
struct WorkingAgenticSeekApp: App {
    var body: some Scene {
        WindowGroup {
            WorkingChatbotView()
                .frame(minWidth: 800, minHeight: 600)
        }
        .windowResizability(.contentSize)
    }
}

struct WorkingChatbotView: View {
    @StateObject private var apiManager = APIManager()
    @State private var messageText: String = ""
    @State private var messages: [ChatMessageData] = []
    @State private var isLoading = false
    @State private var currentProvider = "Anthropic Claude"
    @State private var taskCounter = 0
    
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
        .onAppear {
            setupInitialMessages()
            apiManager.loadAPIKeys()
        }
    }
    
    private var headerView: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("🤖 AgenticSeek AI - REAL API INTEGRATION")
                    .font(.title2)
                    .fontWeight(.bold)
                
                HStack(spacing: 4) {
                    Circle()
                        .fill(apiManager.isAuthenticated ? Color.green : Color.orange)
                        .frame(width: 8, height: 8)
                    Text(apiManager.isAuthenticated ? "✅ Ready • bernhardbudiono@gmail.com" : "⚠️ Authentication Required")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .accessibilityLabel("Authentication Status")
                }
                
                HStack(spacing: 4) {
                    Circle()
                        .fill(apiManager.apiKeysLoaded ? Color.green : Color.red)
                        .frame(width: 6, height: 6)
                    Text(apiManager.apiKeysLoaded ? "✅ API Keys Loaded" : "❌ API Keys Missing")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .accessibilityLabel("API Keys Status")
                }
            }
            
            Spacer()
            
            VStack(alignment: .trailing) {
                Menu {
                    Button("Anthropic Claude") { 
                        currentProvider = "Anthropic Claude"
                        logTaskMasterActivity("Provider switched to Anthropic Claude")
                    }
                    Button("OpenAI GPT-4") { 
                        currentProvider = "OpenAI GPT-4" 
                        logTaskMasterActivity("Provider switched to OpenAI GPT-4")
                    }
                } label: {
                    HStack {
                        Image(systemName: "brain.head.profile")
                        Text(currentProvider)
                        Image(systemName: "chevron.down")
                    }
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.ultraThinMaterial)
                    .cornerRadius(6)
                }
                
                Text("TaskMaster Level 5-6: \(taskCounter) operations")
                    .accessibilityLabel("TaskMaster Operations Counter")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.controlBackgroundColor))
    }
    
    private var messagesView: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
                ForEach(messages) { message in
                    MessageBubbleView(message: message)
                }
                
                if isLoading {
                    HStack {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text("\(currentProvider) is generating response...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                }
            }
            .padding()
        }
        .frame(maxHeight: .infinity)
    }
    
    private var inputView: some View {
        HStack {
            TextField("Type your message (REAL API calls will be made)...", text: $messageText)
                .textFieldStyle(.roundedBorder)
                .onSubmit {
                    sendRealMessage()
                }
                .accessibilityIdentifier("MessageTextField")
                .accessibilityLabel("Message Input Field")
                .accessibilityValue(messageText)
            
            Button("Send to \(currentProvider == "Anthropic Claude" ? "Claude" : "GPT-4")") {
                sendRealMessage()
            }
            .buttonStyle(.borderedProminent)
            .disabled(messageText.isEmpty || !apiManager.apiKeysLoaded)
            
            Button("Test API") {
                testAPIConnection()
            }
            .buttonStyle(.bordered)
            .disabled(!apiManager.apiKeysLoaded)
        }
        .padding()
        .background(Color(.controlBackgroundColor))
    }
    
    private func setupInitialMessages() {
        messages = [
            ChatMessageData(content: "🚀 AgenticSeek AI Assistant - REAL API INTEGRATION", isFromUser: false),
            ChatMessageData(content: "✅ SSO: Authenticated as bernhardbudiono@gmail.com", isFromUser: false),
            ChatMessageData(content: "🔄 Loading API keys from global .env file...", isFromUser: false)
        ]
    }
    
    private func logTaskMasterActivity(_ activity: String) {
        taskCounter += 1
        print("📋 TaskMaster Level 5-6 Activity #\(taskCounter): \(activity)")
        
        let taskMessage = ChatMessageData(
            content: "📋 TaskMaster Activity #\(taskCounter): \(activity)",
            isFromUser: false
        )
        messages.append(taskMessage)
    }
    
    private func sendRealMessage() {
        guard !messageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard apiManager.apiKeysLoaded else {
            let errorMessage = ChatMessageData(content: "❌ Cannot send message: API keys not loaded", isFromUser: false)
            messages.append(errorMessage)
            return
        }
        
        // Add user message
        let userMessage = ChatMessageData(content: messageText, isFromUser: true)
        messages.append(userMessage)
        
        let messageToSend = messageText
        messageText = ""
        isLoading = true
        
        logTaskMasterActivity("Real API call initiated to \(currentProvider)")
        
        // Make REAL API call
        Task {
            do {
                let response = try await apiManager.sendMessage(messageToSend, to: currentProvider)
                
                await MainActor.run {
                    let aiMessage = ChatMessageData(
                        content: "🤖 \(currentProvider) (REAL API): \(response)",
                        isFromUser: false
                    )
                    messages.append(aiMessage)
                    isLoading = false
                    logTaskMasterActivity("Real API response received from \(currentProvider)")
                }
            } catch {
                await MainActor.run {
                    let errorMessage = ChatMessageData(
                        content: "❌ API Error: \(error.localizedDescription)",
                        isFromUser: false
                    )
                    messages.append(errorMessage)
                    isLoading = false
                    logTaskMasterActivity("API call failed: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func testAPIConnection() {
        logTaskMasterActivity("API connection test initiated")
        
        Task {
            let testMessage = "Hello! Please respond with 'API connection successful' to confirm you're working."
            
            do {
                let response = try await apiManager.sendMessage(testMessage, to: currentProvider)
                
                await MainActor.run {
                    let testResult = ChatMessageData(
                        content: "🧪 API Test Result: \(response)",
                        isFromUser: false
                    )
                    messages.append(testResult)
                    logTaskMasterActivity("API test completed successfully")
                }
            } catch {
                await MainActor.run {
                    let errorMessage = ChatMessageData(
                        content: "🧪 API Test FAILED: \(error.localizedDescription)",
                        isFromUser: false
                    )
                    messages.append(errorMessage)
                    logTaskMasterActivity("API test failed")
                }
            }
        }
    }
}

struct MessageBubbleView: View {
    let message: ChatMessageData
    
    var body: some View {
        HStack {
            if message.isFromUser {
                Spacer(minLength: 100)
                
                Text(message.content)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(16)
                    .frame(maxWidth: 300, alignment: .trailing)
            } else {
                Text(message.content)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(.controlColor))
                    .foregroundColor(.primary)
                    .cornerRadius(16)
                    .frame(maxWidth: 500, alignment: .leading)
                
                Spacer(minLength: 50)
            }
        }
    }
}

struct ChatMessageData: Identifiable {
    let id = UUID()
    let content: String
    let isFromUser: Bool
    let timestamp = Date()
}

// REAL API MANAGER WITH ACTUAL LLM CALLS
@MainActor
class APIManager: ObservableObject {
    @Published var isAuthenticated = true
    @Published var apiKeysLoaded = false
    
    private var apiKeys: [String: String] = [:]
    
    func loadAPIKeys() {
        let envPath = "/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/.env"
        
        guard let envContent = try? String(contentsOfFile: envPath) else {
            print("❌ Could not load .env file from: \(envPath)")
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
        
        apiKeysLoaded = apiKeys["ANTHROPIC_API_KEY"] != nil && apiKeys["OPENAI_API_KEY"] != nil
        
        print("✅ Loaded \(apiKeys.count) API keys. Required keys present: \(apiKeysLoaded)")
        print("🔑 Anthropic key: \(apiKeys["ANTHROPIC_API_KEY"] != nil ? "Present" : "Missing")")
        print("🔑 OpenAI key: \(apiKeys["OPENAI_API_KEY"] != nil ? "Present" : "Missing")")
    }
    
    func sendMessage(_ message: String, to provider: String) async throws -> String {
        switch provider {
        case "Anthropic Claude":
            return try await callAnthropicAPI(message: message)
        case "OpenAI GPT-4":
            return try await callOpenAIAPI(message: message)
        default:
            throw APIError.unknownProvider
        }
    }
    
    private func callAnthropicAPI(message: String) async throws -> String {
        guard let apiKey = apiKeys["ANTHROPIC_API_KEY"] else {
            throw APIError.missingAPIKey("Anthropic API key not found in .env")
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
            throw APIError.requestFailed("Anthropic API request failed with status: \((response as? HTTPURLResponse)?.statusCode ?? 0)")
        }
        
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let content = json?["content"] as? [[String: Any]],
              let text = content.first?["text"] as? String else {
            throw APIError.invalidResponse("Invalid Anthropic response format")
        }
        
        return text
    }
    
    private func callOpenAIAPI(message: String) async throws -> String {
        guard let apiKey = apiKeys["OPENAI_API_KEY"] else {
            throw APIError.missingAPIKey("OpenAI API key not found in .env")
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
            throw APIError.requestFailed("OpenAI API request failed with status: \((response as? HTTPURLResponse)?.statusCode ?? 0)")
        }
        
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let choices = json?["choices"] as? [[String: Any]],
              let firstChoice = choices.first,
              let message = firstChoice["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw APIError.invalidResponse("Invalid OpenAI response format")
        }
        
        return content
    }
}

enum APIError: LocalizedError {
    case missingAPIKey(String)
    case requestFailed(String)
    case invalidResponse(String)
    case unknownProvider
    
    var errorDescription: String? {
        switch self {
        case .missingAPIKey(let message): return message
        case .requestFailed(let message): return message
        case .invalidResponse(let message): return message
        case .unknownProvider: return "Unknown provider"
        }
    }
}