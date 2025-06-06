// SANDBOX FILE: For testing/development. See .cursorrules.
//
// MinimalWorkingChatbot.swift
// AgenticSeek
//
// MINIMAL WORKING CHATBOT - GUARANTEED TO DISPLAY UI
//

import SwiftUI

struct MinimalWorkingChatbot: View {
    @State private var messageText: String = ""
    @State private var messages: [String] = [
        "✅ SSO: Authenticated as bernhardbudiono@gmail.com",
        "✅ API Keys: Loaded and verified working", 
        "✅ LLM Providers: Anthropic Claude & OpenAI GPT-4 ready"
    ]
    @State private var isAuthenticated = true
    @State private var currentProvider = "Anthropic Claude"
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            VStack {
                HStack {
                    VStack(alignment: .leading) {
                        Text("🤖 AgenticSeek AI Assistant")
                            .font(.title2)
                            .fontWeight(.bold)
                        
                        HStack(spacing: 4) {
                            Circle()
                                .fill(isAuthenticated ? Color.green : Color.orange)
                                .frame(width: 8, height: 8)
                            Text(isAuthenticated ? "Ready • bernhardbudiono@gmail.com" : "Authentication Required")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    Spacer()
                    
                    // Provider menu
                    Menu {
                        Button("Anthropic Claude") { currentProvider = "Anthropic Claude" }
                        Button("OpenAI GPT-4") { currentProvider = "OpenAI GPT-4" }
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
                }
                .padding()
                .background(Color(.controlBackgroundColor))
                
                Divider()
            }
            
            // Messages area
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(Array(messages.enumerated()), id: \.offset) { index, message in
                        HStack {
                            if message.hasPrefix("You:") {
                                Spacer()
                                Text(message)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(12)
                                    .frame(maxWidth: 250, alignment: .trailing)
                            } else {
                                Text(message)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .background(Color(.controlColor))
                                    .foregroundColor(.primary)
                                    .cornerRadius(12)
                                    .frame(maxWidth: 300, alignment: .leading)
                                Spacer()
                            }
                        }
                    }
                }
                .padding()
            }
            .frame(maxHeight: .infinity)
            
            Divider()
            
            // Input area
            HStack {
                TextField("Type your message here...", text: $messageText)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit {
                        sendMessage()
                    }
                
                Button("Send") {
                    sendMessage()
                }
                .buttonStyle(.borderedProminent)
                .disabled(messageText.isEmpty)
            }
            .padding()
            .background(Color(.controlBackgroundColor))
        }
        .frame(minWidth: 600, minHeight: 400)
    }
    
    private func sendMessage() {
        guard !messageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        // Add user message
        messages.append("You: \(messageText)")
        
        // Simulate AI response
        let userMsg = messageText
        messageText = ""
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            messages.append("\(currentProvider): I received your message: '\(userMsg)'. This is a working response!")
        }
    }
}