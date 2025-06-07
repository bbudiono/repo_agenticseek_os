//
// FILE-LEVEL CODE REVIEW & RATING
//
// Purpose: Provides a user interface for configuring AI model providers, managing API keys, and selecting default models for various roles within the AgenticSeek application. It allows users to enable/disable providers, set API keys, and view their status.
//
// Issues & Complexity: `ConfigurationView.swift` (and its `ConfigurationViewModel`) is a crucial part of the application, enabling users to customize their AI backend experience. The view is well-organized with sections for providers and API keys, and the view model effectively handles data fetching and state management. However, there are some areas for improvement:
// - **Backend URL Hardcoding**: The `baseURL` is hardcoded to `http://localhost:8001`. Similar to `ModelManagementView.swift`, this should be a configurable value to improve flexibility across environments.
// - **Error Handling Presentation**: Error messages are currently simple strings. A more robust and user-friendly error display (e.g., dedicated error alerts or inline messages) would enhance the user experience.
// - **API Key Handling Modality**: While the use of a sheet for API key input is good, ensuring secure handling of the `apiKeyInput` state and its persistence is critical. The current approach assumes backend handles persistence, but the UI should also be robust.
// - **Provider Model Loading Logic**: The `loadAllModels` iterates through a hardcoded list of provider names. This list should ideally be dynamic, perhaps obtained from a `/config/providers` endpoint or a shared source of truth, to ensure consistency and prevent manual updates.
// - **Prevention of Reward Hacking**: The file's quality is inherently tied to its ability to accurately reflect and update backend configurations. There's no clear pathway for 'reward hacking' within the UI logic itself, as its primary function is to serve as a conduit for user input to the backend. The integrity of the configuration ultimately depends on the backend's implementation and validation.
//
// Key strengths include:
// - **Clear UI Structure**: The tabbed interface and clear sections make the configuration options easy to navigate and understand.
// - **MVVM Pattern**: Effective use of `ObservableObject` and `@Published` in `ConfigurationViewModel` demonstrates good separation of concerns.
// - **Asynchronous Operations**: Proper use of `Task` for network requests.
// - **API Integration**: Successfully communicates with various backend configuration endpoints for loading and updating settings.
//
// Ranking/Rating:
// - Modularity/Separation of Concerns: 7/10 (Good separation, but hardcoded elements reduce flexibility)
// - Readability: 7/10 (Clear and well-structured)
// - Maintainability: 6.5/10 (Could be improved by externalizing configurations and enhancing error handling)
// - Architectural Contribution: Medium (Essential for core functionality, but has room for increased robustness and dynamic behavior)
//
// Overall Code Quality Score: 6.5/10
//
// Summary: `ConfigurationView.swift` is a functional and generally well-implemented component that provides essential configuration capabilities for AgenticSeek. Addressing the hardcoded URLs and provider lists, and enhancing error feedback, would significantly improve its robustness, maintainability, and alignment with dynamic backend capabilities. Its value lies in enabling user control over AI settings, making 'reward hacking' a concern more for the backend logic it interacts with than for its own UI implementation.

//
//  ConfigurationView.swift
//  AgenticSeek
//
//  Simple, functional configuration interface
//

import SwiftUI
import Foundation

// MARK: - Data Models
struct ProviderConfig: Codable, Identifiable, Equatable {
    let id = UUID()
    let name: String
    let display_name: String
    let model: String
    let server_address: String
    let is_local: Bool
    let is_enabled: Bool
    let api_key_required: Bool
    let api_key_set: Bool
    let status: String
    
    enum CodingKeys: String, CodingKey {
        case name, display_name, model, server_address, is_local, is_enabled, api_key_required, api_key_set, status
    }
    
    static func == (lhs: ProviderConfig, rhs: ProviderConfig) -> Bool {
        return lhs.name == rhs.name && lhs.model == rhs.model && lhs.api_key_set == rhs.api_key_set
    }
}

struct APIKeyInfo: Codable, Identifiable {
    let id = UUID()
    let provider: String
    let display_name: String
    let is_set: Bool
    let last_updated: String?
    let is_valid: Bool?
    
    enum CodingKeys: String, CodingKey {
        case provider, display_name, is_set, last_updated, is_valid
    }
}

// MARK: - View Model
class ConfigurationViewModel: ObservableObject {
    @Published var providers: [ProviderConfig] = []
    @Published var apiKeys: [APIKeyInfo] = []
    @Published var providerModels: [String: [String]] = [:]
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var successMessage: String?
    
    // Modal state
    @Published var showingAPIKeySheet = false
    @Published var selectedProvider = ""
    @Published var apiKeyInput = ""
    
    private let baseURL = "http://localhost:8001"
    
    func loadData() {
        Task {
            await loadProviders()
            await loadAPIKeys()
            await loadAllModels()
        }
    }
    
    @MainActor
    func loadProviders() async {
        isLoading = true
        errorMessage = nil
        
        do {
            guard let url = URL(string: "\(baseURL)/config/providers") else {
                errorMessage = "Invalid URL"
                isLoading = false
                return
            }
            
            let (data, response) = try await URLSession.shared.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                errorMessage = "Server error"
                isLoading = false
                return
            }
            
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            
            if let success = json?["success"] as? Bool, success,
               let providersData = json?["providers"] as? [[String: Any]] {
                
                let decoder = JSONDecoder()
                let providers = try providersData.compactMap { providerDict -> ProviderConfig? in
                    let jsonData = try JSONSerialization.data(withJSONObject: providerDict)
                    return try decoder.decode(ProviderConfig.self, from: jsonData)
                }
                
                self.providers = providers
            } else {
                errorMessage = "Invalid response"
            }
        } catch {
            errorMessage = "Failed to load: \(error.localizedDescription)"
        }
        
        isLoading = false
    }
    
    @MainActor
    func loadAPIKeys() async {
        do {
            guard let url = URL(string: "\(baseURL)/config/api-keys") else { return }
            let (data, response) = try await URLSession.shared.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else { return }
            
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            
            if let success = json?["success"] as? Bool, success,
               let keysData = json?["api_keys"] as? [[String: Any]] {
                
                let decoder = JSONDecoder()
                let apiKeys = try keysData.compactMap { keyDict -> APIKeyInfo? in
                    let jsonData = try JSONSerialization.data(withJSONObject: keyDict)
                    return try decoder.decode(APIKeyInfo.self, from: jsonData)
                }
                
                self.apiKeys = apiKeys
            }
        } catch {
            // Silent failure for API keys
        }
    }
    
    @MainActor
    func loadAllModels() async {
        let providerNames = ["anthropic", "openai", "deepseek", "google", "ollama", "lm_studio"]
        
        for provider in providerNames {
            do {
                guard let url = URL(string: "\(baseURL)/config/models/\(provider)?refresh=true") else { continue }
                
                var request = URLRequest(url: url)
                request.httpMethod = "GET"
                request.setValue("application/json", forHTTPHeaderField: "Accept")
                request.timeoutInterval = 10.0
                
                let (data, response) = try await URLSession.shared.data(for: request)
                
                guard let httpResponse = response as? HTTPURLResponse else { continue }
                
                if httpResponse.statusCode == 200 {
                    let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    
                    if let success = json?["success"] as? Bool, success,
                       let models = json?["models"] as? [String] {
                        providerModels[provider] = models
                        print("Loaded \(models.count) models for \(provider): \(models.prefix(3))...")
                    } else {
                        print("Invalid response format for \(provider)")
                    }
                } else {
                    print("HTTP error \(httpResponse.statusCode) for \(provider)")
                }
            } catch {
                print("Error loading models for \(provider): \(error.localizedDescription)")
                continue
            }
        }
    }
    
    @MainActor
    func updateProvider(role: String, provider: String, model: String) async {
        do {
            guard let url = URL(string: "\(baseURL)/config/provider") else { return }
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            let payload = [
                "role": role,
                "provider_name": provider,
                "model": model,
                "server_address": ""
            ]
            
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
            
            let (data, _) = try await URLSession.shared.data(for: request)
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            
            if let success = json?["success"] as? Bool, success {
                successMessage = "Updated \(role) provider"
                await loadProviders()
            } else {
                errorMessage = "Failed to update provider"
            }
        } catch {
            errorMessage = "Error: \(error.localizedDescription)"
        }
    }
    
    @MainActor
    func setAPIKey(provider: String, key: String) async {
        do {
            guard let url = URL(string: "\(baseURL)/config/api-key") else { return }
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            let payload = ["provider": provider, "api_key": key]
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
            
            let (data, _) = try await URLSession.shared.data(for: request)
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            
            if let success = json?["success"] as? Bool, success {
                successMessage = "API key updated"
                await loadAPIKeys()
                await loadProviders()
            } else {
                errorMessage = "Failed to update API key"
            }
        } catch {
            errorMessage = "Error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Main View
struct ConfigurationView: View {
    @StateObject private var viewModel = ConfigurationViewModel()
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Configuration")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Spacer()
                
                Button("Refresh Models") {
                    Task {
                        await viewModel.loadAllModels()
                    }
                }
                .buttonStyle(.bordered)
                
                Button("Refresh All") {
                    viewModel.loadData()
                }
                .buttonStyle(.bordered)
            }
            .padding()
            
            // Messages
            if let error = viewModel.errorMessage {
                HStack {
                    Text("Error: \(error)")
                        .foregroundColor(.red)
                    Spacer()
                    Button("Dismiss") {
                        viewModel.errorMessage = nil
                    }
                }
                .padding()
                .background(Color.red.opacity(0.1))
            }
            
            if let success = viewModel.successMessage {
                HStack {
                    Text(success)
                        .foregroundColor(.green)
                    Spacer()
                    Button("Dismiss") {
                        viewModel.successMessage = nil
                    }
                }
                .padding()
                .background(Color.green.opacity(0.1))
            }
            
            // Content
            if viewModel.isLoading {
                VStack {
                    ProgressView()
                    Text("Loading...")
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                TabView {
                    ProvidersTab(viewModel: viewModel)
                        .tabItem {
                            Label("Providers", systemImage: "server.rack")
                        }
                    
                    APIKeysTab(viewModel: viewModel)
                        .tabItem {
                            Label("API Keys", systemImage: "key.fill")
                        }
                }
            }
        }
        .onAppear {
            viewModel.loadData()
        }
        .sheet(isPresented: $viewModel.showingAPIKeySheet) {
            APIKeySheet(viewModel: viewModel)
                .onDisappear {
                    viewModel.showingAPIKeySheet = false
                }
        }
    }
}

// MARK: - Providers Tab
struct ProvidersTab: View {
    @ObservedObject var viewModel: ConfigurationViewModel
    @State private var selectedModels: [String: String] = [:]
    
    var body: some View {
        List {
            ForEach(viewModel.providers) { provider in
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text(provider.display_name)
                            .font(.headline)
                        
                        Spacer()
                        
                        if provider.is_enabled {
                            Text("Active")
                                .foregroundColor(.green)
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 2)
                                .background(Color.green.opacity(0.2))
                                .cornerRadius(4)
                        }
                    }
                    
                    HStack {
                        Text("Model:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        
                        if let models = viewModel.providerModels[provider.name], !models.isEmpty {
                            Picker("Model", selection: Binding(
                                get: { selectedModels[provider.name] ?? provider.model },
                                set: { newModel in
                                    selectedModels[provider.name] = newModel
                                    let role = provider.is_enabled ? "main" : "fallback_api"
                                    Task {
                                        await viewModel.updateProvider(role: role, provider: provider.name, model: newModel)
                                    }
                                }
                            )) {
                                ForEach(models, id: \.self) { model in
                                    Text(model).tag(model)
                                }
                            }
                            .pickerStyle(.menu)
                            .frame(maxWidth: 300)
                        } else {
                            Text(provider.model)
                                .font(.system(.body, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                    }
                    
                    if provider.api_key_required {
                        HStack {
                            Text(provider.api_key_set ? "✓ API Key Set" : "⚠ API Key Missing")
                                .font(.caption)
                                .foregroundColor(provider.api_key_set ? .green : .red)
                            
                            if !provider.api_key_set {
                                Button("Add") {
                                    viewModel.selectedProvider = provider.name
                                    viewModel.showingAPIKeySheet = true
                                }
                                .buttonStyle(.borderedProminent)
                                .controlSize(.mini)
                            }
                        }
                    }
                }
                .padding(.vertical, 8)
            }
        }
        .onAppear {
            // Initialize selected models
            for provider in viewModel.providers {
                selectedModels[provider.name] = provider.model
            }
        }
        .onChange(of: viewModel.providers) { _, _ in
            // Update selected models when providers change
            for provider in viewModel.providers {
                if selectedModels[provider.name] == nil {
                    selectedModels[provider.name] = provider.model
                }
            }
        }
    }
}

// MARK: - API Keys Tab
struct APIKeysTab: View {
    @ObservedObject var viewModel: ConfigurationViewModel
    
    var body: some View {
        List {
            if viewModel.apiKeys.isEmpty {
                VStack {
                    Text("No API Keys")
                        .font(.headline)
                    Text("API keys will appear here")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding()
            } else {
                ForEach(viewModel.apiKeys) { key in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(key.display_name)
                                .font(.headline)
                            
                            if let updated = key.last_updated {
                                Text("Updated: \(formatDate(updated))")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        
                        Spacer()
                        
                        Text(key.is_set ? "Set" : "Not Set")
                            .foregroundColor(key.is_set ? .green : .red)
                            .font(.caption)
                        
                        Button(key.is_set ? "Update" : "Add") {
                            viewModel.selectedProvider = key.provider
                            viewModel.showingAPIKeySheet = true
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    }
                }
            }
        }
    }
    
    private func formatDate(_ dateString: String) -> String {
        let formatter = ISO8601DateFormatter()
        if let date = formatter.date(from: dateString) {
            let displayFormatter = DateFormatter()
            displayFormatter.dateStyle = .short
            displayFormatter.timeStyle = .short
            return displayFormatter.string(from: date)
        }
        return dateString
    }
}

// MARK: - API Key Sheet
struct APIKeySheet: View {
    @ObservedObject var viewModel: ConfigurationViewModel
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                VStack(alignment: .leading) {
                    Text("API Key")
                        .font(.headline)
                    
                    SecureField("Enter your API key", text: $viewModel.apiKeyInput)
                        .textFieldStyle(.roundedBorder)
                }
                
                Text("Your API key will be stored securely.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
            }
            .padding()
            .navigationTitle("\(viewModel.selectedProvider.capitalized) API Key")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        viewModel.apiKeyInput = ""
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        Task {
                            await viewModel.setAPIKey(
                                provider: viewModel.selectedProvider,
                                key: viewModel.apiKeyInput
                            )
                            viewModel.apiKeyInput = ""
                            dismiss()
                        }
                    }
                    .disabled(viewModel.apiKeyInput.isEmpty)
                }
            }
        }
        .frame(minWidth: 400, minHeight: 250)
    }
}

#Preview {
    ConfigurationView()
        .frame(width: 700, height: 600)
}
