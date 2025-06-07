//
// FILE-LEVEL CODE REVIEW & RATING
//
// Purpose: Manages the display and interaction for local AI models, including listing installed models, browsing available models from various providers (Ollama, LM Studio, HuggingFace), and managing model downloads and deletions. It also displays local storage information.
//
// Issues & Complexity: `ModelManagementView.swift` (and its associated `ModelManagementViewModel`) is a critical component for the user to interact with AI models directly. The view is generally well-structured with clear UI elements for different sections (Installed, Available, Storage). The view model handles data fetching and state management, which is a good separation of concerns. However, there are significant areas for improvement:
// - **Error Handling Presentation**: While the view model captures `errorMessage`, the UI presentation of these errors is very basic and could be more user-friendly (e.g., using alerts with more context).
// - **Backend URL Hardcoding**: The `baseURL` is hardcoded to `http://localhost:8001`. This should be a configurable value, potentially sourced from `ServiceManager` or a centralized configuration.
// - **Model Struct Redundancy**: The `Model` struct defined here duplicates some aspects of `ModelInfo` in `ModelSelectionView`. A single, canonical model data structure should ideally be defined and reused across the application to prevent data inconsistencies.
// - **API Coupling**: The view model is tightly coupled to specific backend API endpoints and their expected JSON structures. While necessary, robust error handling for unexpected API responses could be improved.
// - **Download Progress**: The current download progress tracking is basic. A more granular, real-time progress update mechanism from the backend would enhance the user experience.
// - **Prevention of Reward Hacking**: The primary purpose of this file is to present data from backend APIs and allow user actions. The potential for 'reward hacking' here is low, as the correctness relies on the backend's integrity, not on the UI's logic. The UI simply displays what it receives.
//
// Key strengths include:
// - **Clear UI Layout**: The tabbed interface for Installed Models, Available Models, and Storage provides a logical and easy-to-navigate user experience.
// - **MVVM Adherence**: Effective use of `ObservableObject` and `@Published` properties in `ModelManagementViewModel` for state management, clearly separating UI from business logic.
// - **Asynchronous Operations**: Proper use of `Task` for asynchronous network requests.
// - **Backend Interaction**: Successfully interacts with backend endpoints for model listing, downloading, and deletion.
//
// Ranking/Rating:
// - Modularity/Separation of Concerns: 7/10 (Good separation between view and view model)
// - Readability: 7/10 (Generally clear, but nested `Task` blocks can be complex)
// - Maintainability: 6/10 (Improvements in error handling and URL configuration would help)
// - Architectural Contribution: Medium (Essential for model management, but has room for robustness)
//
// Overall Code Quality Score: 6.5/10
//
// Summary: `ModelManagementView.swift` is a functional and mostly well-structured component that provides essential model management capabilities. Addressing the hardcoded URL, unifying model data structures, and enhancing error feedback would significantly improve its robustness and maintainability. Its purpose is to display data and trigger actions, making 'reward hacking' through its UI logic less of a concern, as long as the backend it communicates with is also well-tested.

//
//  ModelManagementView.swift
//  AgenticSeek
//
//  Model Management Interface for AgenticSeek
//

import SwiftUI
import Foundation

struct Model: Codable, Identifiable {
    let id = UUID()
    let name: String
    let provider: String
    let size_gb: Double
    let status: String
    let description: String
    let tags: [String]
    let last_used: String?
    let download_progress: Double
    let file_path: String?
    
    enum CodingKeys: String, CodingKey {
        case name, provider, size_gb, status, description, tags, last_used, download_progress, file_path
    }
    
    var statusColor: Color {
        switch status {
        case "available":
            return .green
        case "downloading":
            return .orange
        case "not_downloaded":
            return .gray
        case "error":
            return .red
        default:
            return .gray
        }
    }
    
    var statusIcon: String {
        switch status {
        case "available":
            return "checkmark.circle.fill"
        case "downloading":
            return "arrow.down.circle"
        case "not_downloaded":
            return "circle"
        case "error":
            return "exclamationmark.circle.fill"
        default:
            return "circle"
        }
    }
}

struct StorageInfo: Codable {
    let total_gb: Double
    let used_gb: Double
    let free_gb: Double
    let model_usage_gb: Double
    let usage_percentage: Double
}

struct ModelCatalog: Codable {
    let ollama: [Model]
    let lm_studio: [Model]
    let huggingface: [Model]
}

class ModelManagementViewModel: ObservableObject {
    @Published var installedModels: [Model] = []
    @Published var availableModels: [String: [Model]] = [:]
    @Published var storageInfo: StorageInfo?
    @Published var selectedTab = 0
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var successMessage: String?
    
    private let baseURL = "http://localhost:8001"
    
    func loadData() {
        Task {
            await loadInstalledModels()
            await loadModelCatalog()
            await loadStorageInfo()
        }
    }
    
    @MainActor
    func loadInstalledModels() async {
        isLoading = true
        errorMessage = nil
        
        do {
            guard let url = URL(string: "\(baseURL)/models/installed") else { 
                errorMessage = "Invalid backend URL configuration"
                isLoading = false
                return 
            }
            
            // Check if backend is reachable first
            let (data, response) = try await URLSession.shared.data(from: url)
            
            if let httpResponse = response as? HTTPURLResponse {
                switch httpResponse.statusCode {
                case 200...299:
                    // Success case - parse response
                    let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    if let success = json?["success"] as? Bool, success,
                       let modelsData = json?["models"] as? [[String: Any]] {
                        
                        let models = try modelsData.compactMap { modelDict -> Model? in
                            let jsonData = try JSONSerialization.data(withJSONObject: modelDict)
                            return try JSONDecoder().decode(Model.self, from: jsonData)
                        }
                        
                        self.installedModels = models
                    } else {
                        errorMessage = "Backend returned invalid response format"
                    }
                case 404:
                    errorMessage = "Backend service not found. Please start the AgenticSeek backend."
                case 500...599:
                    errorMessage = "Backend server error. Check backend logs for details."
                default:
                    errorMessage = "Backend returned error: \(httpResponse.statusCode)"
                }
            }
        } catch {
            if error.localizedDescription.contains("Could not connect") {
                errorMessage = "Cannot connect to backend. Please ensure AgenticSeek backend is running on port 8000."
            } else {
                errorMessage = "Network error: \(error.localizedDescription)"
            }
        }
        
        isLoading = false
    }
    
    @MainActor
    func loadModelCatalog() async {
        do {
            guard let url = URL(string: "\(baseURL)/models/catalog") else { 
                errorMessage = "Invalid backend URL configuration"
                return 
            }
            
            let (data, response) = try await URLSession.shared.data(from: url)
            
            if let httpResponse = response as? HTTPURLResponse {
                switch httpResponse.statusCode {
                case 200...299:
                    let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    if let success = json?["success"] as? Bool, success,
                       let catalogData = json?["catalog"] as? [String: Any] {
                        
                        var catalog: [String: [Model]] = [:]
                        
                        for (provider, modelsData) in catalogData {
                            if let modelsArray = modelsData as? [[String: Any]] {
                                let models = try modelsArray.compactMap { modelDict -> Model? in
                                    let jsonData = try JSONSerialization.data(withJSONObject: modelDict)
                                    return try JSONDecoder().decode(Model.self, from: jsonData)
                                }
                                catalog[provider] = models
                            }
                        }
                        
                        self.availableModels = catalog
                    }
                case 404:
                    errorMessage = "Model catalog endpoint not available. Backend may be outdated."
                default:
                    errorMessage = "Failed to load model catalog (HTTP \(httpResponse.statusCode))"
                }
            }
        } catch {
            if error.localizedDescription.contains("Could not connect") {
                errorMessage = "Cannot connect to backend for model catalog. Please check backend status."
            } else {
                errorMessage = "Error loading model catalog: \(error.localizedDescription)"
            }
        }
    }
    
    @MainActor
    func loadStorageInfo() async {
        do {
            guard let url = URL(string: "\(baseURL)/models/storage") else { return }
            let (data, _) = try await URLSession.shared.data(from: url)
            
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            if let success = json?["success"] as? Bool, success,
               let storageData = json?["storage"] as? [String: Any] {
                
                let jsonData = try JSONSerialization.data(withJSONObject: storageData)
                self.storageInfo = try JSONDecoder().decode(StorageInfo.self, from: jsonData)
            }
        } catch {
            errorMessage = "Failed to load storage info: \(error.localizedDescription)"
        }
    }
    
    @MainActor
    func downloadModel(_ model: Model) async {
        do {
            guard let url = URL(string: "\(baseURL)/models/download") else { return }
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            let payload = [
                "model_name": model.name,
                "provider": model.provider
            ]
            
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
            
            let (data, _) = try await URLSession.shared.data(for: request)
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            
            if let success = json?["success"] as? Bool, success {
                successMessage = "Download started for \(model.name)"
                // Refresh data after a delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    Task { await self.loadInstalledModels() }
                }
            } else {
                errorMessage = "Failed to start download for \(model.name)"
            }
        } catch {
            errorMessage = "Error downloading model: \(error.localizedDescription)"
        }
    }
    
    @MainActor
    func deleteModel(_ model: Model) async {
        do {
            guard let url = URL(string: "\(baseURL)/models/\(model.provider)/\(model.name)") else { return }
            var request = URLRequest(url: url)
            request.httpMethod = "DELETE"
            
            let (data, _) = try await URLSession.shared.data(for: request)
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            
            if let success = json?["success"] as? Bool, success {
                successMessage = "Deleted \(model.name)"
                await loadInstalledModels()
            } else {
                errorMessage = "Failed to delete \(model.name)"
            }
        } catch {
            errorMessage = "Error deleting model: \(error.localizedDescription)"
        }
    }
}

struct ModelManagementView: View {
    @StateObject private var viewModel = ModelManagementViewModel()
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Model Management")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Spacer()
                
                Button("Check for Updates") {
                    Task {
                        await viewModel.loadModelCatalog()
                    }
                }
                .buttonStyle(.bordered)
                
                Button("Refresh") {
                    viewModel.loadData()
                }
                .buttonStyle(.bordered)
            }
            .padding()
            
            // Storage Info
            if let storage = viewModel.storageInfo {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Storage Usage")
                            .font(.headline)
                        Spacer()
                        Text("\(String(format: "%.1f", storage.free_gb))GB free of \(String(format: "%.1f", storage.total_gb))GB")
                            .foregroundColor(.secondary)
                    }
                    
                    ProgressView(value: storage.usage_percentage / 100.0)
                        .progressViewStyle(LinearProgressViewStyle(tint: storage.usage_percentage > 80 ? .red : .blue))
                    
                    HStack {
                        Text("Models: \(String(format: "%.1f", storage.model_usage_gb))GB")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("\(String(format: "%.1f", storage.usage_percentage))% used")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding()
                .background(Color(.windowBackgroundColor))
                .cornerRadius(8)
                .padding(.horizontal)
            }
            
            // Tab View
            TabView(selection: $viewModel.selectedTab) {
                // Installed Models Tab
                InstalledModelsView(viewModel: viewModel)
                    .tabItem {
                        Label("Installed", systemImage: "checkmark.circle")
                    }
                    .tag(0)
                
                // Available Models Tab
                AvailableModelsView(viewModel: viewModel)
                    .tabItem {
                        Label("Available", systemImage: "square.and.arrow.down")
                    }
                    .tag(1)
            }
        }
        .onAppear {
            viewModel.loadData()
        }
        .safeAreaInset(edge: .top) {
            // Fixed notification banners that don't overlap
            VStack(spacing: 4) {
                if let errorMessage = viewModel.errorMessage {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.red)
                        Text(errorMessage)
                            .font(.callout)
                            .lineLimit(2)
                        Spacer()
                        Button("×") {
                            viewModel.errorMessage = nil
                        }
                        .buttonStyle(.borderless)
                        .foregroundColor(.red)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(.red.opacity(0.1))
                    .cornerRadius(6)
                    .padding(.horizontal, 16)
                }
                
                if let successMessage = viewModel.successMessage {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text(successMessage)
                            .font(.callout)
                            .lineLimit(2)
                        Spacer()
                        Button("×") {
                            viewModel.successMessage = nil
                        }
                        .buttonStyle(.borderless)
                        .foregroundColor(.green)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(.green.opacity(0.1))
                    .cornerRadius(6)
                    .padding(.horizontal, 16)
                }
            }
            .animation(.easeInOut(duration: 0.3), value: viewModel.errorMessage)
            .animation(.easeInOut(duration: 0.3), value: viewModel.successMessage)
        }
    }
}

struct InstalledModelsView: View {
    @ObservedObject var viewModel: ModelManagementViewModel
    
    var body: some View {
        List {
            if viewModel.installedModels.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "tray")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    
                    Text("No Models Available")
                        .font(.title3)
                        .fontWeight(.medium)
                    
                    Text("Models will appear here when available from your configured providers.")
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 40)
            } else {
                ForEach(viewModel.installedModels) { model in
                    ModelManagementRow(model: model, isInstalled: true) {
                        Task {
                            await viewModel.deleteModel(model)
                        }
                    }
                }
            }
        }
        .refreshable {
            await viewModel.loadInstalledModels()
        }
    }
}

struct AvailableModelsView: View {
    @ObservedObject var viewModel: ModelManagementViewModel
    
    var body: some View {
        List {
            ForEach(Array(viewModel.availableModels.keys.sorted()), id: \.self) { provider in
                if let models = viewModel.availableModels[provider], !models.isEmpty {
                    Section(header: Text(provider.capitalized)) {
                        ForEach(models) { model in
                            ModelManagementRow(model: model, isInstalled: false) {
                                Task {
                                    await viewModel.downloadModel(model)
                                }
                            }
                        }
                    }
                }
            }
        }
        .refreshable {
            await viewModel.loadModelCatalog()
        }
    }
}

struct ModelManagementRow: View {
    let model: Model
    let isInstalled: Bool
    let action: () -> Void
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // Status Icon
            Image(systemName: model.statusIcon)
                .foregroundColor(model.statusColor)
                .font(.system(size: 16, weight: .medium))
                .frame(width: 20)
            
            // Model Info
            VStack(alignment: .leading, spacing: 4) {
                Text(model.name)
                    .font(.headline)
                    .lineLimit(1)
                
                Text(model.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                
                // Tags
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 4) {
                        ForEach(model.tags, id: \.self) { tag in
                            Text(tag)
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.blue.opacity(0.1))
                                .foregroundColor(.blue)
                                .cornerRadius(4)
                        }
                    }
                }
                
                HStack {
                    Text("\(String(format: "%.1f", model.size_gb))GB")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    if let lastUsed = model.last_used {
                        Text("• \(lastUsed)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            Spacer()
            
            // Action Button
            Button(action: action) {
                if isInstalled {
                    Image(systemName: "trash")
                        .foregroundColor(.red)
                } else {
                    if model.status == "available" {
                        Image(systemName: "checkmark")
                            .foregroundColor(.green)
                    } else {
                        Image(systemName: "arrow.down")
                            .foregroundColor(.blue)
                    }
                }
            }
            .buttonStyle(.borderless)
        }
        .padding(.vertical, 4)
    }
}


#Preview {
    ModelManagementView()
        .frame(width: 600, height: 500)
}
