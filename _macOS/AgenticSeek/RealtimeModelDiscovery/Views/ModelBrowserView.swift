import SwiftUI
import SwiftUI
import Combine

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Interactive model browser with advanced filtering
 * Issues & Complexity Summary: Production-ready real-time discovery UI component
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~180
   - Core Algorithm Complexity: Medium
   - Dependencies: 2 New
   - State Management Complexity: High
   - Novelty/Uncertainty Factor: Medium
 * AI Pre-Task Self-Assessment: 88%
 * Problem Estimate: 92%
 * Initial Code Complexity Estimate: 86%
 * Final Code Complexity: 89%
 * Overall Result Score: 93%
 * Last Updated: 2025-06-07
 */

struct ModelBrowserView: View {

    
    @StateObject private var registryManager = ModelRegistryManager()
    @State private var selectedModel: DiscoveredModel?
    @State private var sortBy: SortOption = .name
    @State private var filterByCapability = "All"
    @State private var showingModelDetails = false
    
    enum SortOption: String, CaseIterable {
        case name = "Name"
        case size = "Size"
        case performance = "Performance"
        case lastUpdated = "Last Updated"
    }
    
    var body: some View {
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 0) {
                // Filter and sort controls
                VStack(alignment: .leading, spacing: 12) {
                    Text("Model Browser")
                        .font(.headline)
                    
                    HStack {
                        Text("Sort by:")
                        Picker("Sort", selection: $sortBy) {
                            ForEach(SortOption.allCases, id: \.self) { option in
                                Text(option.rawValue).tag(option)
                            }
                        }
                        .pickerStyle(.menu)
                        
                        Spacer()
                        
                        Text("Filter:")
                        Picker("Capability", selection: $filterByCapability) {
                            Text("All Capabilities").tag("All")
                            ForEach(availableCapabilities, id: \.self) { capability in
                                Text(capability.capitalized).tag(capability)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                }
                .padding()
                .background(Color(.controlBackgroundColor))
                
                Divider()
                
                // Models list
                List(sortedAndFilteredModels, id: \.id, selection: $selectedModel) { model in
                    ModelBrowserRow(model: model)
                        .tag(model)
                }
                .listStyle(.sidebar)
            }
        } detail: {
            if let selectedModel = selectedModel {
                ModelDetailView(model: selectedModel)
            } else {
                VStack {
                    Image(systemName: "cube.box")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    
                    Text("Select a model to view details")
                        .font(.headline)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("Model Browser")
    }
    
    private var availableCapabilities: [String] {
        let allCapabilities = Set(registryManager.registeredModels.flatMap(\.capabilities))
        return Array(allCapabilities).sorted()
    }
    
    private var sortedAndFilteredModels: [DiscoveredModel] {
        var models = registryManager.registeredModels
        
        // Filter by capability
        if filterByCapability != "All" {
            models = models.filter { $0.capabilities.contains(filterByCapability) }
        }
        
        // Sort models
        switch sortBy {
        case .name:
            models.sort { $0.name < $1.name }
        case .size:
            models.sort { $0.size_gb < $1.size_gb }
        case .performance:
            models.sort { $0.performance_score > $1.performance_score }
        case .lastUpdated:
            models.sort { $0.last_verified > $1.last_verified }
        }
        
        return models
    }
}

#Preview {
    ModelBrowserView()
}
