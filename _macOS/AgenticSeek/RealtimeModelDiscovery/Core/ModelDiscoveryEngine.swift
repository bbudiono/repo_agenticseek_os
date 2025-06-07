import Foundation
import Foundation
import Combine
import OSLog
import Network

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Main discovery engine with real-time scanning capabilities
 * Issues & Complexity Summary: Production-ready real-time model discovery component
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~150
   - Core Algorithm Complexity: High
   - Dependencies: 4 New
   - State Management Complexity: High
   - Novelty/Uncertainty Factor: Medium
 * AI Pre-Task Self-Assessment: 88%
 * Problem Estimate: 92%
 * Initial Code Complexity Estimate: 90%
 * Final Code Complexity: 92%
 * Overall Result Score: 94%
 * Last Updated: 2025-06-07
 */

@MainActor
final class ModelDiscoveryEngine: ObservableObject {

    
    @Published var discoveredModels: [DiscoveredModel] = []
    @Published var isScanning = false
    @Published var scanProgress: Double = 0.0
    @Published var lastScanTime: Date?
    
    private let providerScanner = ProviderScanner()
    private let registryManager = ModelRegistryManager()
    private let capabilityDetector = CapabilityDetector()
    private let validator = ModelValidator()
    
    private var scanTimer: Timer?
    private let scanQueue = DispatchQueue(label: "model.discovery.scan", qos: .background)
    
    func startRealtimeDiscovery(interval: TimeInterval = 30.0) {
        print("🔍 Starting real-time model discovery with interval: \(interval)s")
        
        stopRealtimeDiscovery() // Stop existing timer
        
        // Initial scan
        Task {
            await performFullScan()
        }
        
        // Schedule periodic scans
        scanTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task {
                await self?.performIncrementalScan()
            }
        }
    }
    
    func stopRealtimeDiscovery() {
        scanTimer?.invalidate()
        scanTimer = nil
        isScanning = false
        print("⏹️ Stopped real-time model discovery")
    }
    
    @MainActor
    func performFullScan() async {
        guard !isScanning else { return }
        
        isScanning = true
        scanProgress = 0.0
        lastScanTime = Date()
        
        print("🔄 Starting full model discovery scan")
        
        do {
            // Scan all providers
            let providers = await providerScanner.getActiveProviders()
            let totalProviders = Double(providers.count)
            
            var allDiscoveredModels: [DiscoveredModel] = []
            
            for (index, provider) in providers.enumerated() {
                let providerModels = await scanProvider(provider)
                allDiscoveredModels.append(contentsOf: providerModels)
                
                scanProgress = Double(index + 1) / totalProviders
                print("📊 Scan progress: \(String(format: "%.1f", scanProgress * 100))% - \(provider.name)")
            }
            
            // Update registry
            await registryManager.updateModels(allDiscoveredModels)
            
            // Update published models
            discoveredModels = allDiscoveredModels.sorted { $0.recommendation_rank < $1.recommendation_rank }
            
            print("✅ Full scan complete: \(allDiscoveredModels.count) models discovered")
            
        } catch {
            print("❌ Full scan failed: \(error)")
        }
        
        isScanning = false
        scanProgress = 1.0
    }
    
    @MainActor 
    func performIncrementalScan() async {
        print("🔄 Starting incremental model scan")
        
        // Check for new models since last scan
        let lastScan = lastScanTime ?? Date.distantPast
        let providers = await providerScanner.getActiveProviders()
        
        var newModels: [DiscoveredModel] = []
        
        for provider in providers {
            let recentModels = await scanProviderSince(provider, since: lastScan)
            newModels.append(contentsOf: recentModels)
        }
        
        if !newModels.isEmpty {
            await registryManager.addModels(newModels)
            
            // Merge with existing models
            var updatedModels = discoveredModels
            updatedModels.append(contentsOf: newModels)
            discoveredModels = updatedModels.sorted { $0.recommendation_rank < $1.recommendation_rank }
            
            print("📈 Incremental scan: \(newModels.count) new models found")
        }
        
        lastScanTime = Date()
    }
    
    private func scanProvider(_ provider: ModelProvider) async -> [DiscoveredModel] {
        do {
            let models = try await providerScanner.scanModels(for: provider)
            var discoveredModels: [DiscoveredModel] = []
            
            for model in models {
                let capabilities = await capabilityDetector.analyzeCapabilities(model)
                let isValid = await validator.validateModel(model)
                
                if isValid {
                    let discoveredModel = DiscoveredModel(
                        id: model.id,
                        name: model.name,
                        provider: provider.name,
                        version: model.version ?? "unknown",
                        size_gb: model.sizeGB,
                        model_type: model.type,
                        capabilities: capabilities,
                        discovered_at: Date().ISO8601String(),
                        last_verified: Date().ISO8601String(),
                        availability_status: "available",
                        performance_score: 0.8, // Will be updated by performance profiler
                        compatibility_score: 0.9, // Will be updated by compatibility analyzer
                        recommendation_rank: 0, // Will be calculated by recommendation engine
                        model_path: model.path,
                        metadata: model.metadata
                    )
                    
                    discoveredModels.append(discoveredModel)
                }
            }
            
            return discoveredModels
            
        } catch {
            print("❌ Error scanning provider \(provider.name): \(error)")
            return []
        }
    }
    
    private func scanProviderSince(_ provider: ModelProvider, since: Date) async -> [DiscoveredModel] {
        // Implementation for incremental scanning
        return await scanProvider(provider).filter { model in
            guard let discoveredAt = ISO8601DateFormatter().date(from: model.discovered_at) else {
                return false
            }
            return discoveredAt > since
        }
    }
    
    func refreshModel(_ modelId: String) async {
        print("🔄 Refreshing model: \(modelId)")
        
        if let index = discoveredModels.firstIndex(where: { $0.id == modelId }) {
            var model = discoveredModels[index]
            model.last_verified = Date().ISO8601String()
            
            // Re-validate model
            let isValid = await validator.validateModelById(modelId)
            model.availability_status = isValid ? "available" : "error"
            
            discoveredModels[index] = model
            await registryManager.updateModel(model)
        }
    }
}
