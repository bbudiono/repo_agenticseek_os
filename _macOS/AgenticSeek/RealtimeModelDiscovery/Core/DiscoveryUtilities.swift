import Foundation
import Combine
import Network
import OSLog

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Utility classes and helpers for MLACS Real-time Model Discovery
 * Issues & Complexity Summary: Helper utilities for discovery operations and analysis
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~200
   - Core Algorithm Complexity: Medium
   - Dependencies: 4 New
   - State Management Complexity: Medium
   - Novelty/Uncertainty Factor: Low
 * AI Pre-Task Self-Assessment: 90%
 * Problem Estimate: 92%
 * Initial Code Complexity Estimate: 88%
 * Final Code Complexity: 90%
 * Overall Result Score: 93%
 * Last Updated: 2025-01-07
 */

// MARK: - Discovery Event Logger

@MainActor
final class DiscoveryEventLogger: ObservableObject {
    
    @Published var events: [DiscoveryEvent] = []
    @Published var isLoggingEnabled = true
    
    private let logger = Logger(subsystem: "AgenticSeek", category: "ModelDiscovery")
    private let maxEvents = 1000
    
    func logEvent(
        type: DiscoveryEvent.EventType,
        message: String,
        modelId: String? = nil,
        providerId: String? = nil,
        severity: DiscoveryEvent.Severity = .medium,
        details: [String: String] = [:]
    ) {
        guard isLoggingEnabled else { return }
        
        let event = DiscoveryEvent(
            timestamp: Date(),
            type: type,
            modelId: modelId,
            providerId: providerId,
            message: message,
            details: details,
            severity: severity
        )
        
        events.insert(event, at: 0)
        
        // Keep only the most recent events
        if events.count > maxEvents {
            events.removeLast(events.count - maxEvents)
        }
        
        // Log to system
        switch severity {
        case .low:
            logger.info("\(message)")
        case .medium:
            logger.notice("\(message)")
        case .high:
            logger.warning("\(message)")
        case .critical:
            logger.error("\(message)")
        }
        
        print("📋 [\(type.rawValue.uppercased())] \(message)")
    }
    
    func clearEvents() {
        events.removeAll()
        logger.info("Discovery events cleared")
    }
    
    func getEventsByType(_ type: DiscoveryEvent.EventType) -> [DiscoveryEvent] {
        return events.filter { $0.type == type }
    }
    
    func getEventsBySeverity(_ severity: DiscoveryEvent.Severity) -> [DiscoveryEvent] {
        return events.filter { $0.severity == severity }
    }
    
    func getEventsForModel(_ modelId: String) -> [DiscoveryEvent] {
        return events.filter { $0.modelId == modelId }
    }
    
    func exportEvents() -> Data? {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = .prettyPrinted
            return try encoder.encode(events)
        } catch {
            logger.error("Failed to export events: \(error)")
            return nil
        }
    }
}

// MARK: - Network Connectivity Monitor

final class NetworkConnectivityMonitor: ObservableObject {
    
    @Published var isConnected = false
    @Published var connectionType: NWInterface.InterfaceType?
    @Published var isExpensive = false
    
    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "NetworkMonitor")
    
    init() {
        startMonitoring()
    }
    
    deinit {
        stopMonitoring()
    }
    
    private func startMonitoring() {
        monitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.isConnected = path.status == .satisfied
                self?.connectionType = path.availableInterfaces.first?.type
                self?.isExpensive = path.isExpensive
            }
        }
        
        monitor.start(queue: queue)
    }
    
    private func stopMonitoring() {
        monitor.cancel()
    }
    
    func canPerformRemoteDiscovery() -> Bool {
        return isConnected && !isExpensive
    }
}

// MARK: - Provider Health Checker

@MainActor
final class ProviderHealthChecker: ObservableObject {
    
    @Published var providerStatuses: [String: ModelProvider.ProviderStatus] = [:]
    
    private let logger = Logger(subsystem: "AgenticSeek", category: "ProviderHealth")
    private var healthCheckTimer: Timer?
    
    func startHealthChecking(interval: TimeInterval = 60.0) {
        healthCheckTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task {
                await self?.performHealthChecks()
            }
        }
        
        // Initial check
        Task {
            await performHealthChecks()
        }
    }
    
    func stopHealthChecking() {
        healthCheckTimer?.invalidate()
        healthCheckTimer = nil
    }
    
    private func performHealthChecks() async {
        let providers = getKnownProviders()
        
        for provider in providers {
            let status = await checkProviderHealth(provider)
            providerStatuses[provider.name] = status
            
            logger.info("Provider \(provider.name) status: \(status.rawValue)")
        }
    }
    
    private func checkProviderHealth(_ provider: ModelProvider) async -> ModelProvider.ProviderStatus {
        do {
            // Create URL request with timeout
            guard let url = URL(string: provider.endpoint) else {
                return .error
            }
            
            var request = URLRequest(url: url)
            request.timeoutInterval = 5.0
            
            // Perform health check based on provider type
            switch provider.name.lowercased() {
            case "ollama":
                return await checkOllamaHealth(request)
            case "lm_studio":
                return await checkLMStudioHealth(request)
            default:
                return await checkGenericHealth(request)
            }
            
        } catch {
            logger.error("Health check failed for \(provider.name): \(error)")
            return .error
        }
    }
    
    private func checkOllamaHealth(_ request: URLRequest) async -> ModelProvider.ProviderStatus {
        do {
            // Check Ollama's /api/tags endpoint
            var healthRequest = request
            healthRequest.url = request.url?.appendingPathComponent("/api/tags")
            
            let (_, response) = try await URLSession.shared.data(for: healthRequest)
            
            if let httpResponse = response as? HTTPURLResponse {
                return httpResponse.statusCode == 200 ? .active : .inactive
            }
            
            return .unknown
            
        } catch {
            return .error
        }
    }
    
    private func checkLMStudioHealth(_ request: URLRequest) async -> ModelProvider.ProviderStatus {
        do {
            // Check LM Studio's /v1/models endpoint
            var healthRequest = request
            healthRequest.url = request.url?.appendingPathComponent("/v1/models")
            
            let (_, response) = try await URLSession.shared.data(for: healthRequest)
            
            if let httpResponse = response as? HTTPURLResponse {
                return httpResponse.statusCode == 200 ? .active : .inactive
            }
            
            return .unknown
            
        } catch {
            return .error
        }
    }
    
    private func checkGenericHealth(_ request: URLRequest) async -> ModelProvider.ProviderStatus {
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            
            if let httpResponse = response as? HTTPURLResponse {
                return (200...299).contains(httpResponse.statusCode) ? .active : .inactive
            }
            
            return .unknown
            
        } catch {
            return .error
        }
    }
    
    private func getKnownProviders() -> [ModelProvider] {
        return [
            ModelProvider(
                name: "Ollama",
                endpoint: "http://localhost:11434",
                type: .local,
                isActive: true,
                lastChecked: Date(),
                status: .unknown,
                supportedFormats: ["GGUF", "GGML"],
                capabilities: ["text-generation", "chat-completion"]
            ),
            ModelProvider(
                name: "LM Studio",
                endpoint: "http://localhost:1234",
                type: .local,
                isActive: true,
                lastChecked: Date(),
                status: .unknown,
                supportedFormats: ["GGUF", "GGML"],
                capabilities: ["text-generation", "chat-completion"]
            )
        ]
    }
    
    func getProviderStatus(_ providerName: String) -> ModelProvider.ProviderStatus {
        return providerStatuses[providerName] ?? .unknown
    }
    
    func isProviderHealthy(_ providerName: String) -> Bool {
        return getProviderStatus(providerName) == .active
    }
}

// MARK: - Discovery Performance Analyzer

final class DiscoveryPerformanceAnalyzer {
    
    private var sessionMetrics: [String: TimeInterval] = [:]
    private var discoveryTrends: [Date: Int] = [:]
    
    func recordScanDuration(_ duration: TimeInterval, for sessionId: String) {
        sessionMetrics[sessionId] = duration
    }
    
    func recordModelsDiscovered(_ count: Int, at date: Date = Date()) {
        let dayStart = Calendar.current.startOfDay(for: date)
        discoveryTrends[dayStart] = (discoveryTrends[dayStart] ?? 0) + count
    }
    
    func getAverageScanDuration() -> TimeInterval {
        guard !sessionMetrics.isEmpty else { return 0 }
        let total = sessionMetrics.values.reduce(0, +)
        return total / Double(sessionMetrics.count)
    }
    
    func getDiscoveryTrend(days: Int = 7) -> [Date: Int] {
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -days, to: Date()) ?? Date()
        return discoveryTrends.filter { $0.key >= cutoffDate }
    }
    
    func getTotalModelsDiscovered() -> Int {
        return discoveryTrends.values.reduce(0, +)
    }
    
    func getPerformanceReport() -> DiscoveryPerformanceReport {
        return DiscoveryPerformanceReport(
            totalSessions: sessionMetrics.count,
            averageDuration: getAverageScanDuration(),
            totalModelsDiscovered: getTotalModelsDiscovered(),
            trendData: getDiscoveryTrend(),
            fastestScan: sessionMetrics.values.min() ?? 0,
            slowestScan: sessionMetrics.values.max() ?? 0
        )
    }
}

struct DiscoveryPerformanceReport {
    let totalSessions: Int
    let averageDuration: TimeInterval
    let totalModelsDiscovered: Int
    let trendData: [Date: Int]
    let fastestScan: TimeInterval
    let slowestScan: TimeInterval
    
    var formattedAverageDuration: String {
        return String(format: "%.1f seconds", averageDuration)
    }
    
    var formattedFastestScan: String {
        return String(format: "%.1f seconds", fastestScan)
    }
    
    var formattedSlowestScan: String {
        return String(format: "%.1f seconds", slowestScan)
    }
}

// MARK: - File System Utilities

final class DiscoveryFileSystemUtils {
    
    static func getModelDirectories() -> [URL] {
        var directories: [URL] = []
        
        // Common Ollama directories
        if let ollamaDir = getOllamaDirectory() {
            directories.append(ollamaDir)
        }
        
        // Common LM Studio directories
        if let lmStudioDir = getLMStudioDirectory() {
            directories.append(lmStudioDir)
        }
        
        // User's Downloads directory
        if let downloadsDir = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first {
            directories.append(downloadsDir)
        }
        
        return directories
    }
    
    static func getOllamaDirectory() -> URL? {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let ollamaPath = homeDir.appendingPathComponent(".ollama/models")
        
        if FileManager.default.fileExists(atPath: ollamaPath.path) {
            return ollamaPath
        }
        
        return nil
    }
    
    static func getLMStudioDirectory() -> URL? {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let lmStudioPath = homeDir.appendingPathComponent(".cache/lm-studio/models")
        
        if FileManager.default.fileExists(atPath: lmStudioPath.path) {
            return lmStudioPath
        }
        
        return nil
    }
    
    static func scanForModelFiles(in directory: URL) -> [URL] {
        var modelFiles: [URL] = []
        
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) else {
            return modelFiles
        }
        
        for case let fileURL as URL in enumerator {
            let pathExtension = fileURL.pathExtension.lowercased()
            
            // Check for common model file extensions
            if ["gguf", "ggml", "bin", "safetensors", "pkl", "pt", "pth"].contains(pathExtension) {
                modelFiles.append(fileURL)
            }
        }
        
        return modelFiles
    }
    
    static func getModelFileSize(_ url: URL) -> Double {
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            if let fileSize = attributes[.size] as? Int64 {
                return Double(fileSize) / (1024 * 1024 * 1024) // Convert to GB
            }
        } catch {
            print("Error getting file size for \(url.path): \(error)")
        }
        
        return 0.0
    }
}
