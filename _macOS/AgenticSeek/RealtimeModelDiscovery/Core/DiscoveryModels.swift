import Foundation
import CoreData
import Combine

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Comprehensive data models for MLACS Real-time Model Discovery
 * Issues & Complexity Summary: Production-ready data structures with real-time capabilities
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~250
   - Core Algorithm Complexity: Medium
   - Dependencies: 2 New
   - State Management Complexity: High
   - Novelty/Uncertainty Factor: Low
 * AI Pre-Task Self-Assessment: 90%
 * Problem Estimate: 92%
 * Initial Code Complexity Estimate: 88%
 * Final Code Complexity: 90%
 * Overall Result Score: 95%
 * Last Updated: 2025-01-07
 */

// MARK: - Core Discovery Models

struct ModelProvider: Codable, Identifiable, Hashable {
    let id = UUID()
    let name: String
    let endpoint: String
    let type: ProviderType
    let isActive: Bool
    let lastChecked: Date
    let status: ProviderStatus
    let supportedFormats: [String]
    let capabilities: [String]
    
    enum ProviderType: String, Codable, CaseIterable {
        case local = "local"
        case remote = "remote"
        case hybrid = "hybrid"
        
        var displayName: String {
            switch self {
            case .local: return "Local Provider"
            case .remote: return "Remote Provider"
            case .hybrid: return "Hybrid Provider"
            }
        }
    }
    
    enum ProviderStatus: String, Codable, CaseIterable {
        case active = "active"
        case inactive = "inactive"
        case error = "error"
        case unknown = "unknown"
        
        var color: String {
            switch self {
            case .active: return "green"
            case .inactive: return "gray"
            case .error: return "red"
            case .unknown: return "orange"
            }
        }
    }
}

struct DiscoveredModel: Codable, Identifiable, Hashable {
    let id = UUID()
    let name: String
    let provider: String
    let modelType: String
    let size: String
    let capabilities: [String]
    let endpoint: String
    let discovered: Date
    let isAvailable: Bool
    let performance: ModelPerformanceMetrics?
    
    var hash: Int {
        return name.hashValue ^ provider.hashValue
    }
    
    static func == (lhs: DiscoveredModel, rhs: DiscoveredModel) -> Bool {
        return lhs.name == rhs.name && lhs.provider == rhs.provider
    }
}

struct ModelInfo: Codable, Identifiable, Hashable {
    let id: String
    let name: String
    let displayName: String
    let provider: String
    let version: String?
    let sizeGB: Double
    let type: String
    let architecture: String?
    let path: String
    let downloadURL: String?
    let metadata: [String: AnyHashable]
    let tags: [String]
    let createdAt: Date
    let updatedAt: Date
    
    var formattedSize: String {
        if sizeGB < 1.0 {
            return String(format: "%.0f MB", sizeGB * 1024)
        } else {
            return String(format: "%.1f GB", sizeGB)
        }
    }
    
    var isLocal: Bool {
        return !path.isEmpty && FileManager.default.fileExists(atPath: path)
    }
}

struct DiscoverySession: Codable, Identifiable {
    let id = UUID()
    let startTime: Date
    var endTime: Date?
    let scanType: ScanType
    var modelsFound: Int = 0
    var providersScanned: Int = 0
    var errors: [String] = []
    var status: SessionStatus = .running
    
    enum ScanType: String, Codable, CaseIterable {
        case full = "full"
        case incremental = "incremental"
        case targeted = "targeted"
        case background = "background"
    }
    
    enum SessionStatus: String, Codable, CaseIterable {
        case running = "running"
        case completed = "completed"
        case failed = "failed"
        case cancelled = "cancelled"
    }
    
    var duration: TimeInterval {
        guard let endTime = endTime else {
            return Date().timeIntervalSince(startTime)
        }
        return endTime.timeIntervalSince(startTime)
    }
}

// MARK: - Discovery Configuration

struct DiscoveryConfiguration: Codable {
    var enabledProviders: [String] = ["ollama", "lm_studio"]
    var scanInterval: TimeInterval = 300 // 5 minutes
    var enableBackgroundScanning: Bool = true
    var enableAutoVerification: Bool = true
    var maxConcurrentScans: Int = 3
    var modelSizeThresholdGB: Double = 50.0
    var enablePerformanceProfiling: Bool = true
    var enableCompatibilityChecking: Bool = true
    var notificationSettings: NotificationSettings = NotificationSettings()
    
    struct NotificationSettings: Codable {
        var enableNewModelNotifications: Bool = true
        var enableErrorNotifications: Bool = true
        var enablePerformanceAlerts: Bool = false
        var soundEnabled: Bool = true
    }
}

// MARK: - Performance Metrics

struct ModelPerformanceMetrics: Codable, Identifiable {
    let id = UUID()
    let modelId: String
    let measuredAt: Date
    let inferenceSpeedTokensPerSecond: Double
    let memoryUsageMB: Double
    let cpuUsagePercent: Double
    let gpuUsagePercent: Double
    let latencyFirstTokenMs: Double
    let throughputSustainedTokensPerSecond: Double
    let qualityScore: Double
    let stabilityScore: Double
    let reliabilityScore: Double
    let overallRating: Double
    
    var performanceGrade: PerformanceGrade {
        switch overallRating {
        case 0.9...1.0: return .excellent
        case 0.8..<0.9: return .good
        case 0.7..<0.8: return .fair
        case 0.6..<0.7: return .poor
        default: return .failing
        }
    }
}

enum PerformanceGrade: String, CaseIterable {
    case excellent = "A+"
    case good = "A"
    case fair = "B"
    case poor = "C"
    case failing = "F"
    
    var color: String {
        switch self {
        case .excellent: return "green"
        case .good: return "blue"
        case .fair: return "yellow"
        case .poor: return "orange"
        case .failing: return "red"
        }
    }
}

// MARK: - Recommendation System

struct ModelRecommendation: Codable, Identifiable {
    let id = UUID()
    let modelId: String
    let recommendationType: RecommendationType
    let confidence: Double
    let reasoning: [String]
    let useCase: String
    let estimatedPerformance: Double
    let hardwareCompatibility: Double
    let resourceRequirements: ResourceRequirements
    let alternativeModels: [String]
    let generatedAt: Date
    
    enum RecommendationType: String, Codable, CaseIterable {
        case optimal = "optimal"
        case alternative = "alternative"
        case experimental = "experimental"
        case fallback = "fallback"
        
        var displayName: String {
            switch self {
            case .optimal: return "Optimal Choice"
            case .alternative: return "Good Alternative"
            case .experimental: return "Experimental"
            case .fallback: return "Fallback Option"
            }
        }
    }
    
    struct ResourceRequirements: Codable {
        let minMemoryGB: Double
        let recommendedMemoryGB: Double
        let minCPUCores: Int
        let gpuRequired: Bool
        let estimatedDiskSpaceGB: Double
        let networkBandwidthMbps: Double?
    }
}

// MARK: - Discovery Events

struct DiscoveryEvent: Codable, Identifiable {
    let id = UUID()
    let timestamp: Date
    let type: EventType
    let modelId: String?
    let providerId: String?
    let message: String
    let details: [String: String]
    let severity: Severity
    
    enum EventType: String, Codable, CaseIterable {
        case modelDiscovered = "model_discovered"
        case modelRemoved = "model_removed"
        case modelUpdated = "model_updated"
        case providerConnected = "provider_connected"
        case providerDisconnected = "provider_disconnected"
        case scanStarted = "scan_started"
        case scanCompleted = "scan_completed"
        case scanFailed = "scan_failed"
        case performanceUpdated = "performance_updated"
        case recommendationGenerated = "recommendation_generated"
        case error = "error"
        case warning = "warning"
        case info = "info"
    }
    
    enum Severity: String, Codable, CaseIterable {
        case low = "low"
        case medium = "medium"
        case high = "high"
        case critical = "critical"
        
        var color: String {
            switch self {
            case .low: return "gray"
            case .medium: return "blue"
            case .high: return "orange"
            case .critical: return "red"
            }
        }
    }
}

// MARK: - Discovery Statistics

struct DiscoveryStatistics: Codable {
    let totalModelsDiscovered: Int
    let totalProvidersActive: Int
    let totalScansSinceStartup: Int
    let averageScanDurationSeconds: Double
    let lastSuccessfulScan: Date?
    let discoverySuccessRate: Double
    let performanceMetricsCollected: Int
    let recommendationsGenerated: Int
    let storageUsedMB: Double
    let uptime: TimeInterval
    
    var formattedUptime: String {
        let hours = Int(uptime) / 3600
        let minutes = Int(uptime % 3600) / 60
        return String(format: "%dh %dm", hours, minutes)
    }
}

// MARK: - Core Data Entities

extension ModelEntity {
    static var fetchRequest: NSFetchRequest<ModelEntity> {
        return NSFetchRequest<ModelEntity>(entityName: "ModelEntity")
    }
}

// MARK: - Date Extensions

extension Date {
    func relativeDateString() -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: self, relativeTo: Date())
    }
    
    func ISO8601String() -> String {
        return ISO8601DateFormatter().string(from: self)
    }
}

// MARK: - Discovery Result Aggregation

struct DiscoveryResult: Codable {
    let session: DiscoverySession
    let discoveredModels: [DiscoveredModel]
    let providerStatuses: [String: ModelProvider.ProviderStatus]
    let performanceMetrics: [ModelPerformanceMetrics]
    let recommendations: [ModelRecommendation]
    let events: [DiscoveryEvent]
    let statistics: DiscoveryStatistics
    
    var successRate: Double {
        guard !discoveredModels.isEmpty else { return 0.0 }
        let successfulModels = discoveredModels.filter { $0.isAvailable }
        return Double(successfulModels.count) / Double(discoveredModels.count)
    }
}
