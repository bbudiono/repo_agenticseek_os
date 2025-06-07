//
// CacheModels.swift
// AgenticSeek Local Model Cache Management
//
// GREEN PHASE: Data models for CacheModels
// Comprehensive data models for cache management system
// Created: 2025-06-07 15:16:17
//

import Foundation
import CoreData
import SwiftUI

// MARK: - CacheModels Main Models

// MARK: - Cache Operation Result
enum CacheOperationResult: String, Codable {
    case success = "success"
    case failure = "failure"
    case inProgress = "in_progress"
    case cancelled = "cancelled"
    
    var displayName: String {
        switch self {
        case .success: return "Success"
        case .failure: return "Failure"
        case .inProgress: return "In Progress"
        case .cancelled: return "Cancelled"
        }
    }
}

// MARK: - Cache Entry Model
struct CacheEntry: Identifiable, Codable {
    let id = UUID()
    let modelId: String
    let dataType: CacheDataType
    let createdAt: Date
    let lastAccessedAt: Date
    let expiresAt: Date?
    let sizeInBytes: Int64
    let compressionRatio: Double
    let accessCount: Int
    let metadata: CacheMetadata
    
    // GREEN PHASE: Basic cache entry implementation
    init(modelId: String, dataType: CacheDataType) {
        self.modelId = modelId
        self.dataType = dataType
        self.createdAt = Date()
        self.lastAccessedAt = Date()
        self.expiresAt = nil
        self.sizeInBytes = 0
        self.compressionRatio = 1.0
        self.accessCount = 0
        self.metadata = CacheMetadata()
    }
}

// MARK: - Cache Data Types
enum CacheDataType: String, Codable, CaseIterable {
    case modelWeights = "model_weights"
    case activations = "activations"
    case computationResults = "computation_results"
    case sharedParameters = "shared_parameters"
    case compressedData = "compressed_data"
    
    var displayName: String {
        switch self {
        case .modelWeights: return "Model Weights"
        case .activations: return "Intermediate Activations"
        case .computationResults: return "Computation Results"
        case .sharedParameters: return "Shared Parameters"
        case .compressedData: return "Compressed Data"
        }
    }
    
    var icon: String {
        switch self {
        case .modelWeights: return "cube.box.fill"
        case .activations: return "brain"
        case .computationResults: return "function"
        case .sharedParameters: return "link"
        case .compressedData: return "archivebox.fill"
        }
    }
}

// MARK: - Cache Metadata
struct CacheMetadata: Codable {
    var modelName: String
    var modelVersion: String
    var sourceProvider: String
    var compressionAlgorithm: String
    var qualityScore: Double
    var performanceMetrics: PerformanceMetrics
    var tags: [String]
    var customProperties: [String: String]
    
    // GREEN PHASE: Default initialization
    init() {
        self.modelName = ""
        self.modelVersion = "1.0.0"
        self.sourceProvider = "local"
        self.compressionAlgorithm = "none"
        self.qualityScore = 0.0
        self.performanceMetrics = PerformanceMetrics()
        self.tags = []
        self.customProperties = [:]
    }
}

// MARK: - Performance Metrics
struct PerformanceMetrics: Codable {
    var inferenceTime: TimeInterval
    var memoryUsage: Int64
    var cacheHitRate: Double
    var compressionRatio: Double
    var accessFrequency: Double
    
    // GREEN PHASE: Default values
    init() {
        self.inferenceTime = 0.0
        self.memoryUsage = 0
        self.cacheHitRate = 0.0
        self.compressionRatio = 1.0
        self.accessFrequency = 0.0
    }
}

// MARK: - Cache Configuration
struct CacheConfiguration: Codable {
    var maxStorageSize: Int64
    var evictionStrategy: EvictionStrategy
    var compressionEnabled: Bool
    var encryptionEnabled: Bool
    var warmingStrategy: WarmingStrategy
    var retentionPolicy: RetentionPolicy
    
    // GREEN PHASE: Default configuration
    init() {
        self.maxStorageSize = 10 * 1024 * 1024 * 1024 // 10GB
        self.evictionStrategy = .lru
        self.compressionEnabled = true
        self.encryptionEnabled = false
        self.warmingStrategy = .predictive
        self.retentionPolicy = .timeBasedExpiry(days: 30)
    }
}

// MARK: - Eviction Strategy
enum EvictionStrategy: String, Codable, CaseIterable {
    case lru = "least_recently_used"
    case lfu = "least_frequently_used"
    case fifo = "first_in_first_out"
    case predictive = "predictive_algorithm"
    case hybrid = "hybrid_strategy"
    
    var displayName: String {
        switch self {
        case .lru: return "Least Recently Used"
        case .lfu: return "Least Frequently Used"
        case .fifo: return "First In, First Out"
        case .predictive: return "Predictive Algorithm"
        case .hybrid: return "Hybrid Strategy"
        }
    }
}

// MARK: - Warming Strategy
enum WarmingStrategy: String, Codable, CaseIterable {
    case none = "no_warming"
    case predictive = "predictive_warming"
    case scheduled = "scheduled_warming"
    case adaptive = "adaptive_warming"
    
    var displayName: String {
        switch self {
        case .none: return "No Warming"
        case .predictive: return "Predictive Warming"
        case .scheduled: return "Scheduled Warming"
        case .adaptive: return "Adaptive Warming"
        }
    }
}

// MARK: - Retention Policy
enum RetentionPolicy: Codable {
    case never
    case timeBasedExpiry(days: Int)
    case accessBasedExpiry(accessCount: Int)
    case sizeBasedExpiry(maxSizeGB: Int)
    
    var displayName: String {
        switch self {
        case .never:
            return "Never Expire"
        case .timeBasedExpiry(let days):
            return "Expire after \(days) days"
        case .accessBasedExpiry(let count):
            return "Expire after \(count) accesses"
        case .sizeBasedExpiry(let size):
            return "Expire when size exceeds \(size)GB"
        }
    }
}

// MARK: - Cache Statistics
struct CacheStatistics: Codable {
    var totalEntries: Int
    var totalStorageUsed: Int64
    var hitRate: Double
    var missRate: Double
    var evictionRate: Double
    var compressionEfficiency: Double
    var averageAccessTime: TimeInterval
    var lastUpdated: Date
    
    // GREEN PHASE: Default statistics
    init() {
        self.totalEntries = 0
        self.totalStorageUsed = 0
        self.hitRate = 0.0
        self.missRate = 0.0
        self.evictionRate = 0.0
        self.compressionEfficiency = 0.0
        self.averageAccessTime = 0.0
        self.lastUpdated = Date()
    }
}

// MARK: - Cache Query
struct CacheQuery {
    var modelId: String?
    var dataType: CacheDataType?
    var dateRange: ClosedRange<Date>?
    var tags: [String]
    var minimumQualityScore: Double
    var sortBy: CacheSortOption
    var limit: Int
    
    // GREEN PHASE: Default query
    init() {
        self.modelId = nil
        self.dataType = nil
        self.dateRange = nil
        self.tags = []
        self.minimumQualityScore = 0.0
        self.sortBy = .lastAccessed
        self.limit = 100
    }
}

// MARK: - Cache Sort Options
enum CacheSortOption: String, CaseIterable {
    case createdAt = "created_at"
    case lastAccessed = "last_accessed"
    case accessCount = "access_count"
    case size = "size"
    case qualityScore = "quality_score"
    
    var displayName: String {
        switch self {
        case .createdAt: return "Created Date"
        case .lastAccessed: return "Last Accessed"
        case .accessCount: return "Access Count"
        case .size: return "Size"
        case .qualityScore: return "Quality Score"
        }
    }
}

// GREEN PHASE: Extensions for additional functionality
extension CacheEntry {
    var isExpired: Bool {
        guard let expiresAt = expiresAt else { return false }
        return Date() > expiresAt
    }
    
    var formattedSize: String {
        return ByteCountFormatter.string(fromByteCount: sizeInBytes, countStyle: .file)
    }
    
    var ageInDays: Int {
        return Calendar.current.dateComponents([.day], from: createdAt, to: Date()).day ?? 0
    }
}

extension CacheStatistics {
    var formattedTotalStorage: String {
        return ByteCountFormatter.string(fromByteCount: totalStorageUsed, countStyle: .file)
    }
    
    var efficiencyScore: Double {
        return (hitRate * 0.4) + (compressionEfficiency * 0.3) + ((1.0 - evictionRate) * 0.3)
    }
}


// MARK: - REFACTOR PHASE: Supporting Enums and Structs

enum CacheError: Error {
    case memoryPressure
    case storageExhausted
    case corruptedData
    case networkUnavailable
}

enum ErrorRecoveryAction {
    case retry
    case fallback
    case abort
}

// MARK: - Cache Type Enum
enum CacheType: String, Codable, CaseIterable {
    case modelWeights = "model_weights"
    case activations = "activations" 
    case computationResults = "computation_results"
    case sharedParameters = "shared_parameters"
    case compressedData = "compressed_data"
    
    var displayName: String {
        switch self {
        case .modelWeights: return "Model Weights"
        case .activations: return "Intermediate Activations"
        case .computationResults: return "Computation Results"
        case .sharedParameters: return "Shared Parameters"
        case .compressedData: return "Compressed Data"
        }
    }
}

// MARK: - Shared Cache Structs (Canonical Definitions)

struct CacheMetrics {
    var hitCount: Int = 0
    var missCount: Int = 0
    var evictionCount: Int = 0
    var storageUsed: Int64 = 0
    var initializationTime: Date = Date()
    
    mutating func updateInitializationTime() {
        initializationTime = Date()
    }
    
    mutating func reset() {
        hitCount = 0
        missCount = 0
        evictionCount = 0
        storageUsed = 0
    }
}

struct PerformanceStatistics {
    var operationCount: Int = 0
    var averageResponseTime: TimeInterval = 0.0
    var optimizationCount: Int = 0
    
    mutating func incrementOperationCount() {
        operationCount += 1
    }
    
    mutating func recordOptimization() {
        optimizationCount += 1
    }
}
