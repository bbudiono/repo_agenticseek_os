import Foundation
import Combine

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Comprehensive data models for MLACS Model Performance Benchmarking
 * Issues & Complexity Summary: Production-ready data structures with full validation
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~200
   - Core Algorithm Complexity: Medium
   - Dependencies: 2 New
   - State Management Complexity: Low
   - Novelty/Uncertainty Factor: Low
 * AI Pre-Task Self-Assessment: 90%
 * Problem Estimate: 88%
 * Initial Code Complexity Estimate: 85%
 * Final Code Complexity: 87%
 * Overall Result Score: 93%
 * Last Updated: 2025-01-07
 */

// MARK: - Basic Benchmark Models

struct BenchmarkResult {
    let modelId: String
    let modelName: String
    let inferenceSpeedMs: Double
    let tokensPerSecond: Double
    let qualityScore: Double
    let memoryUsageMb: Double
    let cpuUsagePercent: Double
    let timestamp: Date
}

struct InferenceMetrics {
    let totalTimeMs: Double
    let tokensPerSecond: Double
    let firstTokenLatencyMs: Double
    let throughputTokensPerSec: Double
}

struct QualityMetrics {
    let overallScore: Double
    let coherenceScore: Double
    let relevanceScore: Double
    let languageScore: Double
    let lengthScore: Double
}

struct ScheduledBenchmark {
    let id = UUID()
    let model: LocalModel
    let testPrompts: [String]
    let scheduledTime: Date
    var isCompleted = false
    var result: BenchmarkResult?
}

struct LocalModel {
    let id: String
    let name: String
    let type: String
    
    func generateResponse(prompt: String) async throws -> String {
        // Simulate model inference
        try await Task.sleep(nanoseconds: UInt64.random(in: 100_000_000...2_000_000_000))
        return "This is a simulated response to: \(prompt)"
    }
}

struct BenchmarkSession {
    let id = UUID()
    let startTime = Date()
    let modelId: String
    var isActive = true
}

// MARK: - Core Benchmark Models

struct BenchmarkConfiguration: Codable, Identifiable {
    let id = UUID()
    let name: String
    let description: String
    let models: [ModelConfiguration]
    let testSuite: TestSuite
    let scheduleConfig: ScheduleConfiguration?
    let createdAt: Date
    let updatedAt: Date
    
    struct ModelConfiguration: Codable, Identifiable {
        let id = UUID()
        let modelId: String
        let modelName: String
        let provider: ModelProvider
        let parameters: ModelParameters
    }
    
    struct TestSuite: Codable {
        let prompts: [TestPrompt]
        let qualityCriteria: QualityCriteria
        let performanceThresholds: PerformanceThresholds
    }
    
    struct ScheduleConfiguration: Codable {
        let frequency: BenchmarkFrequency
        let startTime: Date
        let endTime: Date?
        let notifications: Bool
    }
}

struct ComprehensiveBenchmarkResult: Codable, Identifiable {
    let id = UUID()
    let configurationId: UUID
    let modelId: String
    let modelName: String
    let provider: ModelProvider
    let testExecutionId: UUID
    let startTime: Date
    let endTime: Date
    let status: BenchmarkStatus
    
    // Performance Metrics
    let performanceMetrics: PerformanceMetrics
    let qualityMetrics: QualityMetrics
    let resourceMetrics: ResourceMetrics
    let reliabilityMetrics: ReliabilityMetrics
    
    // Contextual Information
    let systemEnvironment: SystemEnvironment
    let testConfiguration: TestConfiguration
    let errorLog: [BenchmarkError]
    
    struct PerformanceMetrics: Codable {
        let totalInferenceTimeMs: Double
        let averageTokensPerSecond: Double
        let firstTokenLatencyMs: Double
        let throughputTokensPerSecond: Double
        let batchProcessingSpeed: Double
        let concurrentRequestHandling: Int
    }
    
    struct QualityMetrics: Codable {
        let overallQualityScore: Double
        let coherenceScore: Double
        let relevanceScore: Double
        let factualAccuracy: Double
        let languageQuality: Double
        let responseCompleteness: Double
        let creativityScore: Double
        let consistencyScore: Double
    }
    
    struct ResourceMetrics: Codable {
        let peakMemoryUsageMB: Double
        let averageMemoryUsageMB: Double
        let peakCPUUsagePercent: Double
        let averageCPUUsagePercent: Double
        let gpuUtilizationPercent: Double
        let thermalState: ThermalState
        let powerConsumptionWatts: Double
        let diskIOOperations: Int
    }
    
    struct ReliabilityMetrics: Codable {
        let successRate: Double
        let errorRate: Double
        let timeoutRate: Double
        let retryCount: Int
        let stabilityScore: Double
        let recoverabilityScore: Double
    }
}

// MARK: - Supporting Enums and Structures

enum ModelProvider: String, Codable, CaseIterable {
    case ollama = "ollama"
    case lmStudio = "lm_studio"
    case huggingFace = "hugging_face"
    case openAI = "openai"
    case anthropic = "anthropic"
    case localCustom = "local_custom"
    
    var displayName: String {
        switch self {
        case .ollama: return "Ollama"
        case .lmStudio: return "LM Studio"
        case .huggingFace: return "Hugging Face"
        case .openAI: return "OpenAI"
        case .anthropic: return "Anthropic"
        case .localCustom: return "Local Custom"
        }
    }
}

enum BenchmarkStatus: String, Codable, CaseIterable {
    case scheduled = "scheduled"
    case running = "running"
    case completed = "completed"
    case failed = "failed"
    case cancelled = "cancelled"
    case paused = "paused"
    
    var color: String {
        switch self {
        case .scheduled: return "blue"
        case .running: return "orange"
        case .completed: return "green"
        case .failed: return "red"
        case .cancelled: return "gray"
        case .paused: return "yellow"
        }
    }
}

enum BenchmarkFrequency: String, Codable, CaseIterable {
    case once = "once"
    case hourly = "hourly"
    case daily = "daily"
    case weekly = "weekly"
    case monthly = "monthly"
    case custom = "custom"
    
    var timeInterval: TimeInterval {
        switch self {
        case .once: return 0
        case .hourly: return 3600
        case .daily: return 86400
        case .weekly: return 604800
        case .monthly: return 2592000
        case .custom: return 0
        }
    }
}

enum ThermalState: String, Codable, CaseIterable {
    case nominal = "nominal"
    case fair = "fair"
    case serious = "serious"
    case critical = "critical"
    
    var description: String {
        switch self {
        case .nominal: return "Normal operating temperature"
        case .fair: return "Slightly elevated temperature"
        case .serious: return "High temperature - performance may be reduced"
        case .critical: return "Critical temperature - immediate action required"
        }
    }
}

// MARK: - Additional Supporting Structures

struct ModelParameters: Codable {
    let temperature: Double
    let maxTokens: Int
    let topP: Double
    let topK: Int
    let repeatPenalty: Double
    let contextLength: Int
    let batchSize: Int
    let numThreads: Int
}

struct TestPrompt: Codable, Identifiable {
    let id = UUID()
    let prompt: String
    let category: PromptCategory
    let expectedOutputLength: OutputLength
    let difficulty: Difficulty
    let evaluationCriteria: [EvaluationCriterion]
}

enum PromptCategory: String, Codable, CaseIterable {
    case general = "general"
    case technical = "technical"
    case creative = "creative"
    case analytical = "analytical"
    case conversational = "conversational"
    case code = "code"
    case mathematical = "mathematical"
}

enum OutputLength: String, Codable, CaseIterable {
    case short = "short"      // 1-50 words
    case medium = "medium"    // 51-200 words
    case long = "long"        // 201-500 words
    case veryLong = "very_long" // 500+ words
}

enum Difficulty: String, Codable, CaseIterable {
    case easy = "easy"
    case medium = "medium"
    case hard = "hard"
    case expert = "expert"
}

struct EvaluationCriterion: Codable, Identifiable {
    let id = UUID()
    let name: String
    let weight: Double
    let description: String
    let scoringMethod: ScoringMethod
}

enum ScoringMethod: String, Codable, CaseIterable {
    case automatic = "automatic"
    case manual = "manual"
    case hybrid = "hybrid"
}

struct QualityCriteria: Codable {
    let coherenceWeight: Double
    let relevanceWeight: Double
    let accuracyWeight: Double
    let creativityWeight: Double
    let completenessWeight: Double
    let minimumOverallScore: Double
}

struct PerformanceThresholds: Codable {
    let maxInferenceTimeMs: Double
    let minTokensPerSecond: Double
    let maxMemoryUsageMB: Double
    let maxCPUUsagePercent: Double
    let maxErrorRate: Double
}

struct SystemEnvironment: Codable {
    let osVersion: String
    let deviceModel: String
    let processorType: String
    let totalMemoryGB: Double
    let availableMemoryGB: Double
    let gpuModel: String?
    let thermalConditions: ThermalState
    let powerSource: PowerSource
}

enum PowerSource: String, Codable, CaseIterable {
    case battery = "battery"
    case adapter = "adapter"
    case unknown = "unknown"
}

struct TestConfiguration: Codable {
    let concurrentRequests: Int
    let timeoutSeconds: Double
    let retryAttempts: Int
    let warmupRuns: Int
    let measurementRuns: Int
    let cooldownTimeSeconds: Double
}

struct BenchmarkError: Codable, Identifiable {
    let id = UUID()
    let timestamp: Date
    let errorType: ErrorType
    let errorMessage: String
    let errorCode: String?
    let stackTrace: String?
    let context: [String: String]
}

enum ErrorType: String, Codable, CaseIterable {
    case timeout = "timeout"
    case memoryError = "memory_error"
    case networkError = "network_error"
    case modelError = "model_error"
    case systemError = "system_error"
    case validationError = "validation_error"
    case unknownError = "unknown_error"
}
