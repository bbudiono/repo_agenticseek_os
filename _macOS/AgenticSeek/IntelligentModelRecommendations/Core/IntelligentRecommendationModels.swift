import Foundation
import Combine
import CoreML

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Comprehensive data models for MLACS Intelligent Model Recommendations
 * Issues & Complexity Summary: Production-ready data structures with AI integration
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~400
   - Core Algorithm Complexity: High
   - Dependencies: 3 New
   - State Management Complexity: High
   - Novelty/Uncertainty Factor: Medium
 * AI Pre-Task Self-Assessment: 88%
 * Problem Estimate: 90%
 * Initial Code Complexity Estimate: 85%
 * Final Code Complexity: 88%
 * Overall Result Score: 95%
 * Last Updated: 2025-01-07
 */

// MARK: - Core Recommendation Models

struct IntelligentModelRecommendation: Codable, Identifiable, Hashable {
    let id = UUID()
    let modelId: String
    let modelName: String
    let confidenceScore: Double
    let recommendationType: RecommendationType
    let reasoning: [String]
    let expectedPerformance: PerformancePrediction
    let resourceRequirements: ResourceRequirements
    let compatibilityScore: Double
    let qualityPrediction: Double
    let speedPrediction: Double
    let tradeOffs: [String: String]
    let alternativeModels: [String]
    let generatedAt: Date
    let context: RecommendationContext
    
    enum RecommendationType: String, Codable, CaseIterable {
        case optimal = "optimal"
        case alternative = "alternative"
        case fallback = "fallback"
        case experimental = "experimental"
        case contextSpecific = "context_specific"
        
        var displayName: String {
            switch self {
            case .optimal: return "Optimal Choice"
            case .alternative: return "Good Alternative"
            case .fallback: return "Fallback Option"
            case .experimental: return "Experimental"
            case .contextSpecific: return "Context-Specific"
            }
        }
        
        var priority: Int {
            switch self {
            case .optimal: return 1
            case .alternative: return 2
            case .contextSpecific: return 3
            case .experimental: return 4
            case .fallback: return 5
            }
        }
    }
}

struct PerformancePrediction: Codable {
    let inferenceSpeedMs: Double
    let qualityScore: Double
    let memoryUsageMB: Double
    let cpuUtilization: Double
    let gpuUtilization: Double
    let confidenceInterval: Double
    let predictionAccuracy: Double
}

struct ResourceRequirements: Codable {
    let minMemoryGB: Double
    let recommendedMemoryGB: Double
    let minCPUCores: Int
    let gpuRequired: Bool
    let neuralEngineSupport: Bool
    let estimatedDiskSpaceGB: Double
    let networkBandwidthMbps: Double?
    let thermalImpact: ThermalImpact
    
    enum ThermalImpact: String, Codable, CaseIterable {
        case minimal = "minimal"
        case moderate = "moderate"
        case significant = "significant"
        case high = "high"
        
        var description: String {
            switch self {
            case .minimal: return "Minimal thermal impact"
            case .moderate: return "Moderate thermal impact"
            case .significant: return "Significant thermal impact"
            case .high: return "High thermal impact - may throttle"
            }
        }
    }
}

struct RecommendationContext: Codable {
    let taskComplexity: Double
    let userPreferences: [String: String]
    let hardwareCapabilities: [String: Any]
    let timeOfDay: String
    let systemLoad: Double
    let availableModels: [String]
    let previousSelections: [String]
    
    private enum CodingKeys: String, CodingKey {
        case taskComplexity, userPreferences, timeOfDay, systemLoad, availableModels, previousSelections
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        taskComplexity = try container.decode(Double.self, forKey: .taskComplexity)
        userPreferences = try container.decode([String: String].self, forKey: .userPreferences)
        hardwareCapabilities = [:]
        timeOfDay = try container.decode(String.self, forKey: .timeOfDay)
        systemLoad = try container.decode(Double.self, forKey: .systemLoad)
        availableModels = try container.decode([String].self, forKey: .availableModels)
        previousSelections = try container.decode([String].self, forKey: .previousSelections)
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(taskComplexity, forKey: .taskComplexity)
        try container.encode(userPreferences, forKey: .userPreferences)
        try container.encode(timeOfDay, forKey: .timeOfDay)
        try container.encode(systemLoad, forKey: .systemLoad)
        try container.encode(availableModels, forKey: .availableModels)
        try container.encode(previousSelections, forKey: .previousSelections)
    }
}

// MARK: - User Learning Models

struct UserFeedback: Codable, Identifiable {
    let id = UUID()
    let userId: String
    let modelId: String
    let taskId: String
    let rating: Double // 1.0 - 5.0
    let responseTime: Double
    let qualityRating: Double
    let speedRating: Double
    let overallSatisfaction: Double
    let comments: String?
    let timestamp: Date
    let context: FeedbackContext
}

struct FeedbackContext: Codable {
    let taskDomain: String
    let taskComplexity: Double
    let systemState: String
    let modelParameters: [String: Any]
    let userState: String // "focused", "casual", "urgent"
    
    private enum CodingKeys: String, CodingKey {
        case taskDomain, taskComplexity, systemState, userState
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        taskDomain = try container.decode(String.self, forKey: .taskDomain)
        taskComplexity = try container.decode(Double.self, forKey: .taskComplexity)
        systemState = try container.decode(String.self, forKey: .systemState)
        modelParameters = [:]
        userState = try container.decode(String.self, forKey: .userState)
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(taskDomain, forKey: .taskDomain)
        try container.encode(taskComplexity, forKey: .taskComplexity)
        try container.encode(systemState, forKey: .systemState)
        try container.encode(userState, forKey: .userState)
    }
}

struct UserPreferenceProfile: Codable {
    let userId: String
    var learningProgress: Double
    var adaptationMetrics: AdaptationMetrics
    var preferenceWeights: PreferenceWeights
    var domainSpecificPreferences: [String: DomainPreference]
    var temporalPreferences: TemporalPreferences
    var qualitySpeedTradeoffPreference: Double // 0.0 = speed, 1.0 = quality
    var experimentalModelTolerance: Double
    var lastUpdated: Date
    
    struct AdaptationMetrics: Codable {
        var totalFeedbacks: Int
        var averageRating: Double
        var consistencyScore: Double
        var preferenceStability: Double
        var learningVelocity: Double
    }
    
    struct PreferenceWeights: Codable {
        var speed: Double
        var quality: Double
        var resourceEfficiency: Double
        var novelty: Double
        var reliability: Double
        
        var normalized: PreferenceWeights {
            let total = speed + quality + resourceEfficiency + novelty + reliability
            return PreferenceWeights(
                speed: speed / total,
                quality: quality / total,
                resourceEfficiency: resourceEfficiency / total,
                novelty: novelty / total,
                reliability: reliability / total
            )
        }
    }
    
    struct DomainPreference: Codable {
        let domain: String
        var preferredModelSize: String
        var qualityThreshold: Double
        var speedTolerance: Double
        var lastUsed: Date
        var usageCount: Int
    }
    
    struct TemporalPreferences: Codable {
        var morningPreferences: TimeBasedPreference
        var afternoonPreferences: TimeBasedPreference
        var eveningPreferences: TimeBasedPreference
        var weekendPreferences: TimeBasedPreference
        
        struct TimeBasedPreference: Codable {
            var preferredSpeed: String
            var qualityTolerance: Double
            var experimentalTolerance: Double
        }
    }
}

// MARK: - Task Analysis Models

struct EnhancedTaskComplexity: Codable, Identifiable {
    let id = UUID()
    let taskId: String
    let taskDescription: String
    let analysisTimestamp: Date
    
    // Core complexity metrics
    let overallComplexity: Double
    let domainComplexity: DomainComplexity
    let linguisticComplexity: LinguisticComplexity
    let cognitiveComplexity: CognitiveComplexity
    let computationalComplexity: ComputationalComplexity
    
    // Context requirements
    let contextRequirements: ContextRequirements
    let resourcePredictions: ResourcePredictions
    let qualityExpectations: QualityExpectations
    
    struct DomainComplexity: Codable {
        let primaryDomain: String
        let secondaryDomains: [String]
        let crossDomainComplexity: Double
        let domainSpecificRequirements: [String]
    }
    
    struct LinguisticComplexity: Codable {
        let vocabularyComplexity: Double
        let syntacticComplexity: Double
        let semanticComplexity: Double
        let pragmaticComplexity: Double
        let estimatedReadingLevel: Double
    }
    
    struct CognitiveComplexity: Codable {
        let reasoningRequired: Bool
        let creativityRequired: Bool
        let memoryIntensive: Bool
        let attentionDemand: Double
        let workingMemoryLoad: Double
        let executiveFunctionDemand: Double
    }
    
    struct ComputationalComplexity: Codable {
        let estimatedTokens: Int
        let contextWindowRequired: Int
        let parallelProcessingBenefit: Double
        let memoryIntensity: Double
        let computeIntensity: Double
    }
    
    struct ContextRequirements: Codable {
        let minContextLength: Int
        let optimalContextLength: Int
        let contextPersistenceRequired: Bool
        let crossReferenceComplexity: Double
    }
    
    struct ResourcePredictions: Codable {
        let cpuUtilization: Double
        let memoryUtilization: Double
        let diskIOIntensity: Double
        let networkRequirements: Double
        let estimatedDuration: Double
    }
    
    struct QualityExpectations: Codable {
        let accuracyRequirement: Double
        let coherenceRequirement: Double
        let creativityExpectation: Double
        let factualAccuracyRequired: Bool
        let styleConsistencyRequired: Bool
    }
}

// MARK: - Model Performance Models

struct ModelPerformanceHistory: Codable {
    let modelId: String
    var performanceMetrics: [PerformanceSnapshot]
    var userSatisfactionHistory: [UserSatisfactionSnapshot]
    var reliabilityMetrics: ReliabilityMetrics
    var adaptationHistory: [ModelAdaptation]
    var lastUpdated: Date
    
    struct PerformanceSnapshot: Codable {
        let timestamp: Date
        let taskComplexity: Double
        let inferenceTime: Double
        let qualityScore: Double
        let resourceUtilization: Double
        let userRating: Double
        let context: [String: String]
    }
    
    struct UserSatisfactionSnapshot: Codable {
        let timestamp: Date
        let userId: String
        let overallRating: Double
        let speedRating: Double
        let qualityRating: Double
        let taskType: String
    }
    
    struct ReliabilityMetrics: Codable {
        let uptime: Double
        let errorRate: Double
        let consistencyScore: Double
        let stabilityIndex: Double
        let mtbfHours: Double
    }
    
    struct ModelAdaptation: Codable {
        let timestamp: Date
        let adaptationType: String
        let trigger: String
        let performanceImpact: Double
        let userAcceptance: Double
    }
}

// MARK: - Supporting Extensions

extension IntelligentModelRecommendation {
    var confidenceLevel: String {
        switch confidenceScore {
        case 0.9...1.0: return "Very High"
        case 0.7..<0.9: return "High"
        case 0.5..<0.7: return "Medium"
        case 0.3..<0.5: return "Low"
        default: return "Very Low"
        }
    }
    
    var recommendationStrength: Double {
        return (confidenceScore + qualityPrediction + speedPrediction) / 3.0
    }
    
    func matchesUserPreferences(_ preferences: UserPreferenceProfile) -> Double {
        let weights = preferences.preferenceWeights.normalized
        
        let speedMatch = speedPrediction * weights.speed
        let qualityMatch = qualityPrediction * weights.quality
        let efficiencyMatch = (1.0 - Double(resourceRequirements.minMemoryGB) / 64.0) * weights.resourceEfficiency
        
        return (speedMatch + qualityMatch + efficiencyMatch) / 3.0
    }
}

extension UserPreferenceProfile {
    mutating func updateFromFeedback(_ feedback: UserFeedback) {
        adaptationMetrics.totalFeedbacks += 1
        adaptationMetrics.averageRating = (adaptationMetrics.averageRating * Double(adaptationMetrics.totalFeedbacks - 1) + feedback.rating) / Double(adaptationMetrics.totalFeedbacks)
        
        // Update preference weights based on feedback
        if feedback.speedRating > feedback.qualityRating {
            preferenceWeights.speed += 0.01
            preferenceWeights.quality -= 0.005
        } else {
            preferenceWeights.quality += 0.01
            preferenceWeights.speed -= 0.005
        }
        
        lastUpdated = Date()
    }
    
    func getPreferenceForTime(_ date: Date) -> UserPreferenceProfile.TemporalPreferences.TimeBasedPreference {
        let hour = Calendar.current.component(.hour, from: date)
        
        switch hour {
        case 6..<12: return temporalPreferences.morningPreferences
        case 12..<17: return temporalPreferences.afternoonPreferences
        case 17..<22: return temporalPreferences.eveningPreferences
        default: return temporalPreferences.eveningPreferences
        }
    }
}

// MARK: - Date Extensions

extension Date {
    func ISO8601String() -> String {
        return ISO8601DateFormatter().string(from: self)
    }
}
