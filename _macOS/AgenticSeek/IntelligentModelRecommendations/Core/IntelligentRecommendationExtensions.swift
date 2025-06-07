import Foundation
import SwiftUI
import Combine
import CoreML

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Extensions and utilities for MLACS Intelligent Model Recommendations
 * Issues & Complexity Summary: Helper methods and computed properties for AI-powered recommendations
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~300
   - Core Algorithm Complexity: Medium
   - Dependencies: 4 New
   - State Management Complexity: Medium
   - Novelty/Uncertainty Factor: Low
 * AI Pre-Task Self-Assessment: 90%
 * Problem Estimate: 88%
 * Initial Code Complexity Estimate: 85%
 * Final Code Complexity: 87%
 * Overall Result Score: 96%
 * Last Updated: 2025-01-07
 */

// MARK: - IntelligentModelRecommendation Extensions

extension IntelligentModelRecommendation {
    
    var priorityScore: Double {
        let typeWeight = Double(5 - recommendationType.priority) / 4.0
        return (confidenceScore * 0.4) + (qualityPrediction * 0.3) + (speedPrediction * 0.2) + (typeWeight * 0.1)
    }
    
    var recommendationIcon: String {
        switch recommendationType {
        case .optimal: return "star.fill"
        case .alternative: return "star"
        case .contextSpecific: return "brain.head.profile"
        case .experimental: return "flask"
        case .fallback: return "arrow.down.circle"
        }
    }
    
    var recommendationColor: Color {
        switch recommendationType {
        case .optimal: return .green
        case .alternative: return .blue
        case .contextSpecific: return .purple
        case .experimental: return .orange
        case .fallback: return .gray
        }
    }
    
    var formattedRecommendationType: String {
        return recommendationType.displayName
    }
    
    var confidenceBadgeText: String {
        return "\(Int(confidenceScore * 100))% confident"
    }
    
    var qualitySpeedRatio: Double {
        guard speedPrediction > 0 else { return 0 }
        return qualityPrediction / speedPrediction
    }
    
    func estimatedResponseTime(for taskComplexity: Double) -> Double {
        let baseTime = speedPrediction
        let complexityMultiplier = 1.0 + (taskComplexity * 2.0)
        return baseTime * complexityMultiplier
    }
    
    func resourceEfficiencyScore() -> Double {
        let memoryEfficiency = max(0, 1.0 - (resourceRequirements.minMemoryGB / 64.0))
        let cpuEfficiency = max(0, 1.0 - (expectedPerformance.cpuUtilization / 100.0))
        return (memoryEfficiency + cpuEfficiency) / 2.0
    }
    
    func suitabilityForTask(_ taskComplexity: EnhancedTaskComplexity) -> Double {
        var suitability = 0.0
        
        // Domain alignment
        let domainMatch = taskComplexity.domainComplexity.primaryDomain
        suitability += context.userPreferences["preferred_domain"] == domainMatch ? 0.3 : 0.1
        
        // Complexity alignment
        let complexityAlignment = 1.0 - abs(taskComplexity.overallComplexity - qualityPrediction)
        suitability += complexityAlignment * 0.4
        
        // Resource compatibility
        let resourceMatch = resourceRequirements.minMemoryGB <= taskComplexity.resourcePredictions.memoryUtilization * 0.8
        suitability += resourceMatch ? 0.3 : 0.0
        
        return min(suitability, 1.0)
    }
}

// MARK: - Array Extensions

extension Array where Element == IntelligentModelRecommendation {
    
    func sortedByPriority() -> [IntelligentModelRecommendation] {
        return sorted { $0.priorityScore > $1.priorityScore }
    }
    
    func sortedByConfidence() -> [IntelligentModelRecommendation] {
        return sorted { $0.confidenceScore > $1.confidenceScore }
    }
    
    func sortedByQuality() -> [IntelligentModelRecommendation] {
        return sorted { $0.qualityPrediction > $1.qualityPrediction }
    }
    
    func sortedBySpeed() -> [IntelligentModelRecommendation] {
        return sorted { $0.speedPrediction > $1.speedPrediction }
    }
    
    func filteredBy(type: IntelligentModelRecommendation.RecommendationType) -> [IntelligentModelRecommendation] {
        return filter { $0.recommendationType == type }
    }
    
    func filteredBy(minConfidence: Double) -> [IntelligentModelRecommendation] {
        return filter { $0.confidenceScore >= minConfidence }
    }
    
    func topRecommendations(limit: Int = 3) -> [IntelligentModelRecommendation] {
        return sortedByPriority().prefix(limit).map { $0 }
    }
    
    func averageConfidence() -> Double {
        guard !isEmpty else { return 0.0 }
        return reduce(0) { $0 + $1.confidenceScore } / Double(count)
    }
    
    func diversityScore() -> Double {
        let uniqueTypes = Set(map { $0.recommendationType })
        return Double(uniqueTypes.count) / Double(IntelligentModelRecommendation.RecommendationType.allCases.count)
    }
}

// MARK: - TaskComplexity Extensions

extension EnhancedTaskComplexity {
    
    var complexityGrade: String {
        switch overallComplexity {
        case 0.0..<0.2: return "Very Simple"
        case 0.2..<0.4: return "Simple"
        case 0.4..<0.6: return "Moderate"
        case 0.6..<0.8: return "Complex"
        default: return "Very Complex"
        }
    }
    
    var complexityColor: Color {
        switch overallComplexity {
        case 0.0..<0.2: return .green
        case 0.2..<0.4: return .blue
        case 0.4..<0.6: return .yellow
        case 0.6..<0.8: return .orange
        default: return .red
        }
    }
    
    var recommendedModelSize: String {
        switch overallComplexity {
        case 0.0..<0.3: return "small"
        case 0.3..<0.7: return "medium"
        default: return "large"
        }
    }
    
    var estimatedProcessingTime: Double {
        let baseTime = Double(computationalComplexity.estimatedTokens) / 100.0 // tokens per second estimation
        let complexityMultiplier = 1.0 + overallComplexity
        return baseTime * complexityMultiplier
    }
    
    func requiresSpecializedModel() -> Bool {
        return domainComplexity.crossDomainComplexity > 0.7 || 
               cognitiveComplexity.reasoningRequired ||
               linguisticComplexity.vocabularyComplexity > 0.8
    }
    
    func getOptimalContextLength() -> Int {
        let baseContext = contextRequirements.minContextLength
        let complexityBonus = Int(Double(baseContext) * overallComplexity * 0.5)
        return min(baseContext + complexityBonus, contextRequirements.optimalContextLength)
    }
}

// MARK: - UserPreferenceProfile Extensions

extension UserPreferenceProfile {
    
    var preferenceStability: String {
        switch adaptationMetrics.preferenceStability {
        case 0.8...1.0: return "Very Stable"
        case 0.6..<0.8: return "Stable"
        case 0.4..<0.6: return "Moderately Stable"
        case 0.2..<0.4: return "Unstable"
        default: return "Very Unstable"
        }
    }
    
    var learningProgressDescription: String {
        switch learningProgress {
        case 0.8...1.0: return "Fully Adapted"
        case 0.6..<0.8: return "Well Adapted"
        case 0.4..<0.6: return "Adapting"
        case 0.2..<0.4: return "Learning"
        default: return "Initial Learning"
        }
    }
    
    func getDominantPreference() -> String {
        let weights = preferenceWeights.normalized
        let maxWeight = max(weights.speed, weights.quality, weights.resourceEfficiency, weights.novelty, weights.reliability)
        
        switch maxWeight {
        case weights.speed: return "Speed"
        case weights.quality: return "Quality"
        case weights.resourceEfficiency: return "Efficiency"
        case weights.novelty: return "Novelty"
        default: return "Reliability"
        }
    }
    
    func getRecommendedModelCharacteristics() -> [String: Double] {
        let weights = preferenceWeights.normalized
        
        return [
            "model_size_preference": weights.quality + (weights.reliability * 0.5),
            "speed_importance": weights.speed + (weights.resourceEfficiency * 0.3),
            "experimental_tolerance": weights.novelty,
            "resource_consciousness": weights.resourceEfficiency + (weights.reliability * 0.2)
        ]
    }
    
    func predictSatisfactionFor(recommendation: IntelligentModelRecommendation) -> Double {
        let weights = preferenceWeights.normalized
        
        let speedSatisfaction = recommendation.speedPrediction * weights.speed
        let qualitySatisfaction = recommendation.qualityPrediction * weights.quality
        let efficiencySatisfaction = recommendation.resourceEfficiencyScore() * weights.resourceEfficiency
        let reliabilitySatisfaction = recommendation.confidenceScore * weights.reliability
        let noveltySatisfaction = (recommendation.recommendationType == .experimental ? 1.0 : 0.5) * weights.novelty
        
        return speedSatisfaction + qualitySatisfaction + efficiencySatisfaction + reliabilitySatisfaction + noveltySatisfaction
    }
}

// MARK: - View Helper Components

struct RecommendationCard: View {
    let recommendation: IntelligentModelRecommendation
    let onTap: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with model name and type
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(recommendation.modelName)
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Text(recommendation.formattedRecommendationType)
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(recommendation.recommendationColor.opacity(0.2))
                        .foregroundColor(recommendation.recommendationColor)
                        .cornerRadius(6)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Image(systemName: recommendation.recommendationIcon)
                        .foregroundColor(recommendation.recommendationColor)
                        .font(.title2)
                    
                    Text(recommendation.confidenceBadgeText)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            // Performance predictions
            HStack(spacing: 16) {
                MetricIndicator(
                    title: "Quality",
                    value: recommendation.qualityPrediction,
                    color: .blue,
                    icon: "star"
                )
                
                MetricIndicator(
                    title: "Speed",
                    value: recommendation.speedPrediction,
                    color: .green,
                    icon: "bolt"
                )
                
                MetricIndicator(
                    title: "Efficiency",
                    value: recommendation.resourceEfficiencyScore(),
                    color: .orange,
                    icon: "leaf"
                )
            }
            
            // Key reasoning points
            if !recommendation.reasoning.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Why this model:")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(.secondary)
                    
                    ForEach(recommendation.reasoning.prefix(2), id: \.self) { reason in
                        HStack(spacing: 6) {
                            Circle()
                                .fill(recommendation.recommendationColor)
                                .frame(width: 4, height: 4)
                            
                            Text(reason)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.controlBackgroundColor))
        .cornerRadius(12)
        .onTapGesture(perform: onTap)
    }
}

struct MetricIndicator: View {
    let title: String
    let value: Double
    let color: Color
    let icon: String
    
    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption)
                    .foregroundColor(color)
                
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text("\(Int(value * 100))%")
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(color)
        }
    }
}

struct RequirementRow: View {
    let title: String
    var required: Bool = false
    var beneficial: Bool = false
    
    var body: some View {
        HStack {
            Image(systemName: iconName)
                .foregroundColor(iconColor)
                .frame(width: 16)
            
            Text(title)
                .font(.subheadline)
            
            Spacer()
            
            Text(statusText)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(iconColor)
        }
    }
    
    private var iconName: String {
        if required { return "checkmark.circle.fill" }
        if beneficial { return "plus.circle.fill" }
        return "circle"
    }
    
    private var iconColor: Color {
        if required { return .green }
        if beneficial { return .blue }
        return .gray
    }
    
    private var statusText: String {
        if required { return "Required" }
        if beneficial { return "Beneficial" }
        return "Optional"
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(.blue)
                    .frame(width: 16)
                
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
            }
            
            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
        }
        .padding(12)
        .background(Color(.controlBackgroundColor))
        .cornerRadius(8)
    }
}
