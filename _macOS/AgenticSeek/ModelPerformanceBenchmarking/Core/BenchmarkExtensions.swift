import Foundation
import SwiftUI
import Charts

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Extensions and utilities for MLACS Model Performance Benchmarking
 * Issues & Complexity Summary: Helper methods and computed properties for enhanced functionality
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~150
   - Core Algorithm Complexity: Low
   - Dependencies: 3 New
   - State Management Complexity: Low
   - Novelty/Uncertainty Factor: Low
 * AI Pre-Task Self-Assessment: 92%
 * Problem Estimate: 88%
 * Initial Code Complexity Estimate: 80%
 * Final Code Complexity: 82%
 * Overall Result Score: 94%
 * Last Updated: 2025-01-07
 */

// MARK: - BenchmarkResult Extensions

extension ComprehensiveBenchmarkResult {
    
    var formattedDuration: String {
        let duration = endTime.timeIntervalSince(startTime)
        if duration < 60 {
            return String(format: "%.1fs", duration)
        } else if duration < 3600 {
            return String(format: "%.1fm", duration / 60)
        } else {
            return String(format: "%.1fh", duration / 3600)
        }
    }
    
    var overallPerformanceGrade: PerformanceGrade {
        let speedScore = min(1.0, performanceMetrics.averageTokensPerSecond / 50.0)
        let qualityScore = qualityMetrics.overallQualityScore
        let reliabilityScore = reliabilityMetrics.stabilityScore
        let resourceScore = 1.0 - (resourceMetrics.averageCPUUsagePercent / 100.0)
        
        let overallScore = (speedScore + qualityScore + reliabilityScore + resourceScore) / 4.0
        
        switch overallScore {
        case 0.9...1.0: return .excellent
        case 0.8..<0.9: return .good
        case 0.7..<0.8: return .fair
        case 0.6..<0.7: return .poor
        default: return .failing
        }
    }
    
    var isWithinThresholds: Bool {
        // Check if the benchmark result meets performance thresholds
        return performanceMetrics.totalInferenceTimeMs < 5000 &&
               qualityMetrics.overallQualityScore > 0.7 &&
               reliabilityMetrics.errorRate < 0.05
    }
    
    func comparisonMetrics(to other: ComprehensiveBenchmarkResult) -> ComparisonMetrics {
        return ComparisonMetrics(
            speedImprovement: (performanceMetrics.averageTokensPerSecond / other.performanceMetrics.averageTokensPerSecond) - 1.0,
            qualityImprovement: qualityMetrics.overallQualityScore - other.qualityMetrics.overallQualityScore,
            memoryEfficiency: (other.resourceMetrics.averageMemoryUsageMB / resourceMetrics.averageMemoryUsageMB) - 1.0,
            reliabilityImprovement: reliabilityMetrics.stabilityScore - other.reliabilityMetrics.stabilityScore
        )
    }
}

enum PerformanceGrade: String, CaseIterable {
    case excellent = "A+"
    case good = "A"
    case fair = "B"
    case poor = "C"
    case failing = "F"
    
    var color: Color {
        switch self {
        case .excellent: return .green
        case .good: return .blue
        case .fair: return .yellow
        case .poor: return .orange
        case .failing: return .red
        }
    }
    
    var description: String {
        switch self {
        case .excellent: return "Exceptional performance across all metrics"
        case .good: return "Strong performance with minor optimization opportunities"
        case .fair: return "Adequate performance with room for improvement"
        case .poor: return "Below average performance, optimization recommended"
        case .failing: return "Poor performance, significant issues detected"
        }
    }
}

struct ComparisonMetrics {
    let speedImprovement: Double
    let qualityImprovement: Double
    let memoryEfficiency: Double
    let reliabilityImprovement: Double
    
    var overallImprovement: Double {
        return (speedImprovement + qualityImprovement + memoryEfficiency + reliabilityImprovement) / 4.0
    }
}

// MARK: - Date Extensions

extension Date {
    
    func relativeDateString() -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: self, relativeTo: Date())
    }
    
    func benchmarkDateString() -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: self)
    }
}

// MARK: - Double Extensions

extension Double {
    
    func formatAsTokensPerSecond() -> String {
        if self < 1 {
            return String(format: "%.2f tok/s", self)
        } else if self < 10 {
            return String(format: "%.1f tok/s", self)
        } else {
            return String(format: "%.0f tok/s", self)
        }
    }
    
    func formatAsMemory() -> String {
        if self < 1024 {
            return String(format: "%.0f MB", self)
        } else {
            return String(format: "%.1f GB", self / 1024)
        }
    }
    
    func formatAsPercentage() -> String {
        return String(format: "%.1f%%", self)
    }
    
    func formatAsLatency() -> String {
        if self < 1000 {
            return String(format: "%.0f ms", self)
        } else {
            return String(format: "%.1f s", self / 1000)
        }
    }
}

// MARK: - Array Extensions

extension Array where Element == ComprehensiveBenchmarkResult {
    
    func averagePerformance() -> ModelPerformanceMetrics? {
        guard !isEmpty else { return nil }
        
        let totalInference = reduce(0) { $0 + $1.performanceMetrics.totalInferenceTimeMs } / Double(count)
        let avgTokensPerSec = reduce(0) { $0 + $1.performanceMetrics.averageTokensPerSecond } / Double(count)
        let firstTokenLatency = reduce(0) { $0 + $1.performanceMetrics.firstTokenLatencyMs } / Double(count)
        let throughput = reduce(0) { $0 + $1.performanceMetrics.throughputTokensPerSecond } / Double(count)
        let batchSpeed = reduce(0) { $0 + $1.performanceMetrics.batchProcessingSpeed } / Double(count)
        let concurrentHandling = reduce(0) { $0 + $1.performanceMetrics.concurrentRequestHandling } / count
        
        return ComprehensiveBenchmarkResult.PerformanceMetrics(
            totalInferenceTimeMs: totalInference,
            averageTokensPerSecond: avgTokensPerSec,
            firstTokenLatencyMs: firstTokenLatency,
            throughputTokensPerSecond: throughput,
            batchProcessingSpeed: batchSpeed,
            concurrentRequestHandling: concurrentHandling
        )
    }
    
    func filteredByProvider(_ provider: ModelProvider) -> [ComprehensiveBenchmarkResult] {
        return filter { $0.provider == provider }
    }
    
    func filteredByDateRange(from startDate: Date, to endDate: Date) -> [ComprehensiveBenchmarkResult] {
        return filter { $0.startTime >= startDate && $0.endTime <= endDate }
    }
    
    func sortedByPerformance() -> [ComprehensiveBenchmarkResult] {
        return sorted { first, second in
            first.performanceMetrics.averageTokensPerSecond > second.performanceMetrics.averageTokensPerSecond
        }
    }
    
    func sortedByQuality() -> [ComprehensiveBenchmarkResult] {
        return sorted { first, second in
            first.qualityMetrics.overallQualityScore > second.qualityMetrics.overallQualityScore
        }
    }
    
    func topPerformers(count: Int = 5) -> [ComprehensiveBenchmarkResult] {
        return Array(sortedByPerformance().prefix(count))
    }
}

// MARK: - Color Extensions

extension Color {
    
    static func forPerformanceScore(_ score: Double) -> Color {
        switch score {
        case 0.9...1.0: return .green
        case 0.8..<0.9: return .blue
        case 0.7..<0.8: return .yellow
        case 0.6..<0.7: return .orange
        default: return .red
        }
    }
    
    static func forResourceUsage(_ usage: Double) -> Color {
        switch usage {
        case 0..<30: return .green
        case 30..<60: return .yellow
        case 60..<80: return .orange
        default: return .red
        }
    }
}

// MARK: - Chart Data Helpers

extension ComprehensiveBenchmarkResult {
    
    var chartDataPoints: [ChartDataPoint] {
        return [
            ChartDataPoint(category: "Speed", value: performanceMetrics.averageTokensPerSecond / 50.0),
            ChartDataPoint(category: "Quality", value: qualityMetrics.overallQualityScore),
            ChartDataPoint(category: "Reliability", value: reliabilityMetrics.stabilityScore),
            ChartDataPoint(category: "Efficiency", value: 1.0 - (resourceMetrics.averageCPUUsagePercent / 100.0))
        ]
    }
}

struct ChartDataPoint: Identifiable {
    let id = UUID()
    let category: String
    let value: Double
}
