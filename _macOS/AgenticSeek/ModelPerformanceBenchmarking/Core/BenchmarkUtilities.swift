import Foundation
import Combine
import OSLog

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Utility classes and helpers for MLACS Model Performance Benchmarking
 * Issues & Complexity Summary: Helper utilities for data processing and analysis
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~180
   - Core Algorithm Complexity: Medium
   - Dependencies: 3 New
   - State Management Complexity: Low
   - Novelty/Uncertainty Factor: Low
 * AI Pre-Task Self-Assessment: 88%
 * Problem Estimate: 88%
 * Initial Code Complexity Estimate: 85%
 * Final Code Complexity: 87%
 * Overall Result Score: 90%
 * Last Updated: 2025-01-07
 */

// MARK: - Benchmark Data Processor

@MainActor
final class BenchmarkDataProcessor: ObservableObject {
    
    private let logger = Logger(subsystem: "AgenticSeek", category: "BenchmarkDataProcessor")
    
    func processRawBenchmarkData(_ rawData: [String: Any]) -> ComprehensiveBenchmarkResult? {
        logger.info("Processing raw benchmark data")
        
        guard let modelId = rawData["modelId"] as? String,
              let modelName = rawData["modelName"] as? String,
              let providerString = rawData["provider"] as? String,
              let provider = ModelProvider(rawValue: providerString) else {
            logger.error("Invalid raw benchmark data structure")
            return nil
        }
        
        // Process performance metrics
        let performanceMetrics = extractPerformanceMetrics(from: rawData)
        let qualityMetrics = extractQualityMetrics(from: rawData)
        let resourceMetrics = extractResourceMetrics(from: rawData)
        let reliabilityMetrics = extractReliabilityMetrics(from: rawData)
        
        return ComprehensiveBenchmarkResult(
            configurationId: UUID(),
            modelId: modelId,
            modelName: modelName,
            provider: provider,
            testExecutionId: UUID(),
            startTime: Date(),
            endTime: Date(),
            status: .completed,
            performanceMetrics: performanceMetrics,
            qualityMetrics: qualityMetrics,
            resourceMetrics: resourceMetrics,
            reliabilityMetrics: reliabilityMetrics,
            systemEnvironment: getCurrentSystemEnvironment(),
            testConfiguration: getDefaultTestConfiguration(),
            errorLog: []
        )
    }
    
    private func extractPerformanceMetrics(from data: [String: Any]) -> ComprehensiveBenchmarkResult.PerformanceMetrics {
        return ComprehensiveBenchmarkResult.PerformanceMetrics(
            totalInferenceTimeMs: data["totalInferenceTimeMs"] as? Double ?? 0,
            averageTokensPerSecond: data["averageTokensPerSecond"] as? Double ?? 0,
            firstTokenLatencyMs: data["firstTokenLatencyMs"] as? Double ?? 0,
            throughputTokensPerSecond: data["throughputTokensPerSecond"] as? Double ?? 0,
            batchProcessingSpeed: data["batchProcessingSpeed"] as? Double ?? 0,
            concurrentRequestHandling: data["concurrentRequestHandling"] as? Int ?? 1
        )
    }
    
    private func extractQualityMetrics(from data: [String: Any]) -> ComprehensiveBenchmarkResult.QualityMetrics {
        return ComprehensiveBenchmarkResult.QualityMetrics(
            overallQualityScore: data["overallQualityScore"] as? Double ?? 0,
            coherenceScore: data["coherenceScore"] as? Double ?? 0,
            relevanceScore: data["relevanceScore"] as? Double ?? 0,
            factualAccuracy: data["factualAccuracy"] as? Double ?? 0,
            languageQuality: data["languageQuality"] as? Double ?? 0,
            responseCompleteness: data["responseCompleteness"] as? Double ?? 0,
            creativityScore: data["creativityScore"] as? Double ?? 0,
            consistencyScore: data["consistencyScore"] as? Double ?? 0
        )
    }
    
    private func extractResourceMetrics(from data: [String: Any]) -> ComprehensiveBenchmarkResult.ResourceMetrics {
        return ComprehensiveBenchmarkResult.ResourceMetrics(
            peakMemoryUsageMB: data["peakMemoryUsageMB"] as? Double ?? 0,
            averageMemoryUsageMB: data["averageMemoryUsageMB"] as? Double ?? 0,
            peakCPUUsagePercent: data["peakCPUUsagePercent"] as? Double ?? 0,
            averageCPUUsagePercent: data["averageCPUUsagePercent"] as? Double ?? 0,
            gpuUtilizationPercent: data["gpuUtilizationPercent"] as? Double ?? 0,
            thermalState: ThermalState(rawValue: data["thermalState"] as? String ?? "nominal") ?? .nominal,
            powerConsumptionWatts: data["powerConsumptionWatts"] as? Double ?? 0,
            diskIOOperations: data["diskIOOperations"] as? Int ?? 0
        )
    }
    
    private func extractReliabilityMetrics(from data: [String: Any]) -> ComprehensiveBenchmarkResult.ReliabilityMetrics {
        return ComprehensiveBenchmarkResult.ReliabilityMetrics(
            successRate: data["successRate"] as? Double ?? 1.0,
            errorRate: data["errorRate"] as? Double ?? 0.0,
            timeoutRate: data["timeoutRate"] as? Double ?? 0.0,
            retryCount: data["retryCount"] as? Int ?? 0,
            stabilityScore: data["stabilityScore"] as? Double ?? 1.0,
            recoverabilityScore: data["recoverabilityScore"] as? Double ?? 1.0
        )
    }
    
    private func getCurrentSystemEnvironment() -> SystemEnvironment {
        let processInfo = ProcessInfo.processInfo
        
        return SystemEnvironment(
            osVersion: processInfo.operatingSystemVersionString,
            deviceModel: getDeviceModel(),
            processorType: getProcessorType(),
            totalMemoryGB: getTotalMemoryGB(),
            availableMemoryGB: getAvailableMemoryGB(),
            gpuModel: getGPUModel(),
            thermalConditions: .nominal,
            powerSource: .adapter
        )
    }
    
    private func getDefaultTestConfiguration() -> TestConfiguration {
        return TestConfiguration(
            concurrentRequests: 1,
            timeoutSeconds: 30.0,
            retryAttempts: 3,
            warmupRuns: 1,
            measurementRuns: 5,
            cooldownTimeSeconds: 2.0
        )
    }
    
    // MARK: - System Information Helpers
    
    private func getDeviceModel() -> String {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &model, &size, nil, 0)
        return String(cString: model)
    }
    
    private func getProcessorType() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var processor = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &processor, &size, nil, 0)
        return String(cString: processor)
    }
    
    private func getTotalMemoryGB() -> Double {
        var size = MemoryLayout<Int64>.size
        var physicalMemory: Int64 = 0
        sysctlbyname("hw.memsize", &physicalMemory, &size, nil, 0)
        return Double(physicalMemory) / (1024 * 1024 * 1024)
    }
    
    private func getAvailableMemoryGB() -> Double {
        let host = mach_host_self()
        var hostInfo = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        
        let result = withUnsafeMutablePointer(to: &hostInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                host_statistics64(host, HOST_VM_INFO64, $0, &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let pageSize = vm_kernel_page_size
            let freePages = hostInfo.free_count
            let availableBytes = Double(freePages * pageSize)
            return availableBytes / (1024 * 1024 * 1024)
        }
        
        return 0.0
    }
    
    private func getGPUModel() -> String? {
        // This would require Metal framework integration for actual GPU detection
        return "Apple M-Series GPU"
    }
}

// MARK: - Benchmark Report Generator

final class BenchmarkReportGenerator {
    
    private let logger = Logger(subsystem: "AgenticSeek", category: "BenchmarkReportGenerator")
    
    func generateComprehensiveReport(for results: [ComprehensiveBenchmarkResult]) -> BenchmarkReport {
        logger.info("Generating comprehensive benchmark report for \(results.count) results")
        
        let summary = generateSummary(from: results)
        let modelComparisons = generateModelComparisons(from: results)
        let performanceAnalysis = generatePerformanceAnalysis(from: results)
        let recommendations = generateRecommendations(from: results)
        
        return BenchmarkReport(
            id: UUID(),
            generatedAt: Date(),
            totalBenchmarks: results.count,
            summary: summary,
            modelComparisons: modelComparisons,
            performanceAnalysis: performanceAnalysis,
            recommendations: recommendations,
            rawResults: results
        )
    }
    
    private func generateSummary(from results: [ComprehensiveBenchmarkResult]) -> ReportSummary {
        let avgPerformance = results.averagePerformance()
        let successRate = Double(results.filter { $0.status == .completed }.count) / Double(results.count)
        let topPerformer = results.sortedByPerformance().first
        
        return ReportSummary(
            averageInferenceTime: avgPerformance?.totalInferenceTimeMs ?? 0,
            averageTokensPerSecond: avgPerformance?.averageTokensPerSecond ?? 0,
            successRate: successRate,
            topPerformingModel: topPerformer?.modelName ?? "N/A",
            totalExecutionTime: calculateTotalExecutionTime(from: results)
        )
    }
    
    private func generateModelComparisons(from results: [ComprehensiveBenchmarkResult]) -> [ModelComparison] {
        let groupedResults = Dictionary(grouping: results) { $0.modelId }
        
        return groupedResults.compactMap { modelId, modelResults in
            guard let firstResult = modelResults.first else { return nil }
            
            let avgPerformance = modelResults.averagePerformance()
            let avgQuality = modelResults.reduce(0) { $0 + $1.qualityMetrics.overallQualityScore } / Double(modelResults.count)
            
            return ModelComparison(
                modelId: modelId,
                modelName: firstResult.modelName,
                provider: firstResult.provider,
                benchmarkCount: modelResults.count,
                averagePerformance: avgPerformance,
                averageQualityScore: avgQuality,
                rank: 0 // Will be calculated after sorting
            )
        }.sorted { $0.averageQualityScore > $1.averageQualityScore }
          .enumerated()
          .map { index, comparison in
              var updatedComparison = comparison
              updatedComparison.rank = index + 1
              return updatedComparison
          }
    }
    
    private func generatePerformanceAnalysis(from results: [ComprehensiveBenchmarkResult]) -> PerformanceAnalysis {
        let speedTrends = analyzeSpeedTrends(from: results)
        let resourceUtilization = analyzeResourceUtilization(from: results)
        let reliabilityMetrics = analyzeReliability(from: results)
        
        return PerformanceAnalysis(
            speedTrends: speedTrends,
            resourceUtilization: resourceUtilization,
            reliabilityMetrics: reliabilityMetrics,
            identifiedBottlenecks: identifyBottlenecks(from: results)
        )
    }
    
    private func generateRecommendations(from results: [ComprehensiveBenchmarkResult]) -> [String] {
        var recommendations: [String] = []
        
        let avgCPU = results.reduce(0) { $0 + $1.resourceMetrics.averageCPUUsagePercent } / Double(results.count)
        let avgMemory = results.reduce(0) { $0 + $1.resourceMetrics.averageMemoryUsageMB } / Double(results.count)
        let avgQuality = results.reduce(0) { $0 + $1.qualityMetrics.overallQualityScore } / Double(results.count)
        
        if avgCPU > 80 {
            recommendations.append("Consider optimizing for lower CPU usage - current average: \(String(format: "%.1f%%", avgCPU))")
        }
        
        if avgMemory > 8000 {
            recommendations.append("Memory usage is high - consider model optimization or increased system memory")
        }
        
        if avgQuality < 0.7 {
            recommendations.append("Quality scores are below recommended threshold - consider prompt engineering or model fine-tuning")
        }
        
        return recommendations
    }
    
    // Helper methods for analysis
    private func calculateTotalExecutionTime(from results: [ComprehensiveBenchmarkResult]) -> TimeInterval {
        return results.reduce(0) { total, result in
            total + result.endTime.timeIntervalSince(result.startTime)
        }
    }
    
    private func analyzeSpeedTrends(from results: [ComprehensiveBenchmarkResult]) -> [Double] {
        return results.map { $0.performanceMetrics.averageTokensPerSecond }
    }
    
    private func analyzeResourceUtilization(from results: [ComprehensiveBenchmarkResult]) -> ResourceUtilizationAnalysis {
        let avgCPU = results.reduce(0) { $0 + $1.resourceMetrics.averageCPUUsagePercent } / Double(results.count)
        let avgMemory = results.reduce(0) { $0 + $1.resourceMetrics.averageMemoryUsageMB } / Double(results.count)
        let avgGPU = results.reduce(0) { $0 + $1.resourceMetrics.gpuUtilizationPercent } / Double(results.count)
        
        return ResourceUtilizationAnalysis(
            averageCPUUsage: avgCPU,
            averageMemoryUsage: avgMemory,
            averageGPUUsage: avgGPU
        )
    }
    
    private func analyzeReliability(from results: [ComprehensiveBenchmarkResult]) -> ReliabilityAnalysis {
        let avgStability = results.reduce(0) { $0 + $1.reliabilityMetrics.stabilityScore } / Double(results.count)
        let avgErrorRate = results.reduce(0) { $0 + $1.reliabilityMetrics.errorRate } / Double(results.count)
        
        return ReliabilityAnalysis(
            averageStabilityScore: avgStability,
            averageErrorRate: avgErrorRate
        )
    }
    
    private func identifyBottlenecks(from results: [ComprehensiveBenchmarkResult]) -> [String] {
        var bottlenecks: [String] = []
        
        let highLatencyResults = results.filter { $0.performanceMetrics.firstTokenLatencyMs > 1000 }
        let highMemoryResults = results.filter { $0.resourceMetrics.peakMemoryUsageMB > 16000 }
        
        if !highLatencyResults.isEmpty {
            bottlenecks.append("High first token latency detected in \(highLatencyResults.count) tests")
        }
        
        if !highMemoryResults.isEmpty {
            bottlenecks.append("Excessive memory usage detected in \(highMemoryResults.count) tests")
        }
        
        return bottlenecks
    }
}

// MARK: - Report Data Structures

struct BenchmarkReport: Identifiable {
    let id: UUID
    let generatedAt: Date
    let totalBenchmarks: Int
    let summary: ReportSummary
    let modelComparisons: [ModelComparison]
    let performanceAnalysis: PerformanceAnalysis
    let recommendations: [String]
    let rawResults: [ComprehensiveBenchmarkResult]
}

struct ReportSummary {
    let averageInferenceTime: Double
    let averageTokensPerSecond: Double
    let successRate: Double
    let topPerformingModel: String
    let totalExecutionTime: TimeInterval
}

struct ModelComparison {
    let modelId: String
    let modelName: String
    let provider: ModelProvider
    let benchmarkCount: Int
    let averagePerformance: ComprehensiveBenchmarkResult.PerformanceMetrics?
    let averageQualityScore: Double
    var rank: Int
}

struct PerformanceAnalysis {
    let speedTrends: [Double]
    let resourceUtilization: ResourceUtilizationAnalysis
    let reliabilityMetrics: ReliabilityAnalysis
    let identifiedBottlenecks: [String]
}

struct ResourceUtilizationAnalysis {
    let averageCPUUsage: Double
    let averageMemoryUsage: Double
    let averageGPUUsage: Double
}

struct ReliabilityAnalysis {
    let averageStabilityScore: Double
    let averageErrorRate: Double
}
