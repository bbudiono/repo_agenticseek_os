
import Foundation
import IOKit

struct SystemCapabilities {
    let cpu_cores: Int
    let cpu_brand: String
    let total_ram_gb: Double
    let available_ram_gb: Double
    let gpu_info: [[String: Any]]
    let platform: String
    let architecture: String
    let performance_class: String
}

struct ModelRecommendation {
    let modelName: String
    let suitabilityScore: Double
    let expectedPerformance: String
    let memoryRequirement: Double
}

class SystemPerformanceAnalyzer {
    
    func analyzeSystemCapabilities() -> SystemCapabilities {
        let cpuCores = ProcessInfo.processInfo.activeProcessorCount
        let cpuBrand = getCPUBrand()
        let totalRAM = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
        let availableRAM = getAvailableRAM()
        let gpuInfo = getGPUInfo()
        let platform = "macOS"
        let architecture = getArchitecture()
        let performanceClass = calculatePerformanceClass(cpuCores: cpuCores, totalRAM: totalRAM)
        
        return SystemCapabilities(
            cpu_cores: cpuCores,
            cpu_brand: cpuBrand,
            total_ram_gb: totalRAM,
            available_ram_gb: availableRAM,
            gpu_info: gpuInfo,
            platform: platform,
            architecture: architecture,
            performance_class: performanceClass
        )
    }
    
    private func getCPUBrand() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &machine, &size, nil, 0)
        return String(cString: machine)
    }
    
    private func getAvailableRAM() -> Double {
        let pageSize = vm_page_size
        var vmStat = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        
        let result = withUnsafeMutablePointer(to: &vmStat) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        
        guard result == KERN_SUCCESS else { return 0.0 }
        
        let freeMemory = Double(vmStat.free_count) * Double(pageSize)
        return freeMemory / (1024 * 1024 * 1024)
    }
    
    private func getGPUInfo() -> [[String: Any]] {
        // Simplified GPU detection for macOS
        return [["name": "Integrated GPU", "memory": "Shared"]]
    }
    
    private func getArchitecture() -> String {
        var size = 0
        sysctlbyname("hw.target", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.target", &machine, &size, nil, 0)
        let arch = String(cString: machine)
        
        return arch.contains("arm") ? "arm64" : "x86_64"
    }
    
    private func calculatePerformanceClass(cpuCores: Int, totalRAM: Double) -> String {
        if cpuCores >= 8 && totalRAM >= 16 {
            return "high"
        } else if cpuCores >= 4 && totalRAM >= 8 {
            return "medium"
        } else {
            return "low"
        }
    }
    
    func calculatePerformanceScore(_ capabilities: SystemCapabilities) -> Double {
        var score = 0.0
        
        // CPU score (40% of total)
        let cpuScore = min(Double(capabilities.cpu_cores) / 12.0, 1.0) * 0.4
        
        // RAM score (40% of total)
        let ramScore = min(capabilities.total_ram_gb / 32.0, 1.0) * 0.4
        
        // Architecture bonus (20% of total)
        let archScore = capabilities.architecture == "arm64" ? 0.2 : 0.15
        
        score = cpuScore + ramScore + archScore
        
        return min(score, 1.0)
    }
    
    func recommendModels(for capabilities: SystemCapabilities) -> [ModelRecommendation] {
        var recommendations: [ModelRecommendation] = []
        
        if capabilities.performance_class == "high" {
            recommendations.append(ModelRecommendation(
                modelName: "llama3.1:70b",
                suitabilityScore: 0.9,
                expectedPerformance: "Excellent",
                memoryRequirement: 48.0
            ))
        }
        
        if capabilities.performance_class == "medium" || capabilities.performance_class == "high" {
            recommendations.append(ModelRecommendation(
                modelName: "llama3.1:13b",
                suitabilityScore: 0.85,
                expectedPerformance: "Very Good",
                memoryRequirement: 16.0
            ))
        }
        
        // Always recommend lightweight models
        recommendations.append(ModelRecommendation(
            modelName: "llama3.2:3b",
            suitabilityScore: 0.8,
            expectedPerformance: "Good",
            memoryRequirement: 4.0
        ))
        
        return recommendations
    }
}


// MARK: - Advanced System Analysis

extension SystemPerformanceAnalyzer {
    
    func benchmarkSystem() -> [String: Double] {
        var benchmarks: [String: Double] = [:]
        
        // CPU benchmark
        let cpuStart = Date()
        var result = 0
        for i in 0..<1000000 {
            result += i * i
        }
        let cpuTime = Date().timeIntervalSince(cpuStart)
        benchmarks["cpu_score"] = 1.0 / cpuTime * 1000
        
        // Memory benchmark
        let memoryStart = Date()
        let largeArray = Array(0..<100000)
        let sortedArray = largeArray.sorted()
        let memoryTime = Date().timeIntervalSince(memoryStart)
        benchmarks["memory_score"] = 1.0 / memoryTime * 100
        
        return benchmarks
    }
    
    func predictModelPerformance(_ model: LocalModelInfo, capabilities: SystemCapabilities) -> [String: Any] {
        let benchmarks = benchmarkSystem()
        
        let cpuScore = benchmarks["cpu_score"] ?? 0.0
        let memoryScore = benchmarks["memory_score"] ?? 0.0
        
        let estimatedTokensPerSecond = cpuScore * memoryScore / (model.size_gb * 10)
        let estimatedLatency = 1.0 / estimatedTokensPerSecond
        
        return [
            "estimated_tokens_per_second": estimatedTokensPerSecond,
            "estimated_latency": estimatedLatency,
            "memory_efficiency": min(capabilities.available_ram_gb / model.size_gb, 1.0),
            "overall_performance": (estimatedTokensPerSecond + memoryScore) / 2.0
        ]
    }
    
    func generateOptimizationSuggestions(for capabilities: SystemCapabilities) -> [String] {
        var suggestions: [String] = []
        
        if capabilities.available_ram_gb < 8 {
            suggestions.append("Consider closing other applications to free up memory")
            suggestions.append("Use smaller models (3B-7B parameters) for better performance")
        }
        
        if capabilities.cpu_cores < 4 {
            suggestions.append("Consider using quantized models for faster inference")
        }
        
        if capabilities.architecture == "arm64" {
            suggestions.append("Apple Silicon detected - use optimized ARM models when available")
        }
        
        if suggestions.isEmpty {
            suggestions.append("System is well-suited for AI workloads")
        }
        
        return suggestions
    }
}
