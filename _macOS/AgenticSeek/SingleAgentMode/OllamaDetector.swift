
import Foundation

struct LocalModelInfo {
    let name: String
    let path: String
    let size_gb: Double
    let format: String
    let parameters: String
    let platform: String
    let capabilities: [String]
    var performance_score: Double = 0.0
    var compatibility_score: Double = 0.0
}

struct ModelCompatibility {
    let score: Double
    let recommendations: [String]
    let limitations: [String]
}

class OllamaDetector {
    
    func isOllamaInstalled() -> Bool {
        // Check common Ollama installation paths
        let possiblePaths = [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            "~/.ollama/bin/ollama"
        ]
        
        for path in possiblePaths {
            let expandedPath = NSString(string: path).expandingTildeInPath
            if FileManager.default.fileExists(atPath: expandedPath) {
                return true
            }
        }
        
        // Check if ollama command is available
        let process = Process()
        process.launchPath = "/usr/bin/which"
        process.arguments = ["ollama"]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        process.launch()
        process.waitUntilExit()
        
        return process.terminationStatus == 0
    }
    
    func getOllamaPath() -> String? {
        let possiblePaths = [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            "~/.ollama/bin/ollama"
        ]
        
        for path in possiblePaths {
            let expandedPath = NSString(string: path).expandingTildeInPath
            if FileManager.default.fileExists(atPath: expandedPath) {
                return expandedPath
            }
        }
        
        return nil
    }
    
    func discoverModels() -> [LocalModelInfo] {
        guard isOllamaInstalled() else { return [] }
        
        // Execute ollama list command
        let process = Process()
        process.launchPath = "/usr/bin/env"
        process.arguments = ["ollama", "list"]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        process.launch()
        process.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        
        return parseOllamaModels(output)
    }
    
    private func parseOllamaModels(_ output: String) -> [LocalModelInfo] {
        var models: [LocalModelInfo] = []
        let lines = output.components(separatedBy: .newlines)
        
        for line in lines.dropFirst() { // Skip header
            let components = line.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
            if components.count >= 3 {
                let name = components[0]
                let size = components[2]
                
                let model = LocalModelInfo(
                    name: name,
                    path: "~/.ollama/models/\(name)",
                    size_gb: parseSizeToGB(size),
                    format: "gguf",
                    parameters: extractParameters(from: name),
                    platform: "ollama",
                    capabilities: ["text-generation", "conversation"],
                    performance_score: 0.8,
                    compatibility_score: 0.9
                )
                models.append(model)
            }
        }
        
        return models
    }
    
    private func parseSizeToGB(_ sizeString: String) -> Double {
        let size = sizeString.lowercased()
        if size.contains("gb") {
            return Double(size.replacingOccurrences(of: "gb", with: "")) ?? 0.0
        } else if size.contains("mb") {
            return (Double(size.replacingOccurrences(of: "mb", with: "")) ?? 0.0) / 1024.0
        }
        return 0.0
    }
    
    private func extractParameters(from modelName: String) -> String {
        let name = modelName.lowercased()
        if name.contains("7b") { return "7B" }
        if name.contains("13b") { return "13B" }
        if name.contains("70b") { return "70B" }
        return "Unknown"
    }
    
    func validateModelCompatibility(_ model: LocalModelInfo) -> ModelCompatibility {
        var score = 1.0
        var recommendations: [String] = []
        var limitations: [String] = []
        
        // Check system requirements
        let systemRAM = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)
        let requiredRAM = model.size_gb * 1.5 // Rough estimate
        
        if Double(systemRAM) < requiredRAM {
            score *= 0.5
            limitations.append("Insufficient RAM for optimal performance")
            recommendations.append("Consider upgrading RAM or using a smaller model")
        }
        
        return ModelCompatibility(
            score: score,
            recommendations: recommendations,
            limitations: limitations
        )
    }
}


// MARK: - Enhanced Features

extension OllamaDetector {
    
    private static var cachedModels: [LocalModelInfo]?
    private static var lastCacheUpdate: Date?
    
    func discoverModelsWithCaching(forceRefresh: Bool = false) -> [LocalModelInfo] {
        let cacheExpiry: TimeInterval = 300 // 5 minutes
        
        if !forceRefresh,
           let cached = Self.cachedModels,
           let lastUpdate = Self.lastCacheUpdate,
           Date().timeIntervalSince(lastUpdate) < cacheExpiry {
            return cached
        }
        
        let models = discoverModels()
        Self.cachedModels = models
        Self.lastCacheUpdate = Date()
        return models
    }
    
    func getModelMetadata(_ modelName: String) -> [String: Any]? {
        guard isOllamaInstalled() else { return nil }
        
        let process = Process()
        process.launchPath = "/usr/bin/env"
        process.arguments = ["ollama", "show", modelName]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        process.launch()
        process.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        
        return parseModelMetadata(output)
    }
    
    private func parseModelMetadata(_ output: String) -> [String: Any] {
        var metadata: [String: Any] = [:]
        let lines = output.components(separatedBy: .newlines)
        
        for line in lines {
            if line.contains("Model:") {
                metadata["architecture"] = line.replacingOccurrences(of: "Model:", with: "").trimmingCharacters(in: .whitespaces)
            }
            if line.contains("Parameters:") {
                metadata["parameters"] = line.replacingOccurrences(of: "Parameters:", with: "").trimmingCharacters(in: .whitespaces)
            }
            if line.contains("Quantization:") {
                metadata["quantization"] = line.replacingOccurrences(of: "Quantization:", with: "").trimmingCharacters(in: .whitespaces)
            }
        }
        
        return metadata
    }
    
    func validateModelHealth(_ model: LocalModelInfo) -> Bool {
        // Test if model can be loaded and respond
        guard isOllamaInstalled() else { return false }
        
        let process = Process()
        process.launchPath = "/usr/bin/env"
        process.arguments = ["ollama", "run", model.name, "Hello"]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        process.launch()
        
        // Wait max 10 seconds for response
        DispatchQueue.global().asyncAfter(deadline: .now() + 10) {
            if process.isRunning {
                process.terminate()
            }
        }
        
        process.waitUntilExit()
        return process.terminationStatus == 0
    }
}
