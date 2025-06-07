import Foundation

// SANDBOX FILE: For testing/development. See .cursorrules.
/**
 * Purpose: Scan for local models in common directories (HuggingFace cache, etc.)
 * Issues & Complexity Summary: Model detection across different directory structures
 * Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~150
  - Core Algorithm Complexity: Medium
  - Dependencies: 2 New, 1 Mod
  - State Management Complexity: Low
  - Novelty/Uncertainty Factor: Medium
 * AI Pre-Task Self-Assessment: 78%
 * Problem Estimate: 75%
 * Initial Code Complexity Estimate: 78%
 * Final Code Complexity: 82%
 * Overall Result Score: 85%
 * Key Variances/Learnings: Model directory scanning patterns vary significantly
 * Last Updated: 2025-06-07
 */

class GenericModelScanner {
    private var foundModels: [LocalModel] = []
    
    struct LocalModel {
        let name: String
        let path: String
        let format: ModelFormat
        let sizeBytes: Int64
    }
    
    enum ModelFormat {
        case gguf
        case safeTensors
        case pytorch
        case onnx
        case unknown
    }
    
    func scanForModels() -> [LocalModel] {
        foundModels.removeAll()
        
        // Common model directories
        let directories = [
            "~/.cache/huggingface/transformers/",
            "~/.cache/huggingface/hub/",
            "~/Library/Application Support/com.lmstudio.LMStudio/models/",
            "~/.ollama/models/"
        ]
        
        for dir in directories {
            let expandedPath = NSString(string: dir).expandingTildeInPath
            scanDirectory(expandedPath)
        }
        
        return foundModels
    }
    
    private func scanDirectory(_ path: String) {
        let fileManager = FileManager.default
        
        guard let enumerator = fileManager.enumerator(atPath: path) else { return }
        
        while let fileName = enumerator.nextObject() as? String {
            let fullPath = path + "/" + fileName
            
            if isModelFile(fileName) {
                let model = LocalModel(
                    name: extractModelName(fileName),
                    path: fullPath,
                    format: detectFormat(fileName),
                    sizeBytes: getFileSize(fullPath)
                )
                foundModels.append(model)
            }
        }
    }
    
    private func isModelFile(_ fileName: String) -> Bool {
        let modelExtensions = [".gguf", ".bin", ".safetensors", ".onnx", ".pt", ".pth"]
        return modelExtensions.contains { fileName.hasSuffix($0) }
    }
    
    private func detectFormat(_ fileName: String) -> ModelFormat {
        if fileName.hasSuffix(".gguf") { return .gguf }
        if fileName.hasSuffix(".safetensors") { return .safeTensors }
        if fileName.hasSuffix(".pt") || fileName.hasSuffix(".pth") { return .pytorch }
        if fileName.hasSuffix(".onnx") { return .onnx }
        return .unknown
    }
    
    private func extractModelName(_ fileName: String) -> String {
        return NSString(string: fileName).deletingPathExtension
    }
    
    private func getFileSize(_ path: String) -> Int64 {
        let fileManager = FileManager.default
        guard let attributes = try? fileManager.attributesOfItem(atPath: path) else { return 0 }
        return attributes[.size] as? Int64 ?? 0
    }
}
