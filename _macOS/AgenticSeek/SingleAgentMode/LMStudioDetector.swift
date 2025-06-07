
import Foundation

class LMStudioDetector {
    
    func isLMStudioInstalled() -> Bool {
        // Check for LM Studio application
        let appPath = "/Applications/LM Studio.app"
        return FileManager.default.fileExists(atPath: appPath)
    }
    
    func getLMStudioPath() -> String? {
        let appPath = "/Applications/LM Studio.app"
        if FileManager.default.fileExists(atPath: appPath) {
            return appPath
        }
        return nil
    }
    
    func getModelDirectory() -> URL? {
        // LM Studio typically stores models in user directory
        let homeDirectory = FileManager.default.homeDirectoryForCurrentUser
        let lmStudioDir = homeDirectory.appendingPathComponent(".cache/lm-studio/models")
        
        if FileManager.default.fileExists(atPath: lmStudioDir.path) {
            return lmStudioDir
        }
        
        return nil
    }
    
    func isAPIAvailable() -> Bool {
        // Check if LM Studio local server is running
        let url = URL(string: "http://localhost:1234/v1/models")!
        let semaphore = DispatchSemaphore(value: 0)
        var isAvailable = false
        
        let task = URLSession.shared.dataTask(with: url) { _, response, _ in
            if let httpResponse = response as? HTTPURLResponse {
                isAvailable = httpResponse.statusCode == 200
            }
            semaphore.signal()
        }
        
        task.resume()
        semaphore.wait()
        
        return isAvailable
    }
    
    func getAPIEndpoint() -> String? {
        if isAPIAvailable() {
            return "http://localhost:1234/v1"
        }
        return nil
    }
}


// MARK: - Enhanced Features
extension LMStudioDetector {
    
    func executeWithLogging() -> Bool {
        print("Starting execution of LM Studio Integration Detection")
        let result = execute()
        print("Execution completed with result: \(result)")
        return result
    }
    
    func getStatus() -> [String: Any] {
        return [
            "component": "LMStudioDetector.swift",
            "last_updated": Date().timeIntervalSince1970,
            "status": "active"
        ]
    }
}
