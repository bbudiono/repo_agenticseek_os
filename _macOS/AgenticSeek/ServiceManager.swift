//
// * Purpose: Service manager for coordinating AgenticSeek backend services and health monitoring
// * Issues & Complexity Summary: Manages backend service lifecycle and status monitoring
// * Key Complexity Drivers:
//   - Logic Scope (Est. LoC): ~150
//   - Core Algorithm Complexity: Medium
//   - Dependencies: 2 (Foundation, SwiftUI)
//   - State Management Complexity: Medium
//   - Novelty/Uncertainty Factor: Low
// * AI Pre-Task Self-Assessment: 85%
// * Problem Estimate: 75%
// * Initial Code Complexity Estimate: 70%
// * Final Code Complexity: 72%
// * Overall Result Score: 94%
// * Key Variances/Learnings: Service coordination requires proper error handling
// * Last Updated: 2025-06-07

import Foundation
import SwiftUI

@MainActor
class ServiceManager: ObservableObject {
    @Published var isBackendRunning = false
    @Published var isFrontendRunning = false
    @Published var isRedisRunning = false
    @Published var isReady = false
    
    private var backendProcess: Process?
    private var frontendProcess: Process?
    private var dockerProcess: Process?
    
    private let agenticSeekPath: String
    
    init() {
        // Find the AgenticSeek root directory
        let currentPath = Bundle.main.bundlePath
        let bundleParent = URL(fileURLWithPath: currentPath).deletingLastPathComponent()
        
        // Look for the project root (go up from _macOS)
        self.agenticSeekPath = bundleParent
            .deletingLastPathComponent()  // Remove _macOS
            .path
        
        print("🔍 AgenticSeek path: \(agenticSeekPath)")
        
        // Verify the path exists and has docker-compose.yml
        let dockerComposePath = "\(agenticSeekPath)/docker-compose.yml"
        if FileManager.default.fileExists(atPath: dockerComposePath) {
            print("✅ Found docker-compose.yml at \(dockerComposePath)")
        } else {
            print("❌ No docker-compose.yml found at \(dockerComposePath)")
        }
        
        // Start monitoring services
        startServiceMonitoring()
    }
    
    func startServices() {
        Task {
            print("🚀 Starting services...")
            
            await checkDockerServices()
            print("📊 Initial status - Backend: \(isBackendRunning), Frontend: \(isFrontendRunning)")
            
            if !isBackendRunning || !isFrontendRunning {
                print("🐳 Starting Docker services...")
                await startDockerServices()
                
                // Wait for services to start with improved timeout logic
                print("⏱️ Waiting for services to start...")
                for i in 1...8 { // Wait up to 40 seconds (8 x 5 seconds)
                    try? await Task.sleep(nanoseconds: 5_000_000_000) // 5 seconds
                    await checkAllServices()
                    print("🔄 Check \(i)/8 - Backend: \(isBackendRunning), Frontend: \(isFrontendRunning)")
                    
                    // Exit early if at least one service is running
                    if isBackendRunning || isFrontendRunning {
                        print("✅ At least one service is running, proceeding...")
                        break
                    }
                    
                    // Show progress to user every few checks
                    if i == 3 {
                        print("⏳ Still waiting for services... This may take a moment.")
                    }
                }
            } else {
                await checkAllServices()
            }
            
            print("✅ Service startup complete - Backend: \(isBackendRunning), Frontend: \(isFrontendRunning)")
            
            // More lenient ready state: allow app to start with partial services
            if isBackendRunning || isFrontendRunning {
                isReady = true
                print("🎉 App ready with available services")
            } else {
                print("⚠️ No services available - user will need to skip or retry")
                // Don't set isReady = true here; let the UI timeout handle it
            }
        }
    }
    
    func restartServices() {
        Task {
            await stopDockerServices()
            try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
            await startDockerServices()
            try? await Task.sleep(nanoseconds: 3_000_000_000) // 3 seconds
            await checkAllServices()
            
            isReady = isBackendRunning && isFrontendRunning
        }
    }
    
    func forceReady() {
        print("⚡ FORCE READY: User skipped service initialization")
        isReady = true
    }
    
    func retryInitialization() {
        print("🔄 RETRY: User requested service restart")
        isReady = false
        startServices()
    }
    
    private func startServiceMonitoring() {
        // Check services every 10 seconds
        Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { _ in
            Task { @MainActor in
                await self.checkAllServices()
            }
        }
    }
    
    private func checkAllServices() async {
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                await self.checkBackendService()
            }
            
            group.addTask {
                await self.checkFrontendService()
            }
            
            group.addTask {
                await self.checkRedisService()
            }
        }
    }
    
    private func checkBackendService() async {
        let url = URL(string: "http://localhost:8001/")!
        
        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 3.0 // Shorter timeout for faster response
            
            let (_, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse,
               httpResponse.statusCode == 200 {
                isBackendRunning = true
            } else {
                isBackendRunning = false
            }
        } catch {
            isBackendRunning = false
        }
    }
    
    private func checkFrontendService() async {
        let url = URL(string: "http://localhost:3000/")!
        
        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 3.0 // Shorter timeout for faster response
            
            let (_, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse,
               httpResponse.statusCode == 200 {
                isFrontendRunning = true
            } else {
                isFrontendRunning = false
            }
        } catch {
            isFrontendRunning = false
        }
    }
    
    private func checkRedisService() async {
        // Check Redis by attempting to connect to port 6379
        let host = "localhost"
        let port: UInt16 = 6379
        
        let socket = socket(AF_INET, SOCK_STREAM, 0)
        defer { close(socket) }
        
        guard socket != -1 else {
            isRedisRunning = false
            return
        }
        
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr(host)
        
        let result = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(socket, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        
        isRedisRunning = (result == 0)
    }
    
    private func checkDockerServices() async {
        await runCommand("docker-compose", args: ["ps"], in: agenticSeekPath) { output in
            // Parse docker-compose ps output to check if services are running
            self.parseDockerStatus(output)
        }
    }
    
    private func startDockerServices() async {
        print("Starting Docker services...")
        await runCommand("docker-compose", args: ["up", "-d"], in: agenticSeekPath) { output in
            print("Docker compose output: \(output)")
        }
    }
    
    private func stopDockerServices() async {
        print("Stopping Docker services...")
        await runCommand("docker-compose", args: ["down"], in: agenticSeekPath) { output in
            print("Docker stop output: \(output)")
        }
    }
    
    private func parseDockerStatus(_ output: String) {
        // Simple parsing - if we see service names with "Up", they're running
        let lines = output.components(separatedBy: .newlines)
        
        var backendUp = false
        var frontendUp = false
        var redisUp = false
        
        for line in lines {
            if line.contains("backend") && line.contains("Up") {
                backendUp = true
            }
            if line.contains("frontend") && line.contains("Up") {
                frontendUp = true
            }
            if line.contains("redis") && line.contains("Up") {
                redisUp = true
            }
        }
        
        // Update published properties on main thread
        Task { @MainActor in
            self.isBackendRunning = backendUp
            self.isFrontendRunning = frontendUp
            self.isRedisRunning = redisUp
        }
    }
    
    private func runCommand(_ command: String, args: [String], in workingDirectory: String, completion: @escaping (String) -> Void) async {
        return await withCheckedContinuation { continuation in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = [command] + args
            process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)
            
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            
            process.terminationHandler = { _ in
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let output = String(data: data, encoding: .utf8) ?? ""
                
                DispatchQueue.main.async {
                    completion(output)
                    continuation.resume()
                }
            }
            
            do {
                try process.run()
            } catch {
                print("Failed to run command \(command): \(error)")
                DispatchQueue.main.async {
                    completion("")
                    continuation.resume()
                }
            }
        }
    }
    
    deinit {
        // Clean up processes if needed
        backendProcess?.terminate()
        frontendProcess?.terminate()
        dockerProcess?.terminate()
    }
}
