//
// VoiceAIBridge.swift
// AgenticSeek Enhanced macOS
//
// Bridge between SwiftUI frontend and Python voice backend
// Provides real-time communication via WebSocket and HTTP API
//

import Foundation
import Combine
import OSLog
import Network

/// Bridge for real-time communication with Python voice backend
/// Handles WebSocket connections, HTTP API calls, and state synchronization
@MainActor
class VoiceAIBridge: NSObject, ObservableObject {
    
    // MARK: - Published Properties
    @Published var isConnected = false
    @Published var connectionStatus: ConnectionStatus = .disconnected
    @Published var backendVoiceStatus: BackendVoiceStatus = .idle
    @Published var currentTranscription = ""
    @Published var lastResponse = ""
    @Published var confidenceScore: Double = 0.0
    @Published var activeAgents: [String] = []
    @Published var currentTask = ""
    @Published var processingTime: Double = 0.0
    
    // MARK: - Private Properties
    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession
    private let logger = Logger(subsystem: "com.agenticseek.voice", category: "VoiceAIBridge")
    private var cancellables = Set<AnyCancellable>()
    private var reconnectTimer: Timer?
    private var heartbeatTimer: Timer?
    
    // MARK: - Configuration
    private struct Config {
        static let backendHost = "127.0.0.1"
        static let backendPort = 8765
        static let websocketURL = "ws://\(backendHost):\(backendPort)/ws/voice"
        static let apiBaseURL = "http://\(backendHost):\(backendPort)/api"
        static let reconnectInterval: TimeInterval = 5.0
        static let heartbeatInterval: TimeInterval = 30.0
        static let requestTimeout: TimeInterval = 10.0
    }
    
    // MARK: - Initialization
    override init() {
        // Configure URL session for API calls
        let sessionConfig = URLSessionConfiguration.default
        sessionConfig.timeoutIntervalForRequest = Config.requestTimeout
        sessionConfig.timeoutIntervalForResource = Config.requestTimeout * 2
        
        self.urlSession = URLSession(configuration: sessionConfig)
        super.init()
        
        setupNetworkMonitoring()
        logger.info("VoiceAIBridge initialized")
    }
    
    deinit {
        reconnectTimer?.invalidate()
        heartbeatTimer?.invalidate()
        
        // Force disconnect synchronously in deinit (nonisolated)
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
    }
    
    // MARK: - Public Interface
    
    /// Connect to the Python voice backend
    func connect() {
        guard !isConnected else {
            logger.warning("Already connected to backend")
            return
        }
        
        logger.info("Attempting to connect to Python voice backend...")
        connectionStatus = .connecting
        
        Task {
            await establishWebSocketConnection()
        }
    }
    
    /// Disconnect from the Python voice backend
    func disconnect() {
        logger.info("Disconnecting from Python voice backend...")
        
        connectionStatus = .disconnected
        isConnected = false
        
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        
        reconnectTimer?.invalidate()
        heartbeatTimer?.invalidate()
    }
    
    /// Start voice listening on the backend
    func startVoiceListening() async -> Bool {
        let result: Bool? = await performAPICall(endpoint: "/voice/start", method: "POST")
        return result ?? false
    }
    
    /// Stop voice listening on the backend
    func stopVoiceListening() async -> Bool {
        let result: Bool? = await performAPICall(endpoint: "/voice/stop", method: "POST")
        return result ?? false
    }
    
    /// Send voice command to the backend
    func sendVoiceCommand(_ command: String) async -> Bool {
        let requestBody = ["command": command]
        let result: Bool? = await performAPICall(endpoint: "/voice/command", method: "POST", body: requestBody)
        return result ?? false
    }
    
    /// Get voice capabilities from the backend
    func getVoiceCapabilities() async -> VoiceCapabilities? {
        return await performAPICall(endpoint: "/voice/capabilities", method: "GET")
    }
    
    /// Get performance metrics from the backend
    func getPerformanceMetrics() async -> PerformanceMetrics? {
        return await performAPICall(endpoint: "/voice/metrics", method: "GET")
    }
    
    // MARK: - Private Methods
    
    private func setupNetworkMonitoring() {
        // Monitor network connectivity
        let monitor = NWPathMonitor()
        monitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                if path.status == .satisfied && !(self?.isConnected ?? true) {
                    self?.connect()
                } else if path.status != .satisfied {
                    self?.connectionStatus = .networkUnavailable
                }
            }
        }
        monitor.start(queue: DispatchQueue.global(qos: .background))
    }
    
    private func establishWebSocketConnection() async {
        guard let url = URL(string: Config.websocketURL) else {
            logger.error("Invalid WebSocket URL")
            connectionStatus = .error("Invalid URL")
            return
        }
        
        var request = URLRequest(url: url)
        request.setValue("AgenticSeek-macOS/1.0", forHTTPHeaderField: "User-Agent")
        
        webSocketTask = urlSession.webSocketTask(with: request)
        webSocketTask?.delegate = self
        
        do {
            webSocketTask?.resume()
            
            // Start listening for messages
            await listenForMessages()
            
            isConnected = true
            connectionStatus = .connected
            
            // Start heartbeat
            startHeartbeat()
            
            logger.info("Successfully connected to Python voice backend")
            
        } catch {
            logger.error("WebSocket connection failed: \(error)")
            connectionStatus = .error(error.localizedDescription)
            scheduleReconnect()
        }
    }
    
    private func listenForMessages() async {
        guard let webSocketTask = webSocketTask else { return }
        
        do {
            let message = try await webSocketTask.receive()
            await handleWebSocketMessage(message)
            
            // Continue listening if still connected
            if isConnected {
                await listenForMessages()
            }
        } catch {
            logger.error("WebSocket message receive failed: \(error)")
            handleConnectionLoss()
        }
    }
    
    private func handleWebSocketMessage(_ message: URLSessionWebSocketTask.Message) async {
        switch message {
        case .string(let text):
            await parseJSONMessage(text)
        case .data(let data):
            if let text = String(data: data, encoding: .utf8) {
                await parseJSONMessage(text)
            }
        @unknown default:
            logger.warning("Unknown WebSocket message type received")
        }
    }
    
    private func parseJSONMessage(_ text: String) async {
        guard let data = text.data(using: .utf8) else { return }
        
        do {
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard let messageType = json?["type"] as? String else { return }
            
            let eventData = json?["data"] as? [String: Any] ?? [:]
            
            await handleBackendEvent(type: messageType, data: eventData)
            
        } catch {
            logger.error("Failed to parse WebSocket message: \(error)")
        }
    }
    
    private func handleBackendEvent(type: String, data: [String: Any]) async {
        switch type {
        case "voice_status_changed":
            if let isListening = data["is_listening"] as? Bool {
                backendVoiceStatus = isListening ? .listening : .idle
            }
            
        case "transcription_update":
            currentTranscription = data["transcription"] as? String ?? ""
            confidenceScore = data["confidence"] as? Double ?? 0.0
            
        case "agent_status_update":
            if let statusString = data["agent_status"] as? String {
                backendVoiceStatus = BackendVoiceStatus(rawValue: statusString) ?? .idle
            }
            
        case "task_progress":
            currentTask = data["current_task"] as? String ?? ""
            
        case "response_ready":
            lastResponse = data["response"] as? String ?? ""
            backendVoiceStatus = .completed
            
        case "error_occurred":
            let error = data["error"] as? String ?? "Unknown error"
            logger.error("Backend error: \(error)")
            backendVoiceStatus = .error
            
        case "confirmation_required":
            // Handle confirmation dialog
            logger.info("Backend requires confirmation")
            
        case "pong":
            // Heartbeat response
            logger.debug("Heartbeat pong received")
            
        default:
            logger.debug("Unknown event type: \(type)")
        }
    }
    
    private func performAPICall<T: Codable>(
        endpoint: String,
        method: String,
        body: [String: Any]? = nil
    ) async -> T? {
        guard let url = URL(string: Config.apiBaseURL + endpoint) else {
            logger.error("Invalid API URL for endpoint: \(endpoint)")
            return nil
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("AgenticSeek-macOS/1.0", forHTTPHeaderField: "User-Agent")
        
        if let body = body {
            do {
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
            } catch {
                logger.error("Failed to serialize request body: \(error)")
                return nil
            }
        }
        
        do {
            let (data, response) = try await urlSession.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                logger.error("Invalid response type")
                return nil
            }
            
            guard 200...299 ~= httpResponse.statusCode else {
                logger.error("API call failed with status: \(httpResponse.statusCode)")
                return nil
            }
            
            if T.self == Bool.self {
                // For boolean responses, just return success
                return true as? T
            }
            
            let decoder = JSONDecoder()
            let apiResponse = try decoder.decode(APIResponse<T>.self, from: data)
            return apiResponse.data
            
        } catch {
            logger.error("API call failed: \(error)")
            return nil
        }
    }
    
    private func startHeartbeat() {
        heartbeatTimer?.invalidate()
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: Config.heartbeatInterval, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.sendHeartbeat()
            }
        }
    }
    
    private func sendHeartbeat() {
        guard let webSocketTask = webSocketTask, isConnected else { return }
        
        let heartbeat = [
            "type": "ping",
            "timestamp": ISO8601DateFormatter().string(from: Date())
        ]
        
        do {
            let data = try JSONSerialization.data(withJSONObject: heartbeat)
            let message = URLSessionWebSocketTask.Message.data(data)
            webSocketTask.send(message) { [weak self] error in
                if let error = error {
                    self?.logger.error("Heartbeat send failed: \(error)")
                    Task { @MainActor in
                        self?.handleConnectionLoss()
                    }
                }
            }
        } catch {
            logger.error("Heartbeat serialization failed: \(error)")
        }
    }
    
    private func handleConnectionLoss() {
        logger.warning("Connection lost to Python backend")
        isConnected = false
        connectionStatus = .disconnected
        backendVoiceStatus = .idle
        
        heartbeatTimer?.invalidate()
        scheduleReconnect()
    }
    
    private func scheduleReconnect() {
        reconnectTimer?.invalidate()
        reconnectTimer = Timer.scheduledTimer(withTimeInterval: Config.reconnectInterval, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.connect()
            }
        }
    }
}

// MARK: - URLSessionWebSocketDelegate
extension VoiceAIBridge: URLSessionWebSocketDelegate {
    nonisolated func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        Task { @MainActor in
            logger.info("WebSocket connection opened")
        }
    }
    
    nonisolated func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        Task { @MainActor in
            logger.info("WebSocket connection closed with code: \(closeCode.rawValue)")
            handleConnectionLoss()
        }
    }
}

// MARK: - Supporting Types

enum ConnectionStatus: Equatable {
    case disconnected
    case connecting
    case connected
    case networkUnavailable
    case error(String)
    
    var displayText: String {
        switch self {
        case .disconnected: return "Disconnected"
        case .connecting: return "Connecting..."
        case .connected: return "Connected"
        case .networkUnavailable: return "No Network"
        case .error(let message): return "Error: \(message)"
        }
    }
}

enum BackendVoiceStatus: String, CaseIterable {
    case idle = "idle"
    case listening = "listening"
    case analyzing = "analyzing"
    case executing = "executing"
    case completed = "completed"
    case error = "error"
    
    var displayText: String {
        switch self {
        case .idle: return "Ready"
        case .listening: return "Listening..."
        case .analyzing: return "Analyzing..."
        case .executing: return "Executing..."
        case .completed: return "Completed"
        case .error: return "Error"
        }
    }
}

struct VoiceCapabilities: Codable {
    let productionPipeline: Bool
    let legacyPipeline: Bool
    let streamingTranscription: Bool
    let voiceActivityDetection: Bool
    let noiseReduction: Bool
    let realTimeFeedback: Bool
    let commandRecognition: Bool
    let fallbackSupport: Bool
    
    enum CodingKeys: String, CodingKey {
        case productionPipeline = "production_pipeline"
        case legacyPipeline = "legacy_pipeline"
        case streamingTranscription = "streaming_transcription"
        case voiceActivityDetection = "voice_activity_detection"
        case noiseReduction = "noise_reduction"
        case realTimeFeedback = "real_time_feedback"
        case commandRecognition = "command_recognition"
        case fallbackSupport = "fallback_support"
    }
}

struct PerformanceMetrics: Codable {
    let totalRequests: Int
    let websocketConnections: Int
    let eventsSent: Int
    let apiErrors: Int
    let averageResponseTime: Double
    
    enum CodingKeys: String, CodingKey {
        case totalRequests = "total_requests"
        case websocketConnections = "websocket_connections"
        case eventsSent = "events_sent"
        case apiErrors = "api_errors"
        case averageResponseTime = "average_response_time"
    }
}

struct APIResponse<T: Codable>: Codable {
    let success: Bool
    let data: T
    let timestamp: String
    let sessionId: String
    
    enum CodingKeys: String, CodingKey {
        case success
        case data
        case timestamp
        case sessionId = "session_id"
    }
}
