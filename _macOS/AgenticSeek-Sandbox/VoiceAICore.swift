// SANDBOX FILE: For testing/development. See .cursorrules.
//
// VoiceAICore.swift
// AgenticSeek Enhanced macOS
//
// Voice-enabled AI assistant core with hybrid local/backend processing
// Integrates local Speech Recognition with Python backend for enhanced capabilities
//

import Foundation
import Speech
import AVFoundation
import Combine
import OSLog

/// Core voice AI assistant providing speech recognition, synthesis, and agent orchestration
/// Hybrid implementation: local speech recognition + Python backend processing
@MainActor
class VoiceAICore: NSObject, ObservableObject {
    
    // MARK: - Published Properties
    @Published var isListening = false
    @Published var isProcessing = false
    @Published var isSpeaking = false
    @Published var currentTranscription = ""
    @Published var lastResponse = ""
    @Published var voiceActivated = false
    @Published var agentStatus: AgentStatus = .idle
    @Published var currentTask: String = ""
    @Published var useBackendProcessing = true
    @Published var backendConnectionStatus: ConnectionStatus = .disconnected
    
    // MARK: - Private Properties
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))!
    private let audioEngine = AVAudioEngine()
    private let speechSynthesizer = AVSpeechSynthesizer()
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    
    private let agentOrchestrator = AgentOrchestrator()
    private let taskPlanner = TaskPlanner()
    private let voiceAIBridge = VoiceAIBridge()
    private let logger = Logger(subsystem: "com.agenticseek.voice", category: "VoiceAI")
    
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Voice Configuration
    private struct VoiceConfig {
        static let activationPhrases = ["hey agenti", "agenticseek", "hey assistant"]
        static let speechRate: Float = 0.55
        static let speechPitch: Float = 1.0
        static let speechVolume: Float = 0.8
        static let silenceThreshold: TimeInterval = 2.0
    }
    
    // MARK: - Initialization
    override init() {
        super.init()
        setupVoiceAI()
        setupSpeechSynthesizer()
        setupBackendBridge()
        logger.info("VoiceAICore initialized successfully")
    }
    
    // MARK: - Public Interface
    
    /// Start voice activation listening
    func startVoiceActivation() {
        guard !isListening else { return }
        
        requestSpeechRecognitionPermission { [weak self] authorized in
            guard authorized else {
                self?.logger.error("Speech recognition not authorized")
                return
            }
            
            DispatchQueue.main.async {
                self?.startContinuousListening()
            }
        }
    }
    
    /// Stop voice activation
    func stopVoiceActivation() {
        stopListening()
        voiceActivated = false
        logger.info("Voice activation stopped")
    }
    
    /// Process voice command
    func processVoiceCommand(_ command: String) {
        guard !command.isEmpty else { return }
        
        isProcessing = true
        currentTask = "Processing: \(command)"
        
        Task {
            await handleVoiceCommand(command)
        }
    }
    
    /// Speak response with natural voice
    func speak(_ text: String, priority: SpeechPriority = .normal) {
        guard !text.isEmpty else { return }
        
        // Stop current speech if higher priority
        if priority == .high && isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
        
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = VoiceConfig.speechRate
        utterance.pitchMultiplier = VoiceConfig.speechPitch
        utterance.volume = VoiceConfig.speechVolume
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        
        isSpeaking = true
        speechSynthesizer.speak(utterance)
        
        logger.info("Speaking: \(text)")
    }
    
    // MARK: - Private Methods
    
    private func setupVoiceAI() {
        speechSynthesizer.delegate = self
        speechRecognizer.delegate = self
        
        // Setup audio session for voice processing
        configureAudioSession()
    }
    
    private func setupSpeechSynthesizer() {
        speechSynthesizer.delegate = self
    }
    
    private func setupBackendBridge() {
        // Subscribe to backend bridge updates
        voiceAIBridge.$connectionStatus
            .receive(on: DispatchQueue.main)
            .assign(to: \.backendConnectionStatus, on: self)
            .store(in: &cancellables)
        
        voiceAIBridge.$currentTranscription
            .receive(on: DispatchQueue.main)
            .sink { [weak self] transcription in
                if self?.useBackendProcessing == true {
                    self?.currentTranscription = transcription
                }
            }
            .store(in: &cancellables)
        
        voiceAIBridge.$lastResponse
            .receive(on: DispatchQueue.main)
            .sink { [weak self] response in
                if self?.useBackendProcessing == true {
                    self?.lastResponse = response
                    if !response.isEmpty {
                        self?.speak(response)
                    }
                }
            }
            .store(in: &cancellables)
        
        voiceAIBridge.$backendVoiceStatus
            .receive(on: DispatchQueue.main)
            .sink { [weak self] status in
                if self?.useBackendProcessing == true {
                    self?.agentStatus = self?.mapBackendStatusToAgentStatus(status) ?? .idle
                }
            }
            .store(in: &cancellables)
        
        voiceAIBridge.$currentTask
            .receive(on: DispatchQueue.main)
            .assign(to: \.currentTask, on: self)
            .store(in: &cancellables)
        
        // Attempt to connect to backend
        voiceAIBridge.connect()
    }
    
    private func configureAudioSession() {
        // Note: AVAudioSession is not available on macOS
        // macOS handles audio configuration automatically
        logger.info("Audio configuration handled by macOS system")
    }
    
    private func requestSpeechRecognitionPermission(completion: @escaping (Bool) -> Void) {
        SFSpeechRecognizer.requestAuthorization { status in
            DispatchQueue.main.async {
                completion(status == .authorized)
            }
        }
    }
    
    private func startContinuousListening() {
        guard speechRecognizer.isAvailable else {
            logger.error("Speech recognizer not available")
            return
        }
        
        do {
            let inputNode = audioEngine.inputNode
            
            recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
            guard let recognitionRequest = recognitionRequest else {
                logger.error("Unable to create recognition request")
                return
            }
            
            recognitionRequest.shouldReportPartialResults = true
            
            recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
                self?.handleSpeechRecognitionResult(result: result, error: error)
            }
            
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
                recognitionRequest.append(buffer)
            }
            
            audioEngine.prepare()
            try audioEngine.start()
            
            isListening = true
            logger.info("Started continuous listening for voice activation")
            
        } catch {
            logger.error("Failed to start listening: \(error)")
        }
    }
    
    private func stopListening() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        recognitionTask?.cancel()
        recognitionTask = nil
        isListening = false
    }
    
    private func handleSpeechRecognitionResult(result: SFSpeechRecognitionResult?, error: Error?) {
        if let error = error {
            logger.error("Speech recognition error: \(error)")
            return
        }
        
        guard let result = result else { return }
        
        let transcription = result.bestTranscription.formattedString.lowercased()
        currentTranscription = transcription
        
        // Check for activation phrases
        if !voiceActivated {
            for phrase in VoiceConfig.activationPhrases {
                if transcription.contains(phrase) {
                    voiceActivated = true
                    speak("How can I help you?", priority: .high)
                    logger.info("Voice activated with phrase: \(phrase)")
                    break
                }
            }
        } else {
            // Process command if voice is activated
            if result.isFinal {
                processVoiceCommand(transcription)
                voiceActivated = false
            }
        }
    }
    
    private func handleVoiceCommand(_ command: String) async {
        // Choose processing method based on backend availability
        if useBackendProcessing && voiceAIBridge.isConnected {
            // Use backend processing
            agentStatus = .analyzing
            let success = await voiceAIBridge.sendVoiceCommand(command)
            
            if !success {
                logger.warning("Backend processing failed, falling back to local processing")
                await handleLocalVoiceCommand(command)
            }
            // Backend will update status via bridge
        } else {
            // Use local processing
            await handleLocalVoiceCommand(command)
        }
    }
    
    private func handleLocalVoiceCommand(_ command: String) async {
        agentStatus = .analyzing
        
        // Determine appropriate agent and task
        let taskAnalysis = await taskPlanner.analyzeTask(command)
        let selectedAgent = await agentOrchestrator.selectOptimalAgent(for: taskAnalysis)
        
        agentStatus = .executing
        currentTask = taskAnalysis.taskDescription
        
        do {
            let response = try await selectedAgent.execute(task: taskAnalysis)
            
            DispatchQueue.main.async {
                self.lastResponse = response.content
                self.speak(response.spokenSummary)
                self.agentStatus = .completed
                self.isProcessing = false
            }
            
            logger.info("Task completed successfully: \(command)")
            
        } catch {
            DispatchQueue.main.async {
                self.speak("I encountered an error processing your request. Please try again.")
                self.agentStatus = .error
                self.isProcessing = false
            }
            
            logger.error("Task execution failed: \(error)")
        }
    }
    
    private func mapBackendStatusToAgentStatus(_ backendStatus: BackendVoiceStatus) -> AgentStatus {
        switch backendStatus {
        case .idle:
            return .idle
        case .listening:
            return .listening
        case .analyzing:
            return .analyzing
        case .executing:
            return .executing
        case .completed:
            return .completed
        case .error:
            return .error
        }
    }
    
    /// Toggle between backend and local processing
    func toggleProcessingMode() {
        useBackendProcessing.toggle()
        logger.info("Processing mode changed to: \(self.useBackendProcessing ? "backend" : "local")")
    }
    
    /// Connect to backend voice processing
    func connectToBackend() {
        voiceAIBridge.connect()
    }
    
    /// Disconnect from backend voice processing
    func disconnectFromBackend() {
        voiceAIBridge.disconnect()
    }
    
    /// Start backend voice listening
    func startBackendVoiceListening() async -> Bool {
        return await voiceAIBridge.startVoiceListening()
    }
    
    /// Stop backend voice listening
    func stopBackendVoiceListening() async -> Bool {
        return await voiceAIBridge.stopVoiceListening()
    }
}

// MARK: - Speech Synthesizer Delegate
extension VoiceAICore: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        Task { @MainActor in
            isSpeaking = true
        }
    }
    
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            isSpeaking = false
        }
    }
    
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            isSpeaking = false
        }
    }
}

// MARK: - Speech Recognizer Delegate
extension VoiceAICore: SFSpeechRecognizerDelegate {
    nonisolated func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        logger.info("Speech recognizer availability changed: \(available)")
    }
}

// MARK: - Supporting Types

enum AgentStatus {
    case idle
    case listening
    case analyzing
    case executing
    case completed
    case error
    
    var displayText: String {
        switch self {
        case .idle: return "Ready"
        case .listening: return "Listening..."
        case .analyzing: return "Analyzing request..."
        case .executing: return "Executing task..."
        case .completed: return "Task completed"
        case .error: return "Error occurred"
        }
    }
}

enum SpeechPriority {
    case low, normal, high
}

// MARK: - Agent Orchestrator

class AgentOrchestrator {
    private let webBrowsingAgent = WebBrowsingAgent()
    private let codingAgent = CodingAgent()
    private let taskPlanningAgent = TaskPlanningAgent()
    private let generalAgent = GeneralAgent()
    
    func selectOptimalAgent(for task: TaskAnalysis) async -> any AIAgent {
        switch task.taskType {
        case .webBrowsing:
            return webBrowsingAgent
        case .codeGeneration, .codeDebugging:
            return codingAgent
        case .taskPlanning, .projectManagement:
            return taskPlanningAgent
        case .general, .conversation:
            return generalAgent
        }
    }
}

// MARK: - Task Planner

class TaskPlanner {
    func analyzeTask(_ command: String) async -> TaskAnalysis {
        // AI-powered task analysis
        let lowercaseCommand = command.lowercased()
        
        let taskType: TaskType
        if lowercaseCommand.contains("browse") || lowercaseCommand.contains("search") || lowercaseCommand.contains("website") {
            taskType = .webBrowsing
        } else if lowercaseCommand.contains("code") || lowercaseCommand.contains("program") || lowercaseCommand.contains("debug") {
            taskType = .codeGeneration
        } else if lowercaseCommand.contains("plan") || lowercaseCommand.contains("organize") || lowercaseCommand.contains("schedule") {
            taskType = .taskPlanning
        } else {
            taskType = .general
        }
        
        return TaskAnalysis(
            taskType: taskType,
            taskDescription: command,
            complexity: .medium,
            estimatedDuration: 30,
            requiredCapabilities: [taskType.rawValue]
        )
    }
}

// MARK: - Task Types

enum TaskType: String, CaseIterable {
    case webBrowsing = "web_browsing"
    case codeGeneration = "code_generation"
    case codeDebugging = "code_debugging"
    case taskPlanning = "task_planning"
    case projectManagement = "project_management"
    case general = "general"
    case conversation = "conversation"
}

enum TaskComplexity {
    case simple, medium, complex
}

struct TaskAnalysis {
    let taskType: TaskType
    let taskDescription: String
    let complexity: TaskComplexity
    let estimatedDuration: Int // seconds
    let requiredCapabilities: [String]
}

struct TaskResponse {
    let content: String
    let spokenSummary: String
    let metadata: [String: Any]
}

// MARK: - AI Agent Protocol

protocol AIAgent {
    func execute(task: TaskAnalysis) async throws -> TaskResponse
}

// MARK: - Concrete Agents (Placeholder implementations)

class WebBrowsingAgent: AIAgent {
    func execute(task: TaskAnalysis) async throws -> TaskResponse {
        // TODO: Implement autonomous web browsing
        return TaskResponse(
            content: "Web browsing task completed",
            spokenSummary: "I've completed the web browsing task",
            metadata: [:]
        )
    }
}

class CodingAgent: AIAgent {
    func execute(task: TaskAnalysis) async throws -> TaskResponse {
        // TODO: Implement coding assistant
        return TaskResponse(
            content: "Code generation completed",
            spokenSummary: "I've generated the code you requested",
            metadata: [:]
        )
    }
}

class TaskPlanningAgent: AIAgent {
    func execute(task: TaskAnalysis) async throws -> TaskResponse {
        // TODO: Implement task planning
        return TaskResponse(
            content: "Task plan created",
            spokenSummary: "I've created a plan for your task",
            metadata: [:]
        )
    }
}

class GeneralAgent: AIAgent {
    func execute(task: TaskAnalysis) async throws -> TaskResponse {
        // TODO: Implement general AI assistant
        return TaskResponse(
            content: "General task completed",
            spokenSummary: "I've completed your request",
            metadata: [:]
        )
    }
}