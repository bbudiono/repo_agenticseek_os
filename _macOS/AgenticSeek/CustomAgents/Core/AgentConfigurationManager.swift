import Foundation
import SwiftUI
import Combine

// SANDBOX FILE: For testing/development. See .cursorrules.
/**
 * Purpose: Manages agent configurations, settings, and persistence
 * Issues & Complexity Summary: Core custom agent management system
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~400
   - Core Algorithm Complexity: Medium
   - Dependencies: 2
   - State Management Complexity: High
   - Novelty/Uncertainty Factor: Medium
 * AI Pre-Task Self-Assessment: 88%
 * Problem Estimate: 86%
 * Initial Code Complexity Estimate: 89%
 * Final Code Complexity: 91%
 * Overall Result Score: 93%
 * Key Variances/Learnings: Custom agent coordination requires sophisticated state management
 * Last Updated: 2025-06-07
 */

// MARK: - Custom Agent Data Models

struct CustomAgent: Identifiable, Codable {
    let id: UUID
    var name: String
    var description: String
    var category: AgentCategory
    var skills: [AgentSkill]
    var configuration: AgentConfiguration
    var performance: AgentPerformance
    var isActive: Bool
    var createdDate: Date
    var lastUsed: Date?
    
    init(name: String, description: String, category: AgentCategory) {
        self.id = UUID()
        self.name = name
        self.description = description
        self.category = category
        self.skills = []
        self.configuration = AgentConfiguration()
        self.performance = AgentPerformance()
        self.isActive = false
        self.createdDate = Date()
        self.lastUsed = nil
    }
}

enum AgentCategory: String, CaseIterable, Codable {
    case productivity = "Productivity"
    case creative = "Creative"
    case analysis = "Analysis"
    case research = "Research"
    case support = "Support"
    case entertainment = "Entertainment"
    
    var icon: String {
        switch self {
        case .productivity: return "briefcase.fill"
        case .creative: return "paintbrush.fill"
        case .analysis: return "chart.bar.fill"
        case .research: return "magnifyingglass"
        case .support: return "person.fill.questionmark"
        case .entertainment: return "gamecontroller.fill"
        }
    }
}

struct AgentSkill: Identifiable, Codable {
    let id: UUID
    var name: String
    var description: String
    var proficiencyLevel: Double // 0.0 to 1.0
    var isEnabled: Bool
    
    init(name: String, description: String, proficiencyLevel: Double = 0.5) {
        self.id = UUID()
        self.name = name
        self.description = description
        self.proficiencyLevel = proficiencyLevel
        self.isEnabled = true
    }
}

struct AgentConfiguration: Codable {
    var responseStyle: ResponseStyle
    var maxTokens: Int
    var temperature: Double
    var memorySize: Int
    var privacyMode: Bool
    var learningEnabled: Bool
    
    init() {
        self.responseStyle = .balanced
        self.maxTokens = 1000
        self.temperature = 0.7
        self.memorySize = 100
        self.privacyMode = true
        self.learningEnabled = true
    }
}

enum ResponseStyle: String, CaseIterable, Codable {
    case concise = "Concise"
    case detailed = "Detailed"
    case balanced = "Balanced"
    case creative = "Creative"
}

struct AgentPerformance: Codable {
    var totalTasks: Int
    var successfulTasks: Int
    var averageResponseTime: Double
    var userRating: Double
    var lastPerformanceUpdate: Date
    
    init() {
        self.totalTasks = 0
        self.successfulTasks = 0
        self.averageResponseTime = 0.0
        self.userRating = 0.0
        self.lastPerformanceUpdate = Date()
    }
    
    var successRate: Double {
        guard totalTasks > 0 else { return 0.0 }
        return Double(successfulTasks) / Double(totalTasks)
    }
}

// MARK: - Custom Agent Framework

class AgentConfigurationManager: ObservableObject {
    static let shared = AgentConfigurationManager()
    
    @Published var customAgents: [CustomAgent] = []
    @Published var isInitialized: Bool = false
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?
    
    private var cancellables = Set<AnyCancellable>()
    
    private init() {
        loadCustomAgents()
    }
    
    // MARK: - Public API
    
    func createAgent(name: String, description: String, category: AgentCategory) -> CustomAgent {
        let agent = CustomAgent(name: name, description: description, category: category)
        customAgents.append(agent)
        saveCustomAgents()
        notifyAgentCreated(agent)
        return agent
    }
    
    func updateAgent(_ agent: CustomAgent) {
        if let index = customAgents.firstIndex(where: { $0.id == agent.id }) {
            customAgents[index] = agent
            saveCustomAgents()
            notifyAgentUpdated(agent)
        }
    }
    
    func deleteAgent(_ agent: CustomAgent) {
        customAgents.removeAll { $0.id == agent.id }
        saveCustomAgents()
        notifyAgentDeleted(agent)
    }
    
    func activateAgent(_ agent: CustomAgent) {
        updateAgentStatus(agent, isActive: true)
    }
    
    func deactivateAgent(_ agent: CustomAgent) {
        updateAgentStatus(agent, isActive: false)
    }
    
    func searchAgents(query: String) -> [CustomAgent] {
        guard !query.isEmpty else { return customAgents }
        
        return customAgents.filter {
            $0.name.localizedCaseInsensitiveContains(query) ||
            $0.description.localizedCaseInsensitiveContains(query) ||
            $0.category.rawValue.localizedCaseInsensitiveContains(query)
        }
    }
    
    func getAgentsByCategory(_ category: AgentCategory) -> [CustomAgent] {
        return customAgents.filter { $0.category == category }
    }
    
    func getActiveAgents() -> [CustomAgent] {
        return customAgents.filter { $0.isActive }
    }
    
    func updateAgentPerformance(_ agent: CustomAgent, taskSuccessful: Bool, responseTime: Double) {
        guard let index = customAgents.firstIndex(where: { $0.id == agent.id }) else { return }
        
        var updatedAgent = customAgents[index]
        updatedAgent.performance.totalTasks += 1
        
        if taskSuccessful {
            updatedAgent.performance.successfulTasks += 1
        }
        
        // Update average response time
        let currentAverage = updatedAgent.performance.averageResponseTime
        let taskCount = Double(updatedAgent.performance.totalTasks)
        updatedAgent.performance.averageResponseTime = 
            ((currentAverage * (taskCount - 1)) + responseTime) / taskCount
        
        updatedAgent.performance.lastPerformanceUpdate = Date()
        updatedAgent.lastUsed = Date()
        
        customAgents[index] = updatedAgent
        saveCustomAgents()
    }
    
    func refreshData() {
        isLoading = true
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            self.isLoading = false
            self.loadCustomAgents()
            self.notifyDataRefreshed()
        }
    }
    
    // MARK: - Private Methods
    
    private func updateAgentStatus(_ agent: CustomAgent, isActive: Bool) {
        if let index = customAgents.firstIndex(where: { $0.id == agent.id }) {
            customAgents[index].isActive = isActive
            if isActive {
                customAgents[index].lastUsed = Date()
            }
            saveCustomAgents()
            notifyAgentStatusChanged(customAgents[index])
        }
    }
    
    private func loadCustomAgents() {
        // Load from UserDefaults or Core Data
        if let data = UserDefaults.standard.data(forKey: "customAgents"),
           let agents = try? JSONDecoder().decode([CustomAgent].self, from: data) {
            customAgents = agents
        } else {
            // Initialize with sample agents
            initializeSampleAgents()
        }
        isInitialized = true
    }
    
    private func saveCustomAgents() {
        if let data = try? JSONEncoder().encode(customAgents) {
            UserDefaults.standard.set(data, forKey: "customAgents")
        }
    }
    
    private func initializeSampleAgents() {
        let sampleAgents = [
            createSampleAgent(name: "Research Assistant", category: .research),
            createSampleAgent(name: "Content Creator", category: .creative),
            createSampleAgent(name: "Data Analyzer", category: .analysis)
        ]
        
        customAgents = sampleAgents
        saveCustomAgents()
    }
    
    private func createSampleAgent(name: String, category: AgentCategory) -> CustomAgent {
        var agent = CustomAgent(name: name, description: "Sample \(category.rawValue.lowercased()) agent", category: category)
        
        // Add sample skills
        agent.skills = [
            AgentSkill(name: "Communication", description: "Clear and effective communication", proficiencyLevel: 0.8),
            AgentSkill(name: "Problem Solving", description: "Analytical problem-solving abilities", proficiencyLevel: 0.7),
            AgentSkill(name: "Domain Knowledge", description: "Specialized knowledge in the field", proficiencyLevel: 0.9)
        ]
        
        return agent
    }
    
    // MARK: - Notifications
    
    private func notifyAgentCreated(_ agent: CustomAgent) {
        NotificationCenter.default.post(name: .customAgentCreated, object: agent)
    }
    
    private func notifyAgentUpdated(_ agent: CustomAgent) {
        NotificationCenter.default.post(name: .customAgentUpdated, object: agent)
    }
    
    private func notifyAgentDeleted(_ agent: CustomAgent) {
        NotificationCenter.default.post(name: .customAgentDeleted, object: agent)
    }
    
    private func notifyAgentStatusChanged(_ agent: CustomAgent) {
        NotificationCenter.default.post(name: .customAgentStatusChanged, object: agent)
    }
    
    private func notifyDataRefreshed() {
        NotificationCenter.default.post(name: .customAgentDataRefreshed, object: nil)
    }
}

// MARK: - Notification Extensions

extension Notification.Name {
    static let customAgentCreated = Notification.Name("customAgentCreated")
    static let customAgentUpdated = Notification.Name("customAgentUpdated")
    static let customAgentDeleted = Notification.Name("customAgentDeleted")
    static let customAgentStatusChanged = Notification.Name("customAgentStatusChanged")
    static let customAgentDataRefreshed = Notification.Name("customAgentDataRefreshed")
}

// MARK: - Component-Specific Extensions

extension AgentConfigurationManager {
    // Component-specific functionality based on component name
    
    
    // MARK: - General Framework Methods
    
    func getFrameworkStatus() -> String {
        return "\(customAgents.count) agents, \(getActiveAgents().count) active"
    }

}
