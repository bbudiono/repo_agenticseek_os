import Foundation
import SwiftUI

// SANDBOX FILE: For testing/development. See .cursorrules.
/**
 * Purpose: Coordinates agents within tier constraints
 * Issues & Complexity Summary: Core tiered architecture management system
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~250
   - Core Algorithm Complexity: High
   - Dependencies: 2
   - State Management Complexity: High
   - Novelty/Uncertainty Factor: Medium
 * AI Pre-Task Self-Assessment: 85%
 * Problem Estimate: 82%
 * Initial Code Complexity Estimate: 85%
 * Final Code Complexity: 87%
 * Overall Result Score: 91%
 * Key Variances/Learnings: Tier management requires careful state synchronization
 * Last Updated: 2025-06-07
 */

enum TierLevel: String, CaseIterable {
    case free = "Free"
    case premium = "Premium"
    case enterprise = "Enterprise"
    
    var maxAgents: Int {
        switch self {
        case .free: return 3
        case .premium: return 5
        case .enterprise: return 10
        }
    }
    
    var features: [String] {
        switch self {
        case .free:
            return ["Basic chat functionality", "3 agents maximum", "Local models only"]
        case .premium:
            return ["Advanced features", "5 agents maximum", "Cloud models", "Priority support"]
        case .enterprise:
            return ["All features", "10 agents maximum", "Custom models", "Dedicated support", "Analytics"]
        }
    }
    
    var color: Color {
        switch self {
        case .free: return .gray
        case .premium: return .blue
        case .enterprise: return .purple
        }
    }
    
    var icon: String {
        switch self {
        case .free: return "person.circle"
        case .premium: return "person.2.circle"
        case .enterprise: return "person.3.circle"
        }
    }
}

class TieredAgentCoordinator: ObservableObject {
    static let shared = TieredAgentCoordinator()
    
    @Published var currentTier: TierLevel = .free
    @Published var activeAgentCount: Int = 0
    @Published var isInitialized: Bool = false
    
    private init() {
        loadTierConfiguration()
    }
    
    // MARK: - Public API
    
    func canCreateAgent() -> Bool {
        return activeAgentCount < currentTier.maxAgents
    }
    
    func createAgent() -> Bool {
        guard canCreateAgent() else {
            return false
        }
        
        activeAgentCount += 1
        notifyAgentCreated()
        return true
    }
    
    func removeAgent() {
        guard activeAgentCount > 0 else { return }
        
        activeAgentCount -= 1
        notifyAgentRemoved()
    }
    
    func upgradeTier(to newTier: TierLevel) -> Bool {
        guard newTier.maxAgents > currentTier.maxAgents else {
            return false
        }
        
        currentTier = newTier
        saveTierConfiguration()
        notifyTierUpgraded()
        return true
    }
    
    func refreshTierStatus() {
        // Simulate tier status refresh
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.isInitialized = true
            self.notifyTierRefreshed()
        }
    }
    
    // MARK: - Private Methods
    
    private func loadTierConfiguration() {
        // Load from UserDefaults or remote config
        let savedTier = UserDefaults.standard.string(forKey: "currentTier") ?? "free"
        currentTier = TierLevel(rawValue: savedTier) ?? .free
        activeAgentCount = UserDefaults.standard.integer(forKey: "activeAgentCount")
        isInitialized = true
    }
    
    private func saveTierConfiguration() {
        UserDefaults.standard.set(currentTier.rawValue, forKey: "currentTier")
        UserDefaults.standard.set(activeAgentCount, forKey: "activeAgentCount")
    }
    
    private func notifyAgentCreated() {
        NotificationCenter.default.post(name: .agentCreated, object: nil)
        saveTierConfiguration()
    }
    
    private func notifyAgentRemoved() {
        NotificationCenter.default.post(name: .agentRemoved, object: nil)
        saveTierConfiguration()
    }
    
    private func notifyTierUpgraded() {
        NotificationCenter.default.post(name: .tierUpgraded, object: currentTier)
        saveTierConfiguration()
    }
    
    private func notifyTierRefreshed() {
        NotificationCenter.default.post(name: .tierRefreshed, object: nil)
    }
}

// MARK: - Notification Extensions

extension Notification.Name {
    static let agentCreated = Notification.Name("agentCreated")
    static let agentRemoved = Notification.Name("agentRemoved")
    static let tierUpgraded = Notification.Name("tierUpgraded")
    static let tierRefreshed = Notification.Name("tierRefreshed")
}
