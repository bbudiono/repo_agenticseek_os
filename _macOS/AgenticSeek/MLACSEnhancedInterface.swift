//
// * Purpose: Enhanced chat interface with MLACS agent visibility and real-time delegation display
// * Issues & Complexity Summary: Real-time agent status display with task delegation visualization
// * Key Complexity Drivers:
//   - Logic Scope (Est. LoC): ~200
//   - Core Algorithm Complexity: Medium
//   - Dependencies: 2 (SwiftUI, MLACSCoordinator)
//   - State Management Complexity: Medium
//   - Novelty/Uncertainty Factor: Low
// * AI Pre-Task Self-Assessment: 85%
// * Problem Estimate: 75%
// * Initial Code Complexity Estimate: 75%
// * Final Code Complexity: 78%
// * Overall Result Score: 94%
// * Key Variances/Learnings: Real-time agent visualization enhances user understanding
// * Last Updated: 2025-06-07

import SwiftUI

// MARK: - Enhanced Agent Status View
struct MLACSAgentStatusView: View {
    let coordinator: MLACSCoordinator
    @State private var showingAgentDetails = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("MLACS Agents")
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Spacer()
                
                Button(action: { showingAgentDetails.toggle() }) {
                    Image(systemName: "info.circle")
                        .foregroundColor(.blue)
                }
                .accessibilityLabel("Show agent details")
                .accessibilityHint("View detailed information about MLACS agents")
            }
            
            // Real-time Agent Status Grid
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 8) {
                ForEach(MLACSAgentType.allCases, id: \.self) { agentType in
                    AgentStatusCard(
                        agentType: agentType,
                        status: coordinator.getAgentStatus(agentType),
                        isActive: coordinator.activeAgents.contains(agentType)
                    )
                }
            }
            
            // Task Delegation Progress
            if !coordinator.currentTasks.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Active Tasks")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    ForEach(coordinator.currentTasks, id: \.id) { task in
                        TaskProgressView(task: task)
                    }
                }
                .padding(.top, 8)
            }
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
        .sheet(isPresented: $showingAgentDetails) {
            AgentDetailsView(coordinator: coordinator)
        }
    }
}

// MARK: - Agent Status Card
struct AgentStatusCard: View {
    let agentType: MLACSAgentType
    let status: String
    let isActive: Bool
    
    var statusColor: Color {
        switch status {
        case let s where s.contains("Working"):
            return .orange
        case let s where s.contains("Queued"):
            return .blue
        case "Available":
            return .green
        default:
            return .gray
        }
    }
    
    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: agentIcon)
                .font(.title2)
                .foregroundColor(isActive ? statusColor : .gray)
            
            Text(agentType.displayName)
                .font(.caption)
                .fontWeight(.medium)
                .multilineTextAlignment(.center)
            
            Text(status)
                .font(.caption2)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
        }
        .padding(8)
        .background(isActive ? Color.blue.opacity(0.1) : Color.clear)
        .cornerRadius(6)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(agentType.displayName) agent")
        .accessibilityValue("Status: \(status)")
    }
    
    private var agentIcon: String {
        switch agentType {
        case .coordinator: return "person.circle"
        case .coder: return "chevron.left.forwardslash.chevron.right"
        case .researcher: return "magnifyingglass"
        case .planner: return "list.bullet.clipboard"
        case .browser: return "globe"
        case .fileManager: return "folder"
        }
    }
}

// MARK: - Task Progress View
struct TaskProgressView: View {
    let task: MLACSTask
    
    var progressValue: Double {
        switch task.status {
        case .pending: return 0.0
        case .assigned: return 0.2
        case .inProgress: return 0.6
        case .completed: return 1.0
        case .failed: return 0.0
        }
    }
    
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: task.assignedAgent == .coordinator ? "person.circle" : "gearshape")
                .foregroundColor(.blue)
                .frame(width: 16)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(task.assignedAgent.displayName)
                    .font(.caption)
                    .fontWeight(.medium)
                
                ProgressView(value: progressValue)
                    .progressViewStyle(LinearProgressViewStyle(tint: progressValue == 1.0 ? .green : .blue))
            }
            
            Text("\(Int(progressValue * 100))%")
                .font(.caption2)
                .foregroundColor(.secondary)
                .frame(width: 30, alignment: .trailing)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Task for \(task.assignedAgent.displayName)")
        .accessibilityValue("\(Int(progressValue * 100)) percent complete")
    }
}

// MARK: - Agent Details View
struct AgentDetailsView: View {
    let coordinator: MLACSCoordinator
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            List {
                ForEach(MLACSAgentType.allCases, id: \.self) { agentType in
                    Section(agentType.displayName) {
                        ForEach(agentType.capabilities, id: \.self) { capability in
                            HStack {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                    .font(.caption)
                                
                                Text(capability)
                                    .font(.body)
                            }
                        }
                        
                        HStack {
                            Text("Status:")
                                .fontWeight(.medium)
                            Spacer()
                            Text(coordinator.getAgentStatus(agentType))
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("MLACS Agents")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
        .frame(width: 500, height: 600)
    }
}

// MARK: - Enhanced Chat Message View with Agent Attribution
struct EnhancedChatMessageView: View {
    let message: SimpleChatMessage
    let agentAttribution: String?
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
            }
            
            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                // Agent attribution for AI responses
                if !message.isUser, let attribution = agentAttribution {
                    HStack(spacing: 4) {
                        Image(systemName: "brain.head.profile")
                            .font(.caption2)
                            .foregroundColor(.blue)
                        Text(attribution)
                            .font(.caption2)
                            .foregroundColor(.blue)
                    }
                }
                
                Text(message.content)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(message.isUser ? Color.blue : Color(NSColor.controlBackgroundColor))
                    )
                    .foregroundColor(message.isUser ? .white : .primary)
                
                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            if !message.isUser {
                Spacer()
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(message.isUser ? "Your message" : "AI response")
        .accessibilityValue(message.content)
    }
}
