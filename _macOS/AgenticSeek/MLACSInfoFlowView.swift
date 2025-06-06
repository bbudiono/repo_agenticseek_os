//
// * Purpose: MLACS Information Flow Visualization Components
// * Issues & Complexity Summary: Real-time information flow display with message routing visualization
// * Key Complexity Drivers:
//   - Logic Scope (Est. LoC): ~250
//   - Core Algorithm Complexity: Medium
//   - Dependencies: 2 (SwiftUI, MLACSInfoDissemination)
//   - State Management Complexity: Medium
//   - Novelty/Uncertainty Factor: Low
// * AI Pre-Task Self-Assessment: 88%
// * Problem Estimate: 82%
// * Initial Code Complexity Estimate: 80%
// * Final Code Complexity: 83%
// * Overall Result Score: 96%
// * Key Variances/Learnings: Information flow visualization enhances user understanding
// * Last Updated: 2025-06-07

import SwiftUI

// MARK: - Information Flow View
struct MLACSInfoFlowView: View {
    let infoDisseminationManager: MLACSInfoDisseminationManager
    @State private var showingFlowDetails = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Information Flow")
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Spacer()
                
                Button(action: { showingFlowDetails.toggle() }) {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .foregroundColor(.blue)
                }
                .accessibilityLabel("Show information flow details")
            }
            
            // Recent Messages Flow
            InfoFlowDiagram(messages: Array(infoDisseminationManager.informationFlowLog.suffix(5)))
            
            // Flow Metrics Summary
            let metrics = infoDisseminationManager.getInformationFlowMetrics()
            InfoFlowMetricsView(metrics: metrics)
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
        .sheet(isPresented: $showingFlowDetails) {
            InfoFlowDetailsView(infoDisseminationManager: infoDisseminationManager)
        }
    }
}

// MARK: - Information Flow Diagram
struct InfoFlowDiagram: View {
    let messages: [MLACSMessage]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent Information Flow")
                .font(.subheadline)
                .fontWeight(.medium)
            
            if messages.isEmpty {
                Text("No information flow yet")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding()
            } else {
                LazyVStack(spacing: 4) {
                    ForEach(messages, id: \.id) { message in
                        MessageFlowView(message: message)
                    }
                }
            }
        }
    }
}

// MARK: - Message Flow View
struct MessageFlowView: View {
    let message: MLACSMessage
    
    private var messageTypeColor: Color {
        switch message.type {
        case .userRequest: return .blue
        case .agentResponse: return .green
        case .knowledgeShare: return .purple
        case .contextUpdate: return .orange
        case .coordinatorDirective: return .red
        case .statusUpdate: return .gray
        }
    }
    
    private var messageTypeIcon: String {
        switch message.type {
        case .userRequest: return "person.circle"
        case .agentResponse: return "bubble.left"
        case .knowledgeShare: return "brain.head.profile"
        case .contextUpdate: return "arrow.triangle.2.circlepath"
        case .coordinatorDirective: return "command"
        case .statusUpdate: return "info.circle"
        }
    }
    
    var body: some View {
        HStack(spacing: 8) {
            // Message type indicator
            Image(systemName: messageTypeIcon)
                .foregroundColor(messageTypeColor)
                .frame(width: 20)
            
            // Source agent
            Text(message.sourceAgent.displayName)
                .font(.caption)
                .fontWeight(.medium)
                .frame(width: 80, alignment: .leading)
            
            // Flow arrow
            Image(systemName: "arrow.right")
                .foregroundColor(.secondary)
                .font(.caption2)
            
            // Target agent or "All"
            Text(message.targetAgent?.displayName ?? "All")
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(width: 60, alignment: .leading)
            
            // Status indicator
            Circle()
                .fill(message.isProcessed ? Color.green : Color.orange)
                .frame(width: 6, height: 6)
            
            Spacer()
            
            // Timestamp
            Text(message.timestamp, style: .time)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Message from \(message.sourceAgent.displayName)")
        .accessibilityValue("Type: \(message.type.rawValue), Status: \(message.isProcessed ? "processed" : "pending")")
    }
}

// MARK: - Information Flow Metrics View
struct InfoFlowMetricsView: View {
    let metrics: InfoFlowMetrics
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Flow Statistics")
                .font(.subheadline)
                .fontWeight(.medium)
            
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Messages")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Text("\(metrics.totalMessages)")
                        .font(.caption)
                        .fontWeight(.medium)
                }
                
                Spacer()
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("Knowledge")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Text("\(metrics.knowledgeBaseSize)")
                        .font(.caption)
                        .fontWeight(.medium)
                }
                
                Spacer()
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("Routes")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Text("\(metrics.activeRoutes)")
                        .font(.caption)
                        .fontWeight(.medium)
                }
                
                Spacer()
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("Avg Time")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Text("\(String(format: "%.1f", metrics.averageResponseTime))s")
                        .font(.caption)
                        .fontWeight(.medium)
                }
            }
        }
    }
}

// MARK: - Information Flow Details View
struct InfoFlowDetailsView: View {
    let infoDisseminationManager: MLACSInfoDisseminationManager
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            List {
                Section("Message Queue") {
                    ForEach(infoDisseminationManager.messageQueue, id: \.id) { message in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(message.type.rawValue.capitalized)
                                    .font(.caption)
                                    .fontWeight(.medium)
                                Spacer()
                                Text(message.timestamp, style: .time)
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                            Text(message.content)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .lineLimit(2)
                        }
                    }
                }
                
                Section("Knowledge Base") {
                    ForEach(infoDisseminationManager.knowledgeBase.prefix(10), id: \.id) { entry in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(entry.topic)
                                    .font(.caption)
                                    .fontWeight(.medium)
                                Spacer()
                                Text("\(Int(entry.confidence * 100))%")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                            Text(entry.content)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .lineLimit(2)
                        }
                    }
                }
            }
            .navigationTitle("Information Flow Details")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
        .frame(width: 600, height: 500)
    }
}
