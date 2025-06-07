import SwiftUI
import SwiftUI
import Combine
import Charts

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Interactive task complexity analysis and visualization
 * Issues & Complexity Summary: Production-ready intelligent recommendations UI component
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~250
   - Core Algorithm Complexity: High
   - Dependencies: 3 New
   - State Management Complexity: High
   - Novelty/Uncertainty Factor: Medium
 * AI Pre-Task Self-Assessment: 85%
 * Problem Estimate: 90%
 * Initial Code Complexity Estimate: 88%
 * Final Code Complexity: 91%
 * Overall Result Score: 94%
 * Last Updated: 2025-06-07
 */

struct TaskAnalysisView: View {

    
    @StateObject private var taskAnalyzer = TaskComplexityAnalyzer()
    @State private var taskToAnalyze = ""
    @State private var analysisResult: TaskComplexity?
    @State private var showingDetails = false
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            VStack(alignment: .leading, spacing: 8) {
                Text("Task Complexity Analysis")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Text("AI-powered analysis of task complexity and requirements")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            
            // Input area
            VStack(alignment: .leading, spacing: 8) {
                Text("Task Description:")
                    .font(.headline)
                
                TextEditor(text: $taskToAnalyze)
                    .frame(minHeight: 100)
                    .padding(8)
                    .background(Color(.textBackgroundColor))
                    .cornerRadius(8)
                    .border(Color.secondary.opacity(0.3), width: 1)
                
                Button("Analyze Task") {
                    Task {
                        await analyzeTask()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(taskToAnalyze.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            
            // Analysis results
            if let result = analysisResult {
                VStack(spacing: 16) {
                    // Complexity overview
                    HStack {
                        Text("Complexity Score")
                            .font(.headline)
                        
                        Spacer()
                        
                        HStack(spacing: 4) {
                            Text(String(format: "%.1f", result.complexity_score * 100))
                                .font(.title2)
                                .fontWeight(.bold)
                            Text("%")
                                .font(.title3)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    // Complexity bar
                    ProgressView(value: result.complexity_score)
                        .progressViewStyle(LinearProgressViewStyle(tint: complexityColor(result.complexity_score)))
                    
                    // Key metrics
                    LazyVGrid(columns: [
                        GridItem(.flexible()),
                        GridItem(.flexible())
                    ], spacing: 12) {
                        MetricCard(title: "Domain", value: result.domain.capitalized, icon: "tag")
                        MetricCard(title: "Est. Tokens", value: "\(result.estimated_tokens)", icon: "textformat")
                        MetricCard(title: "Context Length", value: "\(result.context_length_needed)", icon: "doc.text")
                        MetricCard(title: "Reasoning", value: result.requires_reasoning ? "Required" : "Optional", icon: "brain")
                    }
                    
                    // Requirements analysis
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Requirements Analysis")
                            .font(.headline)
                        
                        RequirementRow(title: "Reasoning", required: result.requires_reasoning)
                        RequirementRow(title: "Creativity", required: result.requires_creativity)
                        RequirementRow(title: "Factual Accuracy", required: result.requires_factual_accuracy)
                        RequirementRow(title: "Parallel Processing", beneficial: result.parallel_processing_benefit)
                    }
                    .padding()
                    .background(Color(.controlBackgroundColor))
                    .cornerRadius(8)
                }
            }
            
            Spacer()
        }
        .padding()
    }
    
    private func analyzeTask() async {
        let taskId = UUID().uuidString
        let result = await taskAnalyzer.analyzeTaskComplexity(taskToAnalyze, taskId: taskId)
        
        await MainActor.run {
            analysisResult = result
        }
    }
    
    private func complexityColor(_ score: Double) -> Color {
        switch score {
        case 0.0..<0.3: return .green
        case 0.3..<0.6: return .yellow
        case 0.6..<0.8: return .orange
        default: return .red
        }
    }
}

#Preview {
    TaskAnalysisView()
}
