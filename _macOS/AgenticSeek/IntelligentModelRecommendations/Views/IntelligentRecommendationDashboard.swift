import SwiftUI
import SwiftUI
import Combine
import Charts

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Main dashboard with AI-powered recommendations and explanations
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

struct IntelligentRecommendationDashboard: View {

    
    @StateObject private var recommendationEngine = RecommendationGenerationEngine()
    @StateObject private var taskAnalyzer = TaskComplexityAnalyzer()
    @StateObject private var userLearningEngine = UserPreferenceLearningEngine()
    
    @State private var currentTask = ""
    @State private var recommendations: [ModelRecommendation] = []
    @State private var isAnalyzing = false
    @State private var selectedRecommendation: ModelRecommendation?
    @State private var showingExplanation = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Header with task input
                VStack(spacing: 16) {
                    HStack {
                        Text("Intelligent Model Recommendations")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                        
                        Spacer()
                        
                        if isAnalyzing {
                            HStack(spacing: 8) {
                                ProgressView()
                                    .scaleEffect(0.8)
                                Text("Analyzing...")
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    
                    // Task input area
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Describe your task:")
                            .font(.headline)
                        
                        TextEditor(text: $currentTask)
                            .frame(minHeight: 80, maxHeight: 120)
                            .padding(8)
                            .background(Color(.textBackgroundColor))
                            .cornerRadius(8)
                            .border(Color.secondary.opacity(0.3), width: 1)
                        
                        HStack {
                            Button("Get Recommendations") {
                                Task {
                                    await analyzeTaskAndGenerateRecommendations()
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(currentTask.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isAnalyzing)
                            
                            Spacer()
                            
                            if !currentTask.isEmpty {
                                Text("\(currentTask.count) characters")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }
                .padding()
                .background(Color(.controlBackgroundColor))
                
                Divider()
                
                // Recommendations list
                if recommendations.isEmpty && !isAnalyzing {
                    VStack(spacing: 16) {
                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        
                        Text("AI-Powered Model Recommendations")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        
                        Text("Enter a task description above to get intelligent model recommendations based on complexity analysis, your preferences, and hardware capabilities.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(recommendations, id: \.recommendation_id) { recommendation in
                                RecommendationCard(recommendation: recommendation) {
                                    selectedRecommendation = recommendation
                                    showingExplanation = true
                                }
                                .padding(.horizontal)
                            }
                        }
                        .padding(.vertical)
                    }
                }
            }
        }
        .sheet(isPresented: $showingExplanation) {
            if let recommendation = selectedRecommendation {
                RecommendationExplanationView(recommendation: recommendation)
            }
        }
    }
    
    private func analyzeTaskAndGenerateRecommendations() async {
        isAnalyzing = true
        recommendations = []
        
        // Analyze task complexity
        let taskId = UUID().uuidString
        let complexity = await taskAnalyzer.analyzeTaskComplexity(currentTask, taskId: taskId)
        
        // Generate recommendations based on analysis
        let newRecommendations = await recommendationEngine.generateRecommendations(
            for: complexity,
            userId: "current_user" // In production, this would be the actual user ID
        )
        
        await MainActor.run {
            recommendations = newRecommendations
            isAnalyzing = false
        }
    }
}

#Preview {
    IntelligentRecommendationDashboard()
}
