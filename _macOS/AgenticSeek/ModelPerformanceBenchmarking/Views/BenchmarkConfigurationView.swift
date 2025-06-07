import SwiftUI
import SwiftUI
import Combine

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Benchmark test configuration and setup
 * Issues & Complexity Summary: Production-ready benchmarking UI component
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~150
   - Core Algorithm Complexity: Medium
   - Dependencies: 2 New
   - State Management Complexity: Medium
   - Novelty/Uncertainty Factor: Low
 * AI Pre-Task Self-Assessment: 85%
 * Problem Estimate: 88%
 * Initial Code Complexity Estimate: 83%
 * Final Code Complexity: 86%
 * Overall Result Score: 92%
 * Last Updated: 2025-06-07
 */

struct BenchmarkConfigurationView: View {

    
    @StateObject private var benchmarkEngine = ModelBenchmarkEngine()
    @StateObject private var scheduler = BenchmarkScheduler()
    
    @State private var selectedModels: [LocalModel] = []
    @State private var testPrompts: [String] = ["Explain AI in simple terms"]
    @State private var newPrompt = ""
    @State private var benchmarkInterval = 3600.0 // 1 hour
    @State private var enableScheduling = false
    
    var body: some View {
        NavigationView {
            Form {
                Section("Model Selection") {
                    ForEach(availableModels, id: \.id) { model in
                        HStack {
                            Text(model.name)
                            Spacer()
                            if selectedModels.contains(where: { $0.id == model.id }) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.blue)
                            }
                        }
                        .contentShape(Rectangle())
                        .onTapGesture {
                            toggleModelSelection(model)
                        }
                    }
                }
                
                Section("Test Prompts") {
                    ForEach(Array(testPrompts.enumerated()), id: \.offset) { index, prompt in
                        HStack {
                            Text(prompt)
                            Spacer()
                            Button("Remove") {
                                testPrompts.remove(at: index)
                            }
                            .foregroundColor(.red)
                        }
                    }
                    
                    HStack {
                        TextField("Add new prompt", text: $newPrompt)
                        Button("Add") {
                            if !newPrompt.isEmpty {
                                testPrompts.append(newPrompt)
                                newPrompt = ""
                            }
                        }
                        .disabled(newPrompt.isEmpty)
                    }
                }
                
                Section("Scheduling") {
                    Toggle("Enable Scheduled Benchmarks", isOn: $enableScheduling)
                    
                    if enableScheduling {
                        HStack {
                            Text("Interval")
                            Spacer()
                            Picker("Interval", selection: $benchmarkInterval) {
                                Text("30 min").tag(1800.0)
                                Text("1 hour").tag(3600.0)
                                Text("6 hours").tag(21600.0)
                                Text("Daily").tag(86400.0)
                            }
                        }
                    }
                }
                
                Section {
                    Button("Run Benchmark Now") {
                        runBenchmark()
                    }
                    .disabled(selectedModels.isEmpty || benchmarkEngine.isRunning)
                    
                    if enableScheduling {
                        Button("Schedule Benchmarks") {
                            scheduleBenchmarks()
                        }
                        .disabled(selectedModels.isEmpty)
                    }
                }
            }
            .navigationTitle("Benchmark Configuration")
        }
    }
    
    private var availableModels: [LocalModel] {
        [
            LocalModel(id: "llama2:7b", name: "Llama 2 7B", type: "ollama"),
            LocalModel(id: "codellama:13b", name: "Code Llama 13B", type: "ollama"),
            LocalModel(id: "mistral:7b", name: "Mistral 7B", type: "lm_studio")
        ]
    }
    
    private func toggleModelSelection(_ model: LocalModel) {
        if let index = selectedModels.firstIndex(where: { $0.id == model.id }) {
            selectedModels.remove(at: index)
        } else {
            selectedModels.append(model)
        }
    }
    
    private func runBenchmark() {
        Task {
            for model in selectedModels {
                try await benchmarkEngine.runBenchmark(for: model, prompts: testPrompts)
            }
        }
    }
    
    private func scheduleBenchmarks() {
        for model in selectedModels {
            let scheduledBenchmark = ScheduledBenchmark(
                model: model,
                testPrompts: testPrompts,
                scheduledTime: Date().addingTimeInterval(benchmarkInterval)
            )
            scheduler.scheduleBenchmark(scheduledBenchmark)
        }
    }
}

// MARK: - Supporting Views

struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(.blue)
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
            }
            
            Text(value)
                .font(.title2)
                .fontWeight(.semibold)
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
}

#Preview {
    BenchmarkConfigurationView()
}
