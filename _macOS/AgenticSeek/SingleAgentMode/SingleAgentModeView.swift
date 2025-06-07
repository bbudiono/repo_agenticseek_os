
import SwiftUI

enum AgentMode {
    case single
    case multi
}

struct PerformanceData {
    let cpuUsage: Double?
    let memoryUsage: Double?
    let responseTime: Double?
}

struct SingleAgentModeView: View {
    @State private var currentMode: AgentMode = .single
    @State private var selectedModel: LocalModelInfo?
    @State private var availableModels: [LocalModelInfo] = []
    @State private var performanceData = PerformanceData(cpuUsage: nil, memoryUsage: nil, responseTime: nil)
    
    var modeToggle: some View {
        Picker("Mode", selection: $currentMode) {
            Text("Single Agent").tag(AgentMode.single)
            Text("Multi Agent").tag(AgentMode.multi)
        }
        .pickerStyle(SegmentedPickerStyle())
    }
    
    var modelSelector: some View {
        VStack(alignment: .leading) {
            Text("Available Models")
                .font(.headline)
            
            if availableModels.isEmpty {
                Text("No models found. Please install Ollama or LM Studio.")
                    .foregroundColor(.secondary)
            } else {
                Picker("Model", selection: $selectedModel) {
                    ForEach(availableModels, id: \.name) { model in
                        Text("\(model.name) (\(model.parameters))")
                            .tag(model as LocalModelInfo?)
                    }
                }
            }
        }
    }
    
    var performanceMonitor: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Performance Monitor")
                .font(.headline)
            
            HStack {
                Text("CPU Usage:")
                Spacer()
                Text("\(performanceData.cpuUsage?.formatted(.percent) ?? "N/A")")
            }
            
            HStack {
                Text("Memory Usage:")
                Spacer()
                Text("\(performanceData.memoryUsage?.formatted(.percent) ?? "N/A")")
            }
            
            HStack {
                Text("Response Time:")
                Spacer()
                Text("\(performanceData.responseTime?.formatted() ?? "N/A")s")
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
    
    var body: some View {
        VStack(spacing: 20) {
            modeToggle
            modelSelector
            performanceMonitor
            Spacer()
        }
        .padding()
        .onAppear {
            loadAvailableModels()
            startPerformanceMonitoring()
        }
    }
    
    func getCurrentMode() -> AgentMode {
        return currentMode
    }
    
    func toggleMode() {
        currentMode = currentMode == .single ? .multi : .single
    }
    
    func getAvailableModels() -> [LocalModelInfo] {
        return availableModels
    }
    
    func selectModel(_ model: LocalModelInfo) {
        selectedModel = model
    }
    
    func getSelectedModel() -> LocalModelInfo? {
        return selectedModel
    }
    
    func getPerformanceData() -> PerformanceData {
        return performanceData
    }
    
    private func loadAvailableModels() {
        let detector = OllamaDetector()
        availableModels = detector.discoverModels()
        
        if let firstModel = availableModels.first {
            selectedModel = firstModel
        }
    }
    
    private func startPerformanceMonitoring() {
        // Simulate performance monitoring
        Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
            performanceData = PerformanceData(
                cpuUsage: Double.random(in: 0.1...0.8),
                memoryUsage: Double.random(in: 0.2...0.6),
                responseTime: Double.random(in: 1.5...5.0)
            )
        }
    }
}
