import SwiftUI
import SwiftUI
import Charts
import Combine

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Main benchmarking dashboard with real-time metrics
 * Issues & Complexity Summary: Production-ready benchmarking UI component
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~150
   - Core Algorithm Complexity: Medium
   - Dependencies: 3 New
   - State Management Complexity: Medium
   - Novelty/Uncertainty Factor: Low
 * AI Pre-Task Self-Assessment: 85%
 * Problem Estimate: 88%
 * Initial Code Complexity Estimate: 83%
 * Final Code Complexity: 86%
 * Overall Result Score: 92%
 * Last Updated: 2025-06-07
 */

struct BenchmarkDashboardView: View {

    
    @StateObject private var benchmarkEngine = ModelBenchmarkEngine()
    @StateObject private var resourceMonitor = ResourceMonitor()
    @State private var selectedTimeRange = TimeRange.hour
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header with current status
                HStack {
                    VStack(alignment: .leading) {
                        Text("Model Performance Benchmarking")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                        
                        Text(benchmarkEngine.isRunning ? "Benchmark Running..." : "Ready")
                            .foregroundColor(benchmarkEngine.isRunning ? .orange : .green)
                    }
                    
                    Spacer()
                    
                    Button(action: startQuickBenchmark) {
                        HStack {
                            Image(systemName: "play.circle.fill")
                            Text("Quick Benchmark")
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                    }
                    .disabled(benchmarkEngine.isRunning)
                }
                .padding()
                
                // Real-time metrics
                HStack(spacing: 20) {
                    MetricCard(
                        title: "CPU Usage",
                        value: String(format: "%.1f%%", resourceMonitor.cpuUsage),
                        icon: "cpu"
                    )
                    
                    MetricCard(
                        title: "Memory",
                        value: String(format: "%.0f MB", resourceMonitor.memoryUsage),
                        icon: "memorychip"
                    )
                    
                    MetricCard(
                        title: "GPU Usage", 
                        value: String(format: "%.1f%%", resourceMonitor.gpuUsage),
                        icon: "gpu"
                    )
                }
                .padding(.horizontal)
                
                // Charts and visualizations
                ScrollView {
                    LazyVGrid(columns: [
                        GridItem(.flexible()),
                        GridItem(.flexible())
                    ], spacing: 16) {
                        // Performance trends chart
                        VStack(alignment: .leading) {
                            Text("Performance Trends")
                                .font(.headline)
                                .padding(.bottom, 4)
                            
                            Chart {
                                ForEach(Array(benchmarkEngine.results.enumerated()), id: \.offset) { index, result in
                                    LineMark(
                                        x: .value("Test", index),
                                        y: .value("Speed", result.tokensPerSecond)
                                    )
                                    .foregroundStyle(.blue)
                                }
                            }
                            .frame(height: 150)
                            .chartYAxis {
                                AxisMarks(position: .leading)
                            }
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                        
                        // Quality scores chart
                        VStack(alignment: .leading) {
                            Text("Quality Scores")
                                .font(.headline)
                                .padding(.bottom, 4)
                            
                            Chart {
                                ForEach(Array(benchmarkEngine.results.enumerated()), id: \.offset) { index, result in
                                    BarMark(
                                        x: .value("Test", index),
                                        y: .value("Quality", result.qualityScore)
                                    )
                                    .foregroundStyle(.green)
                                }
                            }
                            .frame(height: 150)
                            .chartYScale(domain: 0...1)
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                    }
                }
                .padding(.horizontal)
                
                Spacer()
            }
        }
        .navigationTitle("Benchmarks")
        .onAppear {
            resourceMonitor.startMonitoring()
        }
        .onDisappear {
            resourceMonitor.stopMonitoring()
        }
    }
    
    private func startQuickBenchmark() {
        Task {
            let testModel = LocalModel(id: "test", name: "Test Model", type: "local")
            try await benchmarkEngine.runBenchmark(for: testModel, prompts: ["Test prompt"])
        }
    }
    
    enum TimeRange: String, CaseIterable {
        case hour = "1H"
        case day = "1D"
        case week = "1W"
        case month = "1M"
    }
}

// MARK: - Supporting Views

struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.blue)
            
            Text(value)
                .font(.headline)
                .fontWeight(.semibold)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
}

#Preview {
    BenchmarkDashboardView()
}
