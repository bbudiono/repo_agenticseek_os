import Foundation
import Foundation
import Combine

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Automated benchmark scheduling and execution
 * Issues & Complexity Summary: Production-ready benchmarking component
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~120
   - Core Algorithm Complexity: High
   - Dependencies: 2 New
   - State Management Complexity: Medium
   - Novelty/Uncertainty Factor: Low
 * AI Pre-Task Self-Assessment: 85%
 * Problem Estimate: 88%
 * Initial Code Complexity Estimate: 87%
 * Final Code Complexity: 89%
 * Overall Result Score: 91%
 * Last Updated: 2025-06-07
 */

@MainActor
final class BenchmarkScheduler: ObservableObject {

    
    @Published var scheduledBenchmarks: [ScheduledBenchmark] = []
    @Published var isSchedulerRunning = false
    
    private var schedulerTimer: Timer?
    private let benchmarkEngine = ModelBenchmarkEngine()
    
    func scheduleBenchmark(_ benchmark: ScheduledBenchmark) {
        scheduledBenchmarks.append(benchmark)
        if !isSchedulerRunning {
            startScheduler()
        }
    }
    
    func cancelBenchmark(id: UUID) {
        scheduledBenchmarks.removeAll { $0.id == id }
    }
    
    private func startScheduler() {
        isSchedulerRunning = true
        schedulerTimer = Timer.scheduledTimer(withTimeInterval: 60.0, repeats: true) { [weak self] _ in
            self?.checkScheduledBenchmarks()
        }
    }
    
    private func checkScheduledBenchmarks() {
        let now = Date()
        let dueBenchmarks = scheduledBenchmarks.filter { $0.scheduledTime <= now && !$0.isCompleted }
        
        for benchmark in dueBenchmarks {
            Task {
                await executeBenchmark(benchmark)
            }
        }
    }
    
    private func executeBenchmark(_ benchmark: ScheduledBenchmark) async {
        do {
            let result = try await benchmarkEngine.runBenchmark(
                for: benchmark.model,
                prompts: benchmark.testPrompts
            )
            
            DispatchQueue.main.async { [weak self] in
                if let index = self?.scheduledBenchmarks.firstIndex(where: { $0.id == benchmark.id }) {
                    self?.scheduledBenchmarks[index].isCompleted = true
                    self?.scheduledBenchmarks[index].result = result
                }
            }
        } catch {
            print("Benchmark execution failed: \(error)")
        }
    }
}

// MARK: - Supporting Data Structures





