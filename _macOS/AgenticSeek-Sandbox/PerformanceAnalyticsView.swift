// SANDBOX FILE: For testing/development. See .cursorrules.
//
// PerformanceAnalyticsView.swift
// AgenticSeek Enhanced macOS
//
// Performance Analytics Dashboard Integration
// Real-time metrics, framework comparison, trend analysis, and bottleneck detection
//

import SwiftUI
import Foundation

// MARK: - Performance Analytics Dashboard View
struct PerformanceAnalyticsView: View {
    @StateObject private var analyticsManager = PerformanceAnalyticsManager()
    @State private var selectedTimeRange: TimeRange = .last24Hours
    @State private var selectedFramework: FrameworkFilter = .all
    @State private var showingDetailedMetrics = false
    @State private var autoRefresh = true
    @State private var refreshTimer: Timer?
    
    var body: some View {
        VStack(spacing: 0) {
            // Header with controls
            analyticsHeader
            
            if analyticsManager.isLoading {
                loadingView
            } else {
                ScrollView {
                    LazyVStack(spacing: DesignSystem.Spacing.space16) {
                        // System Health Overview
                        systemHealthCard
                        
                        // Real-time Metrics Grid
                        realTimeMetricsGrid
                        
                        // Framework Comparison
                        frameworkComparisonCard
                        
                        // Trend Analysis
                        trendAnalysisCard
                        
                        // Active Bottlenecks
                        bottlenecksCard
                        
                        // Recommendations
                        recommendationsCard
                    }
                    .padding(DesignSystem.Spacing.space16)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(DesignSystem.Colors.background)
        .onAppear {
            startAnalytics()
        }
        .onDisappear {
            stopAnalytics()
        }
        .onChange(of: autoRefresh) { newValue in
            if newValue {
                startAutoRefresh()
            } else {
                stopAutoRefresh()
            }
        }
    }
    
    // MARK: - Header View
    private var analyticsHeader: some View {
        VStack(spacing: DesignSystem.Spacing.space8) {
            HStack {
                VStack(alignment: .leading) {
                    Text("Performance Analytics")
                        .font(DesignSystem.Typography.headline)
                        .foregroundColor(DesignSystem.Colors.textPrimary)
                    
                    Text("Real-time LangGraph Framework Performance Monitoring")
                        .font(DesignSystem.Typography.caption)
                        .foregroundColor(DesignSystem.Colors.textSecondary)
                }
                
                Spacer()
                
                HStack(spacing: DesignSystem.Spacing.space8) {
                    // Auto-refresh toggle
                    Toggle("Auto Refresh", isOn: $autoRefresh)
                        .toggleStyle(SwitchToggleStyle())
                        .labelsHidden()
                        .accessibilityLabel("Auto refresh analytics data")
                    
                    // Manual refresh button
                    Button(action: refreshAnalytics) {
                        Image(systemName: "arrow.clockwise")
                            .foregroundColor(DesignSystem.Colors.primary)
                    }
                    .buttonStyle(.borderless)
                    .accessibilityLabel("Refresh analytics data")
                    
                    // Settings button
                    Button(action: { showingDetailedMetrics.toggle() }) {
                        Image(systemName: "gear")
                            .foregroundColor(DesignSystem.Colors.primary)
                    }
                    .buttonStyle(.borderless)
                    .accessibilityLabel("Analytics settings")
                }
            }
            
            // Filter controls
            HStack {
                Picker("Time Range", selection: $selectedTimeRange) {
                    ForEach(TimeRange.allCases, id: \.self) { range in
                        Text(range.rawValue).tag(range)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .frame(maxWidth: 300)
                
                Spacer()
                
                Picker("Framework", selection: $selectedFramework) {
                    ForEach(FrameworkFilter.allCases, id: \.self) { framework in
                        Text(framework.rawValue).tag(framework)
                    }
                }
                .pickerStyle(MenuPickerStyle())
                .frame(width: 120)
            }
        }
        .padding(DesignSystem.Spacing.space16)
        .background(DesignSystem.Colors.surface)
        .shadow(color: Color.black.opacity(0.1), radius: 1, x: 0, y: 1)
    }
    
    // MARK: - System Health Card
    private var systemHealthCard: some View {
        VStack(alignment: .leading, spacing: DesignSystem.Spacing.space12) {
            HStack {
                Image(systemName: "heart.fill")
                    .foregroundColor(healthScoreColor)
                Text("System Health")
                    .font(DesignSystem.Typography.title3)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                
                Spacer()
                
                Text("\(Int(analyticsManager.systemHealthScore * 100))%")
                    .font(DesignSystem.Typography.headline)
                    .foregroundColor(healthScoreColor)
            }
            
            // Health score progress bar
            ProgressView(value: analyticsManager.systemHealthScore)
                .progressViewStyle(LinearProgressViewStyle(tint: healthScoreColor))
                .scaleEffect(x: 1, y: 2, anchor: .center)
            
            HStack {
                systemHealthMetric("Uptime", value: analyticsManager.uptime)
                Spacer()
                systemHealthMetric("Alerts", value: "\(analyticsManager.alertCount)")
                Spacer()
                systemHealthMetric("Collections", value: "\(analyticsManager.metricsCollected)")
            }
        }
        .padding(DesignSystem.Spacing.space16)
        .background(DesignSystem.Colors.surface)
        .cornerRadius(DesignSystem.CornerRadius.medium)
        .shadow(color: Color.black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
    
    private func systemHealthMetric(_ label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(DesignSystem.Typography.caption)
                .foregroundColor(DesignSystem.Colors.textSecondary)
            Text(value)
                .font(DesignSystem.Typography.body)
                .foregroundColor(DesignSystem.Colors.textPrimary)
        }
    }
    
    private var healthScoreColor: Color {
        if analyticsManager.systemHealthScore >= 0.8 {
            return DesignSystem.Colors.success
        } else if analyticsManager.systemHealthScore >= 0.6 {
            return DesignSystem.Colors.warning
        } else {
            return DesignSystem.Colors.error
        }
    }
    
    // MARK: - Real-time Metrics Grid
    private var realTimeMetricsGrid: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: DesignSystem.Spacing.space12) {
            ForEach(analyticsManager.realTimeMetrics.keys.sorted(), id: \.self) { key in
                if let value = analyticsManager.realTimeMetrics[key] {
                    metricCard(title: formatMetricKey(key), value: formatMetricValue(key, value))
                }
            }
        }
    }
    
    private func metricCard(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: DesignSystem.Spacing.space8) {
            Text(title)
                .font(DesignSystem.Typography.caption)
                .foregroundColor(DesignSystem.Colors.textSecondary)
                .lineLimit(1)
            
            Text(value)
                .font(DesignSystem.Typography.title3)
                .foregroundColor(DesignSystem.Colors.textPrimary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(DesignSystem.Spacing.space12)
        .background(DesignSystem.Colors.surface)
        .cornerRadius(DesignSystem.CornerRadius.small)
        .shadow(color: Color.black.opacity(0.05), radius: 1, x: 0, y: 1)
    }
    
    // MARK: - Framework Comparison Card
    private var frameworkComparisonCard: some View {
        VStack(alignment: .leading, spacing: DesignSystem.Spacing.space12) {
            HStack {
                Image(systemName: "chart.bar.fill")
                    .foregroundColor(DesignSystem.Colors.primary)
                Text("Framework Performance Comparison")
                    .font(DesignSystem.Typography.title3)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                
                Spacer()
                
                Text("\(analyticsManager.frameworkComparisons.count) comparisons")
                    .font(DesignSystem.Typography.caption)
                    .foregroundColor(DesignSystem.Colors.textSecondary)
            }
            
            if analyticsManager.frameworkComparisons.isEmpty {
                Text("Collecting comparison data...")
                    .font(DesignSystem.Typography.body)
                    .foregroundColor(DesignSystem.Colors.textSecondary)
                    .padding(DesignSystem.Spacing.space16)
            } else {
                ForEach(Array(analyticsManager.frameworkComparisons.prefix(3)), id: \.comparisonId) { comparison in
                    frameworkComparisonRow(comparison)
                }
            }
        }
        .padding(DesignSystem.Spacing.space16)
        .background(DesignSystem.Colors.surface)
        .cornerRadius(DesignSystem.CornerRadius.medium)
        .shadow(color: Color.black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
    
    private func frameworkComparisonRow(_ comparison: FrameworkComparisonData) -> some View {
        VStack(alignment: .leading, spacing: DesignSystem.Spacing.space4) {
            HStack {
                Text("\(comparison.frameworkA) vs \(comparison.frameworkB)")
                    .font(DesignSystem.Typography.body)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                
                Spacer()
                
                Text("\(comparison.performanceRatio, specifier: "%.2f")x")
                    .font(DesignSystem.Typography.body)
                    .foregroundColor(comparison.performanceRatio > 1.0 ? DesignSystem.Colors.success : DesignSystem.Colors.error)
            }
            
            Text(comparison.recommendation)
                .font(DesignSystem.Typography.caption)
                .foregroundColor(DesignSystem.Colors.textSecondary)
                .lineLimit(2)
        }
        .padding(DesignSystem.Spacing.space8)
        .background(DesignSystem.Colors.background)
        .cornerRadius(DesignSystem.CornerRadius.small)
    }
    
    // MARK: - Trend Analysis Card
    private var trendAnalysisCard: some View {
        VStack(alignment: .leading, spacing: DesignSystem.Spacing.space12) {
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .foregroundColor(DesignSystem.Colors.primary)
                Text("Performance Trends")
                    .font(DesignSystem.Typography.title3)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                
                Spacer()
                
                Text("\(analyticsManager.trendAnalyses.count) trends")
                    .font(DesignSystem.Typography.caption)
                    .foregroundColor(DesignSystem.Colors.textSecondary)
            }
            
            if analyticsManager.trendAnalyses.isEmpty {
                Text("Analyzing performance trends...")
                    .font(DesignSystem.Typography.body)
                    .foregroundColor(DesignSystem.Colors.textSecondary)
                    .padding(DesignSystem.Spacing.space16)
            } else {
                ForEach(Array(analyticsManager.trendAnalyses.prefix(3)), id: \.trendId) { trend in
                    trendAnalysisRow(trend)
                }
            }
        }
        .padding(DesignSystem.Spacing.space16)
        .background(DesignSystem.Colors.surface)
        .cornerRadius(DesignSystem.CornerRadius.medium)
        .shadow(color: Color.black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
    
    private func trendAnalysisRow(_ trend: TrendAnalysisData) -> some View {
        HStack {
            Image(systemName: trendIcon(trend.direction))
                .foregroundColor(trendColor(trend.direction))
                .frame(width: 20)
            
            VStack(alignment: .leading, spacing: 2) {
                Text("\(trend.framework) \(trend.metricType)")
                    .font(DesignSystem.Typography.body)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                
                Text("\(trend.direction) trend (strength: \(Int(trend.trendStrength * 100))%)")
                    .font(DesignSystem.Typography.caption)
                    .foregroundColor(DesignSystem.Colors.textSecondary)
            }
            
            Spacer()
            
            Text("\(trend.slope, specifier: "%.2f")")
                .font(DesignSystem.Typography.caption)
                .foregroundColor(DesignSystem.Colors.textSecondary)
        }
        .padding(DesignSystem.Spacing.space8)
        .background(DesignSystem.Colors.background)
        .cornerRadius(DesignSystem.CornerRadius.small)
    }
    
    private func trendIcon(_ direction: String) -> String {
        switch direction.lowercased() {
        case "improving": return "arrow.up.circle.fill"
        case "degrading": return "arrow.down.circle.fill"
        case "stable": return "minus.circle.fill"
        case "volatile": return "exclamationmark.triangle.fill"
        default: return "questionmark.circle.fill"
        }
    }
    
    private func trendColor(_ direction: String) -> Color {
        switch direction.lowercased() {
        case "improving": return DesignSystem.Colors.success
        case "degrading": return DesignSystem.Colors.error
        case "stable": return DesignSystem.Colors.primary
        case "volatile": return DesignSystem.Colors.warning
        default: return DesignSystem.Colors.disabled
        }
    }
    
    // MARK: - Bottlenecks Card
    private var bottlenecksCard: some View {
        VStack(alignment: .leading, spacing: DesignSystem.Spacing.space12) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(analyticsManager.activeBottlenecks.isEmpty ? DesignSystem.Colors.success : DesignSystem.Colors.warning)
                Text("Active Bottlenecks")
                    .font(DesignSystem.Typography.title3)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                
                Spacer()
                
                Text("\(analyticsManager.activeBottlenecks.count) active")
                    .font(DesignSystem.Typography.caption)
                    .foregroundColor(analyticsManager.activeBottlenecks.isEmpty ? DesignSystem.Colors.success : DesignSystem.Colors.warning)
            }
            
            if analyticsManager.activeBottlenecks.isEmpty {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(DesignSystem.Colors.success)
                    Text("No performance bottlenecks detected")
                        .font(DesignSystem.Typography.body)
                        .foregroundColor(DesignSystem.Colors.textPrimary)
                }
                .padding(DesignSystem.Spacing.space16)
            } else {
                ForEach(analyticsManager.activeBottlenecks, id: \.bottleneckId) { bottleneck in
                    bottleneckRow(bottleneck)
                }
            }
        }
        .padding(DesignSystem.Spacing.space16)
        .background(DesignSystem.Colors.surface)
        .cornerRadius(DesignSystem.CornerRadius.medium)
        .shadow(color: Color.black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
    
    private func bottleneckRow(_ bottleneck: BottleneckData) -> some View {
        VStack(alignment: .leading, spacing: DesignSystem.Spacing.space8) {
            HStack {
                Image(systemName: bottleneckIcon(bottleneck.bottleneckType))
                    .foregroundColor(severityColor(bottleneck.severity))
                
                Text(bottleneck.bottleneckType)
                    .font(DesignSystem.Typography.body)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                
                Spacer()
                
                Text("Severity: \(Int(bottleneck.severity * 100))%")
                    .font(DesignSystem.Typography.caption)
                    .foregroundColor(severityColor(bottleneck.severity))
            }
            
            Text(bottleneck.rootCause)
                .font(DesignSystem.Typography.caption)
                .foregroundColor(DesignSystem.Colors.textSecondary)
                .lineLimit(2)
            
            if !bottleneck.suggestedResolution.isEmpty {
                Text("💡 \(bottleneck.suggestedResolution)")
                    .font(DesignSystem.Typography.caption)
                    .foregroundColor(DesignSystem.Colors.primary)
                    .lineLimit(2)
            }
        }
        .padding(DesignSystem.Spacing.space8)
        .background(DesignSystem.Colors.background)
        .cornerRadius(DesignSystem.CornerRadius.small)
    }
    
    private func bottleneckIcon(_ type: String) -> String {
        switch type.lowercased() {
        case "cpu_bound": return "cpu"
        case "memory_bound": return "memorychip"
        case "io_bound": return "externaldrive"
        case "framework_overhead": return "gearshape"
        default: return "exclamationmark.triangle"
        }
    }
    
    private func severityColor(_ severity: Double) -> Color {
        if severity >= 0.8 {
            return DesignSystem.Colors.error
        } else if severity >= 0.6 {
            return DesignSystem.Colors.warning
        } else {
            return DesignSystem.Colors.primary
        }
    }
    
    // MARK: - Recommendations Card
    private var recommendationsCard: some View {
        VStack(alignment: .leading, spacing: DesignSystem.Spacing.space12) {
            HStack {
                Image(systemName: "lightbulb.fill")
                    .foregroundColor(DesignSystem.Colors.primary)
                Text("Performance Recommendations")
                    .font(DesignSystem.Typography.title3)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                
                Spacer()
                
                Text("\(analyticsManager.recommendations.count) suggestions")
                    .font(DesignSystem.Typography.caption)
                    .foregroundColor(DesignSystem.Colors.textSecondary)
            }
            
            if analyticsManager.recommendations.isEmpty {
                Text("No recommendations at this time")
                    .font(DesignSystem.Typography.body)
                    .foregroundColor(DesignSystem.Colors.textSecondary)
                    .padding(DesignSystem.Spacing.space16)
            } else {
                ForEach(Array(analyticsManager.recommendations.enumerated()), id: \.offset) { index, recommendation in
                    recommendationRow(index + 1, recommendation)
                }
            }
        }
        .padding(DesignSystem.Spacing.space16)
        .background(DesignSystem.Colors.surface)
        .cornerRadius(DesignSystem.CornerRadius.medium)
        .shadow(color: Color.black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
    
    private func recommendationRow(_ number: Int, _ recommendation: String) -> some View {
        HStack(alignment: .top, spacing: DesignSystem.Spacing.space8) {
            Text("\(number)")
                .font(DesignSystem.Typography.caption)
                .foregroundColor(DesignSystem.Colors.primary)
                .frame(width: 20, height: 20)
                .background(DesignSystem.Colors.primary.opacity(0.1))
                .clipShape(Circle())
            
            Text(recommendation)
                .font(DesignSystem.Typography.body)
                .foregroundColor(DesignSystem.Colors.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
            
            Spacer()
        }
        .padding(DesignSystem.Spacing.space8)
        .background(DesignSystem.Colors.background)
        .cornerRadius(DesignSystem.CornerRadius.small)
    }
    
    // MARK: - Loading View
    private var loadingView: some View {
        VStack(spacing: DesignSystem.Spacing.space20) {
            ProgressView()
                .scaleEffect(1.2)
            
            Text("Loading Performance Analytics...")
                .font(DesignSystem.Typography.title3)
                .foregroundColor(DesignSystem.Colors.textSecondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    // MARK: - Helper Functions
    private func formatMetricKey(_ key: String) -> String {
        key.replacingOccurrences(of: "_", with: " ")
            .capitalized
    }
    
    private func formatMetricValue(_ key: String, _ value: Any) -> String {
        if let doubleValue = value as? Double {
            if key.contains("time") || key.contains("latency") {
                return String(format: "%.1fms", doubleValue)
            } else if key.contains("percent") || key.contains("utilization") {
                return String(format: "%.1f%%", doubleValue)
            } else if key.contains("mb") {
                return String(format: "%.1fMB", doubleValue)
            } else {
                return String(format: "%.2f", doubleValue)
            }
        } else if let intValue = value as? Int {
            return "\(intValue)"
        } else {
            return "\(value)"
        }
    }
    
    // MARK: - Analytics Control
    private func startAnalytics() {
        analyticsManager.startAnalytics()
        if autoRefresh {
            startAutoRefresh()
        }
    }
    
    private func stopAnalytics() {
        analyticsManager.stopAnalytics()
        stopAutoRefresh()
    }
    
    private func refreshAnalytics() {
        analyticsManager.refreshData()
    }
    
    private func startAutoRefresh() {
        stopAutoRefresh()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { _ in
            analyticsManager.refreshData()
        }
    }
    
    private func stopAutoRefresh() {
        refreshTimer?.invalidate()
        refreshTimer = nil
    }
}

// MARK: - Performance Analytics Manager
@MainActor
class PerformanceAnalyticsManager: ObservableObject {
    @Published var isLoading = true
    @Published var systemHealthScore: Double = 0.75
    @Published var uptime: String = "0:00:00"
    @Published var alertCount: Int = 0
    @Published var metricsCollected: Int = 0
    @Published var realTimeMetrics: [String: Any] = [:]
    @Published var frameworkComparisons: [FrameworkComparisonData] = []
    @Published var trendAnalyses: [TrendAnalysisData] = []
    @Published var activeBottlenecks: [BottleneckData] = []
    @Published var recommendations: [String] = []
    
    private var analyticsTask: Task<Void, Never>?
    
    func startAnalytics() {
        analyticsTask = Task {
            // Simulate loading
            try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
            
            await MainActor.run {
                self.isLoading = false
                self.loadMockData()
            }
        }
    }
    
    func stopAnalytics() {
        analyticsTask?.cancel()
        analyticsTask = nil
    }
    
    func refreshData() {
        // Simulate data refresh
        loadMockData()
        
        // Update metrics with slight variations
        if let executionTime = realTimeMetrics["avg_execution_time"] as? Double {
            realTimeMetrics["avg_execution_time"] = executionTime + Double.random(in: -5...5)
        }
        
        metricsCollected += Int.random(in: 10...50)
        systemHealthScore = max(0.0, min(1.0, systemHealthScore + Double.random(in: -0.05...0.05)))
    }
    
    private func loadMockData() {
        // System health data
        systemHealthScore = 0.85
        uptime = "2:34:12"
        alertCount = 0
        metricsCollected = 1247
        
        // Real-time metrics
        realTimeMetrics = [
            "avg_execution_time": 127.5,
            "avg_throughput": 94.2,
            "avg_error_rate": 1.8,
            "avg_cpu_utilization": 45.3,
            "avg_memory_usage": 62.1,
            "metrics_collected_total": 1247,
            "collection_errors": 0,
            "avg_collection_time_ms": 0.11,
            "buffer_utilization": 0.15
        ]
        
        // Framework comparisons
        frameworkComparisons = [
            FrameworkComparisonData(
                comparisonId: "1",
                frameworkA: "LangGraph",
                frameworkB: "LangChain",
                performanceRatio: 1.25,
                recommendation: "LangGraph shows 25% better performance for execution time"
            ),
            FrameworkComparisonData(
                comparisonId: "2",
                frameworkA: "Hybrid",
                frameworkB: "LangChain",
                performanceRatio: 1.12,
                recommendation: "Hybrid approach provides 12% performance improvement"
            ),
            FrameworkComparisonData(
                comparisonId: "3",
                frameworkA: "LangGraph",
                frameworkB: "Hybrid",
                performanceRatio: 1.08,
                recommendation: "LangGraph maintains slight edge over hybrid approach"
            )
        ]
        
        // Trend analyses
        trendAnalyses = [
            TrendAnalysisData(
                trendId: "1",
                framework: "LangGraph",
                metricType: "Execution Time",
                direction: "Improving",
                slope: -2.3,
                trendStrength: 0.87
            ),
            TrendAnalysisData(
                trendId: "2",
                framework: "LangChain",
                metricType: "Throughput",
                direction: "Stable",
                slope: 0.1,
                trendStrength: 0.34
            ),
            TrendAnalysisData(
                trendId: "3",
                framework: "Hybrid",
                metricType: "Error Rate",
                direction: "Improving",
                slope: -0.8,
                trendStrength: 0.92
            )
        ]
        
        // Active bottlenecks (empty for good system health)
        activeBottlenecks = []
        
        // Recommendations
        recommendations = [
            "System performance is optimal. Continue monitoring for optimization opportunities.",
            "Consider enabling LangGraph for new workflows based on performance comparison data.",
            "Memory usage is within acceptable ranges. No action required at this time."
        ]
    }
}

// MARK: - Data Models
struct FrameworkComparisonData {
    let comparisonId: String
    let frameworkA: String
    let frameworkB: String
    let performanceRatio: Double
    let recommendation: String
}

struct TrendAnalysisData {
    let trendId: String
    let framework: String
    let metricType: String
    let direction: String
    let slope: Double
    let trendStrength: Double
}

struct BottleneckData {
    let bottleneckId: String
    let bottleneckType: String
    let severity: Double
    let rootCause: String
    let suggestedResolution: String
}

// MARK: - Filter Enums
enum TimeRange: String, CaseIterable {
    case last1Hour = "1H"
    case last6Hours = "6H"
    case last24Hours = "24H"
    case last7Days = "7D"
}

enum FrameworkFilter: String, CaseIterable {
    case all = "All"
    case langchain = "LangChain"
    case langgraph = "LangGraph"
    case hybrid = "Hybrid"
}

#Preview {
    PerformanceAnalyticsView()
}