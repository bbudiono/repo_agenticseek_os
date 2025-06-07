//
// SystemPerformanceAnalyticsDashboard.swift
// AgenticSeek System Performance Analysis
//
// PHASE 6: Comprehensive performance analytics dashboard
// Real-time metrics, charts, optimization controls, and system insights
// Created: 2025-06-07 20:52:15
//

import SwiftUI
import Charts
import Combine

struct SystemPerformanceAnalyticsDashboard: View {
    @StateObject private var performanceProfiler = SystemPerformanceProfiler()
    @State private var selectedTimeRange: TimeRange = .last5Minutes
    @State private var showingOptimizationPanel = false
    @State private var showingDetailedReport = false
    @State private var selectedMetric: MetricType = .cpu
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Header with system health status
                systemHealthHeader
                
                // Main dashboard content
                ScrollView {
                    LazyVStack(spacing: 20) {
                        // Real-time metrics overview
                        realTimeMetricsSection
                        
                        // Performance charts
                        performanceChartsSection
                        
                        // System recommendations
                        recommendationsSection
                        
                        // Quick actions
                        quickActionsSection
                    }
                    .padding()
                }
            }
            .navigationTitle("System Performance")
            .toolbar {
                ToolbarItemGroup(placement: .navigationBarTrailing) {
                    Button("Optimization") {
                        showingOptimizationPanel = true
                    }
                    
                    Button("Report") {
                        showingDetailedReport = true
                    }
                }
            }
        }
        .task {
            await performanceProfiler.startMonitoring()
        }
        .onDisappear {
            performanceProfiler.stopMonitoring()
        }
        .sheet(isPresented: $showingOptimizationPanel) {
            OptimizationControlPanel(profiler: performanceProfiler)
        }
        .sheet(isPresented: $showingDetailedReport) {
            PerformanceReportView(profiler: performanceProfiler)
        }
    }
    
    // MARK: - System Health Header
    
    private var systemHealthHeader: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("System Health")
                    .font(.headline)
                Text(performanceProfiler.systemHealth.rawValue)
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(healthColor)
            }
            
            Spacer()
            
            Circle()
                .fill(healthColor)
                .frame(width: 12, height: 12)
                .animation(.easeInOut, value: performanceProfiler.systemHealth)
        }
        .padding()
        .background(Color(.systemBackground))
        .shadow(radius: 2)
    }
    
    private var healthColor: Color {
        switch performanceProfiler.systemHealth {
        case .optimal: return .green
        case .good: return .blue
        case .warning: return .orange
        case .critical: return .red
        }
    }
    
    // MARK: - Real-time Metrics Section
    
    private var realTimeMetricsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Real-time Metrics")
                .font(.headline)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 16) {
                MetricCard(
                    title: "CPU Usage",
                    value: performanceProfiler.currentMetrics.avgCPUUsage,
                    unit: "%",
                    color: .blue,
                    isSelected: selectedMetric == .cpu
                ) {
                    selectedMetric = .cpu
                }
                
                MetricCard(
                    title: "Memory",
                    value: performanceProfiler.currentMetrics.avgMemoryUsage,
                    unit: "%",
                    color: .green,
                    isSelected: selectedMetric == .memory
                ) {
                    selectedMetric = .memory
                }
                
                MetricCard(
                    title: "GPU",
                    value: performanceProfiler.currentMetrics.avgGPUUsage,
                    unit: "%",
                    color: .purple,
                    isSelected: selectedMetric == .gpu
                ) {
                    selectedMetric = .gpu
                }
            }
        }
    }
    
    // MARK: - Performance Charts Section
    
    private var performanceChartsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Performance Trends")
                    .font(.headline)
                
                Spacer()
                
                Picker("Time Range", selection: $selectedTimeRange) {
                    ForEach(TimeRange.allCases, id: \.self) { range in
                        Text(range.rawValue).tag(range)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
            }
            
            if #available(macOS 13.0, *) {
                Chart {
                    ForEach(filteredPerformanceData, id: \.timestamp) { snapshot in
                        LineMark(
                            x: .value("Time", snapshot.timestamp),
                            y: .value("Usage", metricValue(for: snapshot))
                        )
                        .foregroundStyle(chartColor)
                    }
                }
                .frame(height: 200)
                .chartYAxis {
                    AxisMarks(position: .leading)
                }
                .chartXAxis {
                    AxisMarks(values: .stride(by: .minute, count: 1)) { value in
                        AxisValueLabel(format: .dateTime.hour().minute())
                    }
                }
            } else {
                // Fallback for older macOS versions
                Text("Performance charts require macOS 13.0 or later")
                    .foregroundColor(.secondary)
                    .frame(height: 200)
            }
        }
    }
    
    // MARK: - Recommendations Section
    
    private var recommendationsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Optimization Recommendations")
                .font(.headline)
            
            let report = performanceProfiler.getPerformanceReport()
            ForEach(Array(report.recommendations.enumerated()), id: \.offset) { index, recommendation in
                HStack {
                    Image(systemName: "lightbulb.fill")
                        .foregroundColor(.yellow)
                    
                    Text(recommendation)
                        .font(.body)
                    
                    Spacer()
                }
                .padding()
                .background(Color(.systemBackground))
                .cornerRadius(8)
                .shadow(radius: 1)
            }
        }
    }
    
    // MARK: - Quick Actions Section
    
    private var quickActionsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Quick Actions")
                .font(.headline)
            
            HStack(spacing: 16) {
                Button("Reset Metrics") {
                    performanceProfiler.resetMetrics()
                }
                .buttonStyle(.bordered)
                
                Button("Export Report") {
                    exportPerformanceReport()
                }
                .buttonStyle(.bordered)
                
                Button("Optimize Now") {
                    showingOptimizationPanel = true
                }
                .buttonStyle(.borderedProminent)
                
                Spacer()
            }
        }
    }
    
    // MARK: - Helper Properties
    
    private var filteredPerformanceData: [PerformanceSnapshot] {
        let cutoffDate = Date().addingTimeInterval(-selectedTimeRange.timeInterval)
        return performanceProfiler.performanceHistory.filter { $0.timestamp >= cutoffDate }
    }
    
    private var chartColor: Color {
        switch selectedMetric {
        case .cpu: return .blue
        case .memory: return .green
        case .gpu: return .purple
        }
    }
    
    private func metricValue(for snapshot: PerformanceSnapshot) -> Double {
        switch selectedMetric {
        case .cpu: return snapshot.cpuUsage
        case .memory: return snapshot.memoryUsage
        case .gpu: return snapshot.gpuUsage
        }
    }
    
    private func exportPerformanceReport() {
        let report = performanceProfiler.getPerformanceReport()
        // Implementation for exporting report
        print("Exporting performance report with \(report.history.count) data points")
    }
}

// MARK: - Supporting Views

struct MetricCard: View {
    let title: String
    let value: Double
    let unit: String
    let color: Color
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text("\(value, specifier: "%.1f")\(unit)")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(color)
                
                ProgressView(value: value / 100.0)
                    .progressViewStyle(LinearProgressViewStyle(tint: color))
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(12)
            .shadow(radius: isSelected ? 4 : 2)
            .scaleEffect(isSelected ? 1.05 : 1.0)
            .animation(.easeInOut(duration: 0.2), value: isSelected)
        }
        .buttonStyle(.plain)
    }
}

struct OptimizationControlPanel: View {
    let profiler: SystemPerformanceProfiler
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            VStack {
                Text("Performance Optimization")
                    .font(.title)
                    .padding()
                
                Text("Optimization controls will be implemented in the next iteration")
                    .foregroundColor(.secondary)
                
                Spacer()
            }
            .navigationTitle("Optimization")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct PerformanceReportView: View {
    let profiler: SystemPerformanceProfiler
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    let report = profiler.getPerformanceReport()
                    
                    Text("Performance Report")
                        .font(.title)
                    
                    Text("Generated at: \(report.generatedAt, formatter: DateFormatter.reportFormatter)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    // Report content here
                    Text("Detailed performance report will be implemented in the next iteration")
                        .foregroundColor(.secondary)
                }
                .padding()
            }
            .navigationTitle("Report")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Supporting Types

enum TimeRange: String, CaseIterable {
    case last5Minutes = "5m"
    case last15Minutes = "15m"
    case last1Hour = "1h"
    case last24Hours = "24h"
    
    var timeInterval: TimeInterval {
        switch self {
        case .last5Minutes: return 300
        case .last15Minutes: return 900
        case .last1Hour: return 3600
        case .last24Hours: return 86400
        }
    }
}

enum MetricType {
    case cpu, memory, gpu
}

extension DateFormatter {
    static let reportFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .medium
        return formatter
    }()
}