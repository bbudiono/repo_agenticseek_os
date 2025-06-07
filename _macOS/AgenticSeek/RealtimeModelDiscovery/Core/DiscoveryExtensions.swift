import Foundation
import SwiftUI
import Combine

// SANDBOX FILE: For testing/development. See .cursorrules.

/**
 * Purpose: Extensions and utilities for MLACS Real-time Model Discovery
 * Issues & Complexity Summary: Helper methods and computed properties for enhanced functionality
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~180
   - Core Algorithm Complexity: Low
   - Dependencies: 3 New
   - State Management Complexity: Low
   - Novelty/Uncertainty Factor: Low
 * AI Pre-Task Self-Assessment: 92%
 * Problem Estimate: 92%
 * Initial Code Complexity Estimate: 82%
 * Final Code Complexity: 85%
 * Overall Result Score: 96%
 * Last Updated: 2025-01-07
 */

// MARK: - DiscoveredModel Extensions

extension DiscoveredModel {
    
    var statusIcon: String {
        switch availability_status {
        case "available": return "checkmark.circle.fill"
        case "downloading": return "arrow.down.circle.fill"
        case "error": return "xmark.circle.fill"
        default: return "questionmark.circle.fill"
        }
    }
    
    var statusColor: Color {
        switch availability_status {
        case "available": return .green
        case "downloading": return .blue
        case "error": return .red
        default: return .orange
        }
    }
    
    var formattedSize: String {
        if size_gb < 1.0 {
            return String(format: "%.0f MB", size_gb * 1024)
        } else {
            return String(format: "%.1f GB", size_gb)
        }
    }
    
    var providerIcon: String {
        switch provider.lowercased() {
        case "ollama": return "server.rack"
        case "lm_studio": return "laptopcomputer"
        case "huggingface": return "cloud"
        default: return "questionmark.app"
        }
    }
    
    var capabilitiesFormatted: String {
        return capabilities.map { $0.replacingOccurrences(of: "-", with: " ").capitalized }.joined(separator: ", ")
    }
    
    var discoveredTimeAgo: String {
        guard let date = ISO8601DateFormatter().date(from: discovered_at) else {
            return "Unknown"
        }
        return date.relativeDateString()
    }
    
    var lastVerifiedTimeAgo: String {
        guard let date = ISO8601DateFormatter().date(from: last_verified) else {
            return "Never"
        }
        return date.relativeDateString()
    }
    
    var overallScore: Double {
        return (performance_score + compatibility_score) / 2.0
    }
    
    var overallGrade: String {
        let score = overallScore
        switch score {
        case 0.9...1.0: return "A+"
        case 0.8..<0.9: return "A"
        case 0.7..<0.8: return "B"
        case 0.6..<0.7: return "C"
        default: return "F"
        }
    }
    
    func hasCapability(_ capability: String) -> Bool {
        return capabilities.contains { $0.lowercased() == capability.lowercased() }
    }
    
    func isCompatibleWith(systemMemoryGB: Double) -> Bool {
        return size_gb <= systemMemoryGB * 0.8 // Leave 20% memory buffer
    }
    
    func recommendationText() -> String {
        var text = ""
        
        if performance_score > 0.9 {
            text += "High Performance"
        } else if performance_score > 0.7 {
            text += "Good Performance"
        } else {
            text += "Basic Performance"
        }
        
        if compatibility_score > 0.9 {
            text += " • Excellent Compatibility"
        } else if compatibility_score > 0.7 {
            text += " • Good Compatibility"
        } else {
            text += " • Limited Compatibility"
        }
        
        return text
    }
}

// MARK: - Array Extensions

extension Array where Element == DiscoveredModel {
    
    func sortedByRecommendation() -> [DiscoveredModel] {
        return sorted { first, second in
            if first.recommendation_rank != second.recommendation_rank {
                return first.recommendation_rank < second.recommendation_rank
            }
            return first.overallScore > second.overallScore
        }
    }
    
    func sortedByPerformance() -> [DiscoveredModel] {
        return sorted { $0.performance_score > $1.performance_score }
    }
    
    func sortedBySize() -> [DiscoveredModel] {
        return sorted { $0.size_gb < $1.size_gb }
    }
    
    func sortedByName() -> [DiscoveredModel] {
        return sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }
    
    func filteredBy(provider: String) -> [DiscoveredModel] {
        guard provider != "All" else { return self }
        return filter { $0.provider.lowercased() == provider.lowercased() }
    }
    
    func filteredBy(capability: String) -> [DiscoveredModel] {
        guard capability != "All" else { return self }
        return filter { $0.hasCapability(capability) }
    }
    
    func filteredBy(status: String) -> [DiscoveredModel] {
        guard status != "All" else { return self }
        return filter { $0.availability_status == status }
    }
    
    func search(query: String) -> [DiscoveredModel] {
        guard !query.isEmpty else { return self }
        let lowercaseQuery = query.lowercased()
        
        return filter { model in
            model.name.lowercased().contains(lowercaseQuery) ||
            model.id.lowercased().contains(lowercaseQuery) ||
            model.capabilities.contains { $0.lowercased().contains(lowercaseQuery) } ||
            model.provider.lowercased().contains(lowercaseQuery)
        }
    }
    
    func topRecommendations(limit: Int = 5) -> [DiscoveredModel] {
        return sortedByRecommendation().prefix(limit).map { $0 }
    }
    
    func availableModels() -> [DiscoveredModel] {
        return filter { $0.availability_status == "available" }
    }
    
    func groupedByProvider() -> [String: [DiscoveredModel]] {
        return Dictionary(grouping: self) { $0.provider }
    }
    
    func averagePerformanceScore() -> Double {
        guard !isEmpty else { return 0.0 }
        return reduce(0) { $0 + $1.performance_score } / Double(count)
    }
    
    func totalSizeGB() -> Double {
        return reduce(0) { $0 + $1.size_gb }
    }
}

// MARK: - ModelProvider Extensions

extension ModelProvider {
    
    var displayIcon: String {
        switch type {
        case .local: return "desktopcomputer"
        case .remote: return "cloud"
        case .hybrid: return "laptopcomputer.and.iphone"
        }
    }
    
    var isHealthy: Bool {
        return status == .active && lastChecked.timeIntervalSinceNow > -300 // 5 minutes
    }
    
    func connectionDescription() -> String {
        switch status {
        case .active:
            return "Connected and operational"
        case .inactive:
            return "Temporarily disconnected"
        case .error:
            return "Connection error occurred"
        case .unknown:
            return "Status unknown"
        }
    }
}

// MARK: - DiscoveryEvent Extensions

extension DiscoveryEvent {
    
    var severityIcon: String {
        switch severity {
        case .low: return "info.circle"
        case .medium: return "exclamationmark.circle"
        case .high: return "exclamationmark.triangle"
        case .critical: return "xmark.octagon"
        }
    }
    
    var severityColor: Color {
        switch severity {
        case .low: return .secondary
        case .medium: return .blue
        case .high: return .orange
        case .critical: return .red
        }
    }
    
    var timeAgo: String {
        return timestamp.relativeDateString()
    }
    
    var formattedMessage: String {
        var formatted = message
        
        // Add context if available
        if let modelId = modelId {
            formatted += " (Model: \(modelId))"
        }
        
        if let providerId = providerId {
            formatted += " (Provider: \(providerId))"
        }
        
        return formatted
    }
}

// MARK: - Color Extensions for Discovery

extension Color {
    
    static func forPerformanceScore(_ score: Double) -> Color {
        switch score {
        case 0.9...1.0: return .green
        case 0.8..<0.9: return .blue
        case 0.7..<0.8: return .yellow
        case 0.6..<0.7: return .orange
        default: return .red
        }
    }
    
    static func forModelType(_ type: String) -> Color {
        switch type.lowercased() {
        case "chat": return .blue
        case "completion": return .green
        case "embedding": return .purple
        case "code": return .orange
        default: return .gray
        }
    }
    
    static func forProvider(_ provider: String) -> Color {
        switch provider.lowercased() {
        case "ollama": return .blue
        case "lm_studio": return .purple
        case "huggingface": return .orange
        default: return .gray
        }
    }
}

// MARK: - View Helpers

struct ModelDiscoveryRow: View {
    let model: DiscoveredModel
    
    var body: some View {
        HStack(spacing: 12) {
            // Status indicator
            Image(systemName: model.statusIcon)
                .foregroundColor(model.statusColor)
                .font(.title2)
            
            VStack(alignment: .leading, spacing: 4) {
                // Model name and provider
                HStack {
                    Text(model.name)
                        .font(.headline)
                        .lineLimit(1)
                    
                    Spacer()
                    
                    Image(systemName: model.providerIcon)
                        .foregroundColor(.secondary)
                    
                    Text(model.provider.capitalized)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                // Capabilities
                Text(model.capabilitiesFormatted)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                
                // Performance and size info
                HStack {
                    Text(model.formattedSize)
                        .font(.caption)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.secondary.opacity(0.2))
                        .cornerRadius(4)
                    
                    Text(model.overallGrade)
                        .font(.caption)
                        .fontWeight(.semibold)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.forPerformanceScore(model.overallScore))
                        .foregroundColor(.white)
                        .cornerRadius(4)
                    
                    Spacer()
                    
                    Text("Updated \(model.lastVerifiedTimeAgo)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(.vertical, 4)
    }
}

struct ModelBrowserRow: View {
    let model: DiscoveredModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(model.name)
                    .font(.headline)
                    .lineLimit(1)
                
                Spacer()
                
                Text(model.formattedSize)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(model.capabilitiesFormatted)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(2)
            
            HStack {
                Label(model.provider.capitalized, systemImage: model.providerIcon)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text(model.overallGrade)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(Color.forPerformanceScore(model.overallScore))
            }
        }
        .padding(.vertical, 2)
    }
}

struct ModelDetailView: View {
    let model: DiscoveredModel
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Header
                VStack(alignment: .leading, spacing: 8) {
                    Text(model.name)
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text(model.id)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                // Status and basic info
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Label("Status", systemImage: model.statusIcon)
                            .foregroundColor(model.statusColor)
                        
                        Spacer()
                        
                        Text(model.availability_status.capitalized)
                            .fontWeight(.medium)
                            .foregroundColor(model.statusColor)
                    }
                    
                    Divider()
                    
                    InfoRow(label: "Provider", value: model.provider.capitalized)
                    InfoRow(label: "Size", value: model.formattedSize)
                    InfoRow(label: "Type", value: model.model_type.capitalized)
                    InfoRow(label: "Performance", value: "\(String(format: "%.1f", model.performance_score * 100))%")
                    InfoRow(label: "Compatibility", value: "\(String(format: "%.1f", model.compatibility_score * 100))%")
                    InfoRow(label: "Overall Grade", value: model.overallGrade)
                }
                .padding()
                .background(Color(.controlBackgroundColor))
                .cornerRadius(8)
                
                // Capabilities
                VStack(alignment: .leading, spacing: 8) {
                    Text("Capabilities")
                        .font(.headline)
                    
                    LazyVGrid(columns: [
                        GridItem(.adaptive(minimum: 120))
                    ], spacing: 8) {
                        ForEach(model.capabilities, id: \.self) { capability in
                            Text(capability.replacingOccurrences(of: "-", with: " ").capitalized)
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.blue.opacity(0.2))
                                .cornerRadius(6)
                        }
                    }
                }
                .padding()
                .background(Color(.controlBackgroundColor))
                .cornerRadius(8)
                
                Spacer()
            }
            .padding()
        }
        .navigationTitle("Model Details")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct InfoRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            
            Spacer()
            
            Text(value)
                .fontWeight(.medium)
        }
    }
}
