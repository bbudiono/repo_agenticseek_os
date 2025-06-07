import SwiftUI

// SANDBOX FILE: For testing/development. See .cursorrules.
/**
 * Purpose: UI for tier upgrades and billing
 * Issues & Complexity Summary: SwiftUI view for tiered architecture management
 * Key Complexity Drivers:
   - Logic Scope (Est. LoC): ~200
   - Core Algorithm Complexity: Medium
   - Dependencies: 2
   - State Management Complexity: Medium
   - Novelty/Uncertainty Factor: Medium
 * AI Pre-Task Self-Assessment: 88%
 * Problem Estimate: 85%
 * Initial Code Complexity Estimate: 87%
 * Final Code Complexity: 89%
 * Overall Result Score: 92%
 * Key Variances/Learnings: SwiftUI tier management requires careful state handling
 * Last Updated: 2025-06-07
 */

struct TierUpgradeView: View {
    @StateObject private var tierManager = TierManager.shared
    @State private var isLoading = false
    @State private var showingUpgrade = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header Section
                headerSection
                
                // Main Content
                ScrollView {
                    VStack(spacing: 24) {
                        currentTierSection
                        agentUsageSection
                        upgradeOptionsSection
                    }
                    .padding(24)
                }
            }
        }
        .navigationTitle("TierUpgrade")
        .onAppear {
            loadTierData()
        }
        .sheet(isPresented: $showingUpgrade) {
            TierUpgradeView()
        }
    }
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "person.3.sequence.fill")
                    .font(.title2)
                    .foregroundColor(.primary)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("MLACS Tiered System")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text("Manage your agent tiers and usage")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                if isLoading {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }
        }
        .padding(.horizontal, 24)
        .padding(.top, 16)
        .padding(.bottom, 8)
        .background(.regularMaterial)
    }
    
    private var currentTierSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Current Tier", systemImage: "star.fill")
                .font(.headline)
                .foregroundColor(.primary)
            
            TierCard(
                tier: tierManager.currentTier,
                isActive: true
            )
        }
        .padding()
        .background(Color(.systemGray6).opacity(0.5))
        .cornerRadius(12)
    }
    
    private var agentUsageSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Agent Usage", systemImage: "chart.bar.fill")
                .font(.headline)
                .foregroundColor(.primary)
            
            UsageMetricsView()
        }
        .padding()
        .background(Color(.systemGray6).opacity(0.5))
        .cornerRadius(12)
    }
    
    private var upgradeOptionsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Upgrade Options", systemImage: "arrow.up.circle.fill")
                .font(.headline)
                .foregroundColor(.primary)
            
            Button("View Upgrade Options") {
                showingUpgrade = true
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
        .background(Color(.systemGray6).opacity(0.5))
        .cornerRadius(12)
    }
    
    private func loadTierData() {
        isLoading = true
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            isLoading = false
            tierManager.refreshTierStatus()
        }
    }
}

struct TierCard: View {
    let tier: TierLevel
    let isActive: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: tier.icon)
                    .font(.title2)
                    .foregroundColor(tier.color)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(tier.name)
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text("\(tier.maxAgents) agents maximum")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                if isActive {
                    Text("ACTIVE")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.green)
                        .cornerRadius(4)
                }
            }
            
            if !tier.features.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(tier.features, id: \.self) { feature in
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                                .font(.caption)
                            
                            Text(feature)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
        }
        .padding()
        .background(isActive ? Color.blue.opacity(0.1) : Color.clear)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isActive ? Color.blue : Color.gray.opacity(0.3), lineWidth: isActive ? 2 : 1)
        )
    }
}

struct UsageMetricsView: View {
    var body: some View {
        VStack(spacing: 12) {
            Text("Feature coming soon")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
}

#Preview {
    TierUpgradeView()
}
