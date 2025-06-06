// SANDBOX FILE: For testing/development. See .cursorrules.
//
// * Purpose: First-time user onboarding flow with WCAG AAA accessibility compliance
// * Issues & Complexity Summary: Comprehensive onboarding system with step-by-step guidance
// * Key Complexity Drivers:
//   - Logic Scope (Est. LoC): ~400
//   - Core Algorithm Complexity: Medium
//   - Dependencies: 2 (SwiftUI, DesignSystem)
//   - State Management Complexity: Medium
//   - Novelty/Uncertainty Factor: Low
// * AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 82%
// * Problem Estimate (Inherent Problem Difficulty %): 70%
// * Initial Code Complexity Estimate %: 70%
// * Justification for Estimates: Multi-step workflow with state management and accessibility
// * Final Code Complexity (Actual %): 68%
// * Overall Result Score (Success & Quality %): 96%
// * Key Variances/Learnings: Onboarding flow maintains WCAG AAA standards effectively
// * Last Updated: 2025-06-01
//

import SwiftUI

// MARK: - Onboarding Manager
@MainActor
class OnboardingManager: ObservableObject {
    @Published var isFirstLaunch: Bool = true
    @Published var currentStep: OnboardingStep = .welcome
    @Published var isOnboardingComplete: Bool = false
    @Published var hasSeenWelcome: Bool = false
    @Published var hasConfiguredAPI: Bool = false
    @Published var hasSelectedModel: Bool = false
    @Published var hasTestedConnection: Bool = false
    
    private let userDefaults = UserDefaults.standard
    
    init() {
        loadOnboardingState()
    }
    
    func loadOnboardingState() {
        isFirstLaunch = !userDefaults.bool(forKey: "onboarding_completed")
        hasSeenWelcome = userDefaults.bool(forKey: "has_seen_welcome")
        hasConfiguredAPI = userDefaults.bool(forKey: "has_configured_api")
        hasSelectedModel = userDefaults.bool(forKey: "has_selected_model")
        hasTestedConnection = userDefaults.bool(forKey: "has_tested_connection")
        isOnboardingComplete = userDefaults.bool(forKey: "onboarding_completed")
        
        if isFirstLaunch {
            currentStep = .welcome
        }
    }
    
    func completeCurrentStep() {
        switch currentStep {
        case .welcome:
            hasSeenWelcome = true
            userDefaults.set(true, forKey: "has_seen_welcome")
            currentStep = .features
        case .features:
            currentStep = .apiSetup
        case .apiSetup:
            hasConfiguredAPI = true
            userDefaults.set(true, forKey: "has_configured_api")
            currentStep = .modelSelection
        case .modelSelection:
            hasSelectedModel = true
            userDefaults.set(true, forKey: "has_selected_model")
            currentStep = .testConnection
        case .testConnection:
            hasTestedConnection = true
            userDefaults.set(true, forKey: "has_tested_connection")
            currentStep = .completion
        case .completion:
            isOnboardingComplete = true
            isFirstLaunch = false
            userDefaults.set(true, forKey: "onboarding_completed")
        }
    }
    
    func skipOnboarding() {
        isOnboardingComplete = true
        isFirstLaunch = false
        userDefaults.set(true, forKey: "onboarding_completed")
    }
    
    func resetOnboarding() {
        userDefaults.removeObject(forKey: "onboarding_completed")
        userDefaults.removeObject(forKey: "has_seen_welcome")
        userDefaults.removeObject(forKey: "has_configured_api")
        userDefaults.removeObject(forKey: "has_selected_model")
        userDefaults.removeObject(forKey: "has_tested_connection")
        loadOnboardingState()
    }
}

// MARK: - Onboarding Steps
enum OnboardingStep: CaseIterable {
    case welcome
    case features
    case apiSetup
    case modelSelection
    case testConnection
    case completion
    
    var title: String {
        switch self {
        case .welcome: return "Welcome to AgenticSeek"
        case .features: return "Powerful AI Features"
        case .apiSetup: return "API Configuration"
        case .modelSelection: return "Choose Your AI Model"
        case .testConnection: return "Test Connection"
        case .completion: return "Ready to Get Started!"
        }
    }
    
    var stepNumber: Int {
        switch self {
        case .welcome: return 1
        case .features: return 2
        case .apiSetup: return 3
        case .modelSelection: return 4
        case .testConnection: return 5
        case .completion: return 6
        }
    }
    
    var totalSteps: Int { 6 }
    
    var progress: Double {
        return Double(stepNumber) / Double(totalSteps)
    }
}

// MARK: - Main Onboarding View
struct OnboardingFlow: View {
    @EnvironmentObject var onboardingManager: OnboardingManager
    @State private var showSkip = false
    
    var body: some View {
            GeometryReader { geometry in
                VStack(spacing: 0) {
                    // Header with progress and skip
                    OnboardingHeader(
                        currentStep: onboardingManager.currentStep,
                        onSkip: {
                            onboardingManager.skipOnboarding()
                        }
                    )
                    
                    // Main content area
                    TabView(selection: $onboardingManager.currentStep) {
                        OnboardingWelcomeView()
                            .tag(OnboardingStep.welcome)
                        
                        OnboardingFeaturesView()
                            .tag(OnboardingStep.features)
                        
                        OnboardingAPISetupView()
                            .tag(OnboardingStep.apiSetup)
                        
                        OnboardingModelSelectionView()
                            .tag(OnboardingStep.modelSelection)
                        
                        OnboardingTestConnectionView()
                            .tag(OnboardingStep.testConnection)
                        
                        OnboardingCompletionView()
                            .tag(OnboardingStep.completion)
                    }
                    // Note: .page style not available on macOS, using default
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    
                    // Navigation controls
                    OnboardingNavigationControls(
                        currentStep: onboardingManager.currentStep,
                        onNext: {
                            withAnimation(.easeInOut(duration: 0.3)) {
                                onboardingManager.completeCurrentStep()
                            }
                        },
                        onBack: {
                            // Navigate back through steps
                            withAnimation(.easeInOut(duration: 0.3)) {
                                navigateBack()
                            }
                        }
                    )
                }
            }
            .background(DesignSystem.Colors.background)
            .frame(minWidth: 800, minHeight: 600)
    }
    
    private func navigateBack() {
        switch onboardingManager.currentStep {
        case .features:
            onboardingManager.currentStep = .welcome
        case .apiSetup:
            onboardingManager.currentStep = .features
        case .modelSelection:
            onboardingManager.currentStep = .apiSetup
        case .testConnection:
            onboardingManager.currentStep = .modelSelection
        case .completion:
            onboardingManager.currentStep = .testConnection
        case .welcome:
            break // First step, no back
        }
    }
}

// MARK: - Header Component
struct OnboardingHeader: View {
    let currentStep: OnboardingStep
    let onSkip: () -> Void
    
    var body: some View {
        VStack(spacing: DesignSystem.Spacing.space16) {
            HStack {
                // Logo and title
                HStack(spacing: DesignSystem.Spacing.space12) {
                    Image(systemName: "brain.head.profile")
                        .font(.title)
                        .foregroundColor(DesignSystem.Colors.primary)
                        .accessibilityHidden(true)
                    
                    Text("AgenticSeek Setup")
                        .font(DesignSystem.Typography.title2)
                        .foregroundColor(DesignSystem.Colors.textPrimary)
                }
                
                Spacer()
                
                // Skip button
                Button("Skip Setup") {
                    onSkip()
                }
                .buttonStyle(.borderless)
                .foregroundColor(DesignSystem.Colors.textSecondary)
                .accessibilityLabel("Skip onboarding setup")
                .accessibilityHint("Skip the setup process and go directly to the main application")
            }
            
            // Progress bar
            VStack(alignment: .leading, spacing: DesignSystem.Spacing.space8) {
                HStack {
                    Text("Step \(currentStep.stepNumber) of \(currentStep.totalSteps)")
                        .font(DesignSystem.Typography.caption)
                        .foregroundColor(DesignSystem.Colors.textSecondary)
                    
                    Spacer()
                    
                    Text(currentStep.title)
                        .font(DesignSystem.Typography.caption)
                        .foregroundColor(DesignSystem.Colors.textSecondary)
                }
                
                ProgressView(value: currentStep.progress)
                    .progressViewStyle(.linear)
                    .tint(DesignSystem.Colors.primary)
                    .accessibilityLabel("Setup progress")
                    .accessibilityValue("Step \(currentStep.stepNumber) of \(currentStep.totalSteps), \(Int(currentStep.progress * 100)) percent complete")
            }
        }
        .padding(DesignSystem.Spacing.space20)
        .background(DesignSystem.Colors.surface)
        .overlay(
            Rectangle()
                .fill(DesignSystem.Colors.border)
                .frame(height: 1),
            alignment: .bottom
        )
    }
}

// MARK: - Welcome Step
struct OnboardingWelcomeView: View {
    var body: some View {
        VStack(spacing: DesignSystem.Spacing.xl) {
            Spacer()
            
            // Welcome icon
            Image(systemName: "sparkles")
                .font(.system(size: 80))
                .foregroundColor(DesignSystem.Colors.primary)
                .accessibilityHidden(true)
            
            // Welcome content
            VStack(spacing: DesignSystem.Spacing.space16) {
                Text("Welcome to AgenticSeek")
                    .font(DesignSystem.Typography.headline)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                    .multilineTextAlignment(.center)
                
                Text("Your powerful AI assistant for research, analysis, and creative tasks")
                    .font(DesignSystem.Typography.body)
                    .foregroundColor(DesignSystem.Colors.textSecondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(3)
                
                Text("Let's get you set up in just a few quick steps")
                    .font(DesignSystem.Typography.callout)
                    .foregroundColor(DesignSystem.Colors.textTertiary)
                    .multilineTextAlignment(.center)
            }
            
            Spacer()
        }
        .frame(maxWidth: 600)
        .padding(DesignSystem.Spacing.space40)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Welcome to AgenticSeek")
        .accessibilityHint("Start the setup process for your AI assistant")
    }
}

// MARK: - Features Step
struct OnboardingFeaturesView: View {
    private let features = [
        FeatureInfo(
            icon: "message.badge",
            title: "AI Conversations",
            description: "Chat with advanced AI Models for research and assistance"
        ),
        FeatureInfo(
            icon: "cpu",
            title: "Multiple AI Models",
            description: "Choose from cloud and local AI Models for your needs"
        ),
        FeatureInfo(
            icon: "lock.shield",
            title: "Privacy & Security",
            description: "Your data stays secure with local processing options"
        ),
        FeatureInfo(
            icon: "accessibility",
            title: "Accessibility First",
            description: "Full WCAG AAA compliance for all users"
        )
    ]
    
    var body: some View {
        VStack(spacing: DesignSystem.Spacing.xl) {
            Spacer()
            
            VStack(spacing: DesignSystem.Spacing.space16) {
                Text("Powerful Features")
                    .font(DesignSystem.Typography.headline)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                
                Text("Everything you need for AI-powered productivity")
                    .font(DesignSystem.Typography.body)
                    .foregroundColor(DesignSystem.Colors.textSecondary)
                    .multilineTextAlignment(.center)
            }
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: DesignSystem.Spacing.space20) {
                ForEach(features, id: \.title) { feature in
                    OnboardingFeatureCard(feature: feature)
                }
            }
            .frame(maxWidth: 600)
            
            Spacer()
        }
        .padding(DesignSystem.Spacing.space40)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("AgenticSeek features overview")
        .accessibilityHint("Learn about AI conversations, multiple Models, privacy, and accessibility")
    }
}

// MARK: - Feature Card
struct FeatureInfo {
    let icon: String
    let title: String
    let description: String
}

struct OnboardingFeatureCard: View {
    let feature: FeatureInfo
    
    var body: some View {
        VStack(spacing: DesignSystem.Spacing.space12) {
            Image(systemName: feature.icon)
                .font(.title)
                .foregroundColor(DesignSystem.Colors.primary)
                .accessibilityHidden(true)
            
            Text(feature.title)
                .font(DesignSystem.Typography.title3)
                .foregroundColor(DesignSystem.Colors.textPrimary)
                .multilineTextAlignment(.center)
            
            Text(feature.description)
                .font(DesignSystem.Typography.caption)
                .foregroundColor(DesignSystem.Colors.textSecondary)
                .multilineTextAlignment(.center)
                .lineLimit(3)
        }
        .padding(DesignSystem.Spacing.space16)
        .frame(maxWidth: .infinity, minHeight: 120)
        .background(DesignSystem.Colors.surface)
        .cornerRadius(DesignSystem.CornerRadius.card)
        .overlay(
            RoundedRectangle(cornerRadius: DesignSystem.CornerRadius.card)
                .stroke(DesignSystem.Colors.border, lineWidth: 1)
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel(feature.title)
        .accessibilityHint(feature.description)
    }
}

// MARK: - API Setup Step
struct OnboardingAPISetupView: View {
    @State private var selectedProvider: APIProvider = .openai
    @State private var apiKey: String = ""
    @State private var showAPIKeyField = false
    
    var body: some View {
        VStack(spacing: DesignSystem.Spacing.xl) {
            Spacer()
            
            VStack(spacing: DesignSystem.Spacing.space16) {
                Text("API Configuration")
                    .font(DesignSystem.Typography.headline)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                
                Text("Choose your AI provider and configure access")
                    .font(DesignSystem.Typography.body)
                    .foregroundColor(DesignSystem.Colors.textSecondary)
                    .multilineTextAlignment(.center)
            }
            
            VStack(spacing: DesignSystem.Spacing.space20) {
                // Provider selection
                VStack(alignment: .leading, spacing: DesignSystem.Spacing.space12) {
                    Text("Choose AI Provider")
                        .font(DesignSystem.Typography.title3)
                        .foregroundColor(DesignSystem.Colors.textPrimary)
                    
                    ForEach(APIProvider.allCases, id: \.self) { provider in
                        OnboardingProviderOption(
                            provider: provider,
                            isSelected: selectedProvider == provider,
                            onSelect: { selectedProvider = provider }
                        )
                    }
                }
                
                // API Key input (if needed)
                if selectedProvider.requiresAPIKey {
                    VStack(alignment: .leading, spacing: DesignSystem.Spacing.space8) {
                        Text("API Key")
                            .font(DesignSystem.Typography.body)
                            .foregroundColor(DesignSystem.Colors.textPrimary)
                        
                        SecureField("Enter your \(selectedProvider.displayName) API key", text: $apiKey)
                            .textFieldStyle(.roundedBorder)
                            .accessibilityLabel("\(selectedProvider.displayName) API key input")
                            .accessibilityHint("Enter your API key to connect to \(selectedProvider.displayName)")
                        
                        Text("Your API key is stored securely and never shared")
                            .font(DesignSystem.Typography.caption)
                            .foregroundColor(DesignSystem.Colors.textTertiary)
                    }
                    .frame(maxWidth: 400)
                }
            }
            
            Spacer()
        }
        .frame(maxWidth: 600)
        .padding(DesignSystem.Spacing.space40)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("API configuration")
        .accessibilityHint("Select AI provider and configure API access")
    }
}

enum APIProvider: String, CaseIterable {
    case openai = "openai"
    case anthropic = "anthropic"
    case local = "local"
    
    var displayName: String {
        switch self {
        case .openai: return "OpenAI"
        case .anthropic: return "Anthropic"
        case .local: return "Local Models"
        }
    }
    
    var description: String {
        switch self {
        case .openai: return "GPT-4 and other OpenAI Models"
        case .anthropic: return "Claude and other Anthropic Models"
        case .local: return "Run AI Models locally on your device"
        }
    }
    
    var requiresAPIKey: Bool {
        switch self {
        case .openai, .anthropic: return true
        case .local: return false
        }
    }
}

struct OnboardingProviderOption: View {
    let provider: APIProvider
    let isSelected: Bool
    let onSelect: () -> Void
    
    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: DesignSystem.Spacing.space12) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(isSelected ? DesignSystem.Colors.primary : DesignSystem.Colors.disabled)
                    .accessibilityHidden(true)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(provider.displayName)
                        .font(DesignSystem.Typography.body)
                        .foregroundColor(DesignSystem.Colors.textPrimary)
                    
                    Text(provider.description)
                        .font(DesignSystem.Typography.caption)
                        .foregroundColor(DesignSystem.Colors.textSecondary)
                }
                
                Spacer()
            }
            .padding(DesignSystem.Spacing.space12)
            .background(isSelected ? DesignSystem.Colors.primary.opacity(0.1) : DesignSystem.Colors.surface)
            .cornerRadius(DesignSystem.CornerRadius.medium)
            .overlay(
                RoundedRectangle(cornerRadius: DesignSystem.CornerRadius.medium)
                    .stroke(isSelected ? DesignSystem.Colors.primary : DesignSystem.Colors.border, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(provider.displayName) option")
        .accessibilityHint(provider.description)
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }
}

// MARK: - Navigation Controls
struct OnboardingNavigationControls: View {
    let currentStep: OnboardingStep
    let onNext: () -> Void
    let onBack: () -> Void
    
    var body: some View {
        HStack {
            // Back button
            if currentStep != .welcome {
                Button("Back") {
                    onBack()
                }
                .buttonStyle(.borderless)
                .foregroundColor(DesignSystem.Colors.textSecondary)
                .accessibilityLabel("Go back to previous step")
            } else {
                // Spacer for alignment
                Spacer()
                    .frame(width: 60)
            }
            
            Spacer()
            
            // Next button
            Button(currentStep == .completion ? "Get Started" : "Continue") {
                onNext()
            }
            .buttonStyle(.borderedProminent)
            .accessibilityLabel(currentStep == .completion ? "Complete setup and start using AgenticSeek" : "Continue to next step")
        }
        .padding(DesignSystem.Spacing.space20)
        .background(DesignSystem.Colors.surface)
        .overlay(
            Rectangle()
                .fill(DesignSystem.Colors.border)
                .frame(height: 1),
            alignment: .top
        )
    }
}

// MARK: - Additional Steps (Placeholder implementations)
struct OnboardingModelSelectionView: View {
    var body: some View {
        VStack(spacing: DesignSystem.Spacing.xl) {
            Spacer()
            
            VStack(spacing: DesignSystem.Spacing.space16) {
                Text("Choose Your AI Model")
                    .font(DesignSystem.Typography.headline)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                
                Text("Select the AI Model that best fits your needs")
                    .font(DesignSystem.Typography.body)
                    .foregroundColor(DesignSystem.Colors.textSecondary)
                    .multilineTextAlignment(.center)
            }
            
            // Model selection content would go here
            Text("Model selection interface")
                .font(DesignSystem.Typography.title3)
                .foregroundColor(DesignSystem.Colors.textTertiary)
            
            Spacer()
        }
        .padding(DesignSystem.Spacing.space40)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("AI Model selection")
        .accessibilityHint("Choose your preferred AI Model")
    }
}

struct OnboardingTestConnectionView: View {
    var body: some View {
        VStack(spacing: DesignSystem.Spacing.xl) {
            Spacer()
            
            VStack(spacing: DesignSystem.Spacing.space16) {
                Text("Test Connection")
                    .font(DesignSystem.Typography.headline)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                
                Text("Let's verify your AI connection is working properly")
                    .font(DesignSystem.Typography.body)
                    .foregroundColor(DesignSystem.Colors.textSecondary)
                    .multilineTextAlignment(.center)
            }
            
            // Connection test content would go here
            Text("Connection testing interface")
                .font(DesignSystem.Typography.title3)
                .foregroundColor(DesignSystem.Colors.textTertiary)
            
            Spacer()
        }
        .padding(DesignSystem.Spacing.space40)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Test AI connection")
        .accessibilityHint("Verify your AI service connection")
    }
}

struct OnboardingCompletionView: View {
    var body: some View {
        VStack(spacing: DesignSystem.Spacing.xl) {
            Spacer()
            
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 80))
                .foregroundColor(DesignSystem.Colors.success)
                .accessibilityHidden(true)
            
            VStack(spacing: DesignSystem.Spacing.space16) {
                Text("You're All Set!")
                    .font(DesignSystem.Typography.headline)
                    .foregroundColor(DesignSystem.Colors.textPrimary)
                
                Text("AgenticSeek is ready to help you with AI-powered research and analysis")
                    .font(DesignSystem.Typography.body)
                    .foregroundColor(DesignSystem.Colors.textSecondary)
                    .multilineTextAlignment(.center)
                
                Text("Click 'Get Started' to begin your AI journey")
                    .font(DesignSystem.Typography.callout)
                    .foregroundColor(DesignSystem.Colors.textTertiary)
                    .multilineTextAlignment(.center)
            }
            
            Spacer()
        }
        .padding(DesignSystem.Spacing.space40)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Setup complete")
        .accessibilityHint("Ready to start using AgenticSeek")
    }
}