// SANDBOX FILE: For testing/development. See .cursorrules.
//
// ProductionUIModals.swift
// AgenticSeek
//
// ATOMIC TDD GREEN PHASE: Production UI/UX with Working Modals
// Production-grade SwiftUI modal components with accessibility and state management
//

import SwiftUI
import Combine

// MARK: - Modal State Management

/// Production-grade modal state manager
class ModalStateManager: ObservableObject {
    @Published var isSettingsPresented = false
    @Published var isPreferencesPresented = false
    @Published var isAboutPresented = false
    @Published var isHelpPresented = false
    @Published var currentModal: ModalType?
    
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        // Monitor modal state changes
        $isSettingsPresented
            .combineLatest($isPreferencesPresented, $isAboutPresented, $isHelpPresented)
            .map { settings, preferences, about, help in
                if settings { return .settings }
                if preferences { return .preferences }
                if about { return .about }
                if help { return .help }
                return nil
            }
            .assign(to: \.currentModal, on: self)
            .store(in: &cancellables)
    }
    
    func presentModal(_ type: ModalType) {
        dismissAllModals()
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            switch type {
            case .settings:
                self.isSettingsPresented = true
            case .preferences:
                self.isPreferencesPresented = true
            case .about:
                self.isAboutPresented = true
            case .help:
                self.isHelpPresented = true
            default:
                break
            }
        }
    }
    
    func dismissAllModals() {
        isSettingsPresented = false
        isPreferencesPresented = false
        isAboutPresented = false
        isHelpPresented = false
    }
}

/// Modal types for type-safe presentation
enum ModalType: String, CaseIterable {
    case settings = "settings"
    case preferences = "preferences"
    case about = "about"
    case help = "help"
    case errorDialog = "error_dialog"
    case confirmation = "confirmation"
    case loading = "loading"
    case customAction = "custom_action"
    
    var title: String {
        switch self {
        case .settings: return "Settings"
        case .preferences: return "Preferences"
        case .about: return "About AgenticSeek"
        case .help: return "Help & Support"
        case .errorDialog: return "Error"
        case .confirmation: return "Confirm Action"
        case .loading: return "Loading"
        case .customAction: return "Action Required"
        }
    }
    
    var accessibilityLabel: String {
        switch self {
        case .settings: return "Settings modal"
        case .preferences: return "Preferences modal"
        case .about: return "About information modal"
        case .help: return "Help and support modal"
        case .errorDialog: return "Error dialog"
        case .confirmation: return "Confirmation dialog"
        case .loading: return "Loading dialog"
        case .customAction: return "Custom action dialog"
        }
    }
}

// MARK: - Production Button Components

/// Primary action button with production styling
struct PrimaryActionButton: View {
    let title: String
    let action: () -> Void
    var isEnabled: Bool = true
    var isLoading: Bool = false
    
    @State private var isPressed = false
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if isLoading {
                    ProgressView()
                        .scaleEffect(0.8)
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                }
                
                Text(title)
                    .font(.system(.body, design: .rounded, weight: .semibold))
                    .foregroundColor(.white)
            }
            .frame(minWidth: 120, minHeight: 44)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(isEnabled ? Color.accentColor : Color.gray)
                    .scaleEffect(isPressed ? 0.95 : 1.0)
            )
        }
        .disabled(!isEnabled || isLoading)
        .scaleEffect(isPressed ? 0.95 : 1.0)
        .onLongPressGesture(minimumDuration: 0, maximumDistance: .infinity) { _ in
            withAnimation(.easeInOut(duration: 0.1)) {
                isPressed = true
            }
        } onPressingChanged: { pressing in
            withAnimation(.easeInOut(duration: 0.1)) {
                isPressed = pressing
            }
        }
        .accessibilityLabel("\(title) button")
        .accessibilityHint(isLoading ? "Loading in progress" : "Double tap to activate")
        .accessibilityAddTraits(isEnabled ? .isButton : [.isButton, .isNotEnabled])
    }
}

/// Secondary action button with production styling
struct SecondaryActionButton: View {
    let title: String
    let action: () -> Void
    var isEnabled: Bool = true
    
    @State private var isPressed = false
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(.body, design: .rounded, weight: .medium))
                .foregroundColor(.accentColor)
                .frame(minWidth: 120, minHeight: 44)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.accentColor, lineWidth: 2)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.clear)
                        )
                        .scaleEffect(isPressed ? 0.95 : 1.0)
                )
        }
        .disabled(!isEnabled)
        .scaleEffect(isPressed ? 0.95 : 1.0)
        .onLongPressGesture(minimumDuration: 0, maximumDistance: .infinity) { _ in
            withAnimation(.easeInOut(duration: 0.1)) {
                isPressed = true
            }
        } onPressingChanged: { pressing in
            withAnimation(.easeInOut(duration: 0.1)) {
                isPressed = pressing
            }
        }
        .accessibilityLabel("\(title) button")
        .accessibilityHint("Double tap to activate")
        .accessibilityAddTraits(isEnabled ? .isButton : [.isButton, .isNotEnabled])
    }
}

/// Destructive action button with production styling
struct DestructiveActionButton: View {
    let title: String
    let action: () -> Void
    var isEnabled: Bool = true
    
    @State private var isPressed = false
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(.body, design: .rounded, weight: .semibold))
                .foregroundColor(.white)
                .frame(minWidth: 120, minHeight: 44)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(isEnabled ? Color.red : Color.gray)
                        .scaleEffect(isPressed ? 0.95 : 1.0)
                )
        }
        .disabled(!isEnabled)
        .scaleEffect(isPressed ? 0.95 : 1.0)
        .onLongPressGesture(minimumDuration: 0, maximumDistance: .infinity) { _ in
            withAnimation(.easeInOut(duration: 0.1)) {
                isPressed = true
            }
        } onPressingChanged: { pressing in
            withAnimation(.easeInOut(duration: 0.1)) {
                isPressed = pressing
            }
        }
        .accessibilityLabel("\(title) button")
        .accessibilityHint("Destructive action - double tap to activate")
        .accessibilityAddTraits(isEnabled ? .isButton : [.isButton, .isNotEnabled])
    }
}

// MARK: - Settings Panel Components

/// Model selection panel for settings
struct ModelSelectionPanel: View {
    @Binding var selectedModel: String
    let availableModels = ["Claude-3-Sonnet", "GPT-4", "Claude-3-Opus", "GPT-4-Turbo"]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("AI Model")
                .font(.system(.headline, design: .rounded, weight: .semibold))
                .accessibilityAddTraits(.isHeader)
            
            Picker("Select AI Model", selection: $selectedModel) {
                ForEach(availableModels, id: \.self) { model in
                    Text(model)
                        .tag(model)
                }
            }
            .pickerStyle(MenuPickerStyle())
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityLabel("AI Model selection")
            .accessibilityValue("Currently selected: \(selectedModel)")
        }
        .padding(.vertical, 8)
    }
}

/// Voice settings panel
struct VoiceSettingsPanel: View {
    @Binding var voiceEnabled: Bool
    @Binding var voiceSpeed: Double
    @Binding var voiceVolume: Double
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Voice Settings")
                .font(.system(.headline, design: .rounded, weight: .semibold))
                .accessibilityAddTraits(.isHeader)
            
            Toggle("Enable Voice Assistant", isOn: $voiceEnabled)
                .toggleStyle(SwitchToggleStyle())
                .accessibilityLabel("Voice Assistant")
                .accessibilityValue(voiceEnabled ? "Enabled" : "Disabled")
            
            if voiceEnabled {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Speech Speed")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Slider(value: $voiceSpeed, in: 0.5...2.0, step: 0.1) {
                        Text("Speech Speed")
                    }
                    .accessibilityLabel("Speech speed")
                    .accessibilityValue("Speed: \(String(format: "%.1f", voiceSpeed))")
                    
                    Text("Volume")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Slider(value: $voiceVolume, in: 0.0...1.0, step: 0.1) {
                        Text("Volume")
                    }
                    .accessibilityLabel("Voice volume")
                    .accessibilityValue("Volume: \(String(format: "%.0f", voiceVolume * 100)) percent")
                }
                .transition(.opacity.combined(with: .slide))
            }
        }
        .animation(.easeInOut(duration: 0.3), value: voiceEnabled)
        .padding(.vertical, 8)
    }
}

/// Performance settings panel
struct PerformanceSettingsPanel: View {
    @Binding var maxTokens: Double
    @Binding var temperature: Double
    @Binding var streamingEnabled: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Performance Settings")
                .font(.system(.headline, design: .rounded, weight: .semibold))
                .accessibilityAddTraits(.isHeader)
            
            VStack(alignment: .leading, spacing: 12) {
                Text("Max Tokens: \(Int(maxTokens))")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Slider(value: $maxTokens, in: 100...4000, step: 100) {
                    Text("Max Tokens")
                }
                .accessibilityLabel("Maximum tokens")
                .accessibilityValue("Tokens: \(Int(maxTokens))")
                
                Text("Temperature: \(String(format: "%.1f", temperature))")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Slider(value: $temperature, in: 0.0...2.0, step: 0.1) {
                    Text("Temperature")
                }
                .accessibilityLabel("Response creativity")
                .accessibilityValue("Temperature: \(String(format: "%.1f", temperature))")
                
                Toggle("Enable Streaming", isOn: $streamingEnabled)
                    .toggleStyle(SwitchToggleStyle())
                    .accessibilityLabel("Response streaming")
                    .accessibilityValue(streamingEnabled ? "Enabled" : "Disabled")
            }
        }
        .padding(.vertical, 8)
    }
}

// MARK: - Main Modal Views

/// Production settings modal view
struct SettingsModalView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var modalState = ModalStateManager()
    
    @State private var selectedModel = "Claude-3-Sonnet"
    @State private var voiceEnabled = true
    @State private var voiceSpeed = 1.0
    @State private var voiceVolume = 0.8
    @State private var maxTokens = 2000.0
    @State private var temperature = 0.7
    @State private var streamingEnabled = true
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    ModelSelectionPanel(selectedModel: $selectedModel)
                    
                    Divider()
                    
                    VoiceSettingsPanel(
                        voiceEnabled: $voiceEnabled,
                        voiceSpeed: $voiceSpeed,
                        voiceVolume: $voiceVolume
                    )
                    
                    Divider()
                    
                    PerformanceSettingsPanel(
                        maxTokens: $maxTokens,
                        temperature: $temperature,
                        streamingEnabled: $streamingEnabled
                    )
                }
                .padding()
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    SecondaryActionButton(title: "Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    PrimaryActionButton(title: "Save") {
                        saveSettings()
                        dismiss()
                    }
                }
            }
        }
        .accessibilityLabel("Settings modal")
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }
    
    private func saveSettings() {
        // Production settings save logic would go here
        print("Settings saved: Model=\(selectedModel), Voice=\(voiceEnabled)")
    }
}

/// Production preferences modal view
struct PreferencesModalView: View {
    @Environment(\.dismiss) private var dismiss
    
    @State private var appearance = "Auto"
    @State private var language = "English"
    @State private var notifications = true
    @State private var analyticsEnabled = false
    @State private var autoSave = true
    
    let appearances = ["Auto", "Light", "Dark"]
    let languages = ["English", "Spanish", "French", "German", "Japanese"]
    
    var body: some View {
        NavigationView {
            Form {
                Section("Appearance") {
                    Picker("Theme", selection: $appearance) {
                        ForEach(appearances, id: \.self) { theme in
                            Text(theme).tag(theme)
                        }
                    }
                    .accessibilityLabel("Theme selection")
                    .accessibilityValue("Currently: \(appearance)")
                }
                
                Section("Language & Region") {
                    Picker("Language", selection: $language) {
                        ForEach(languages, id: \.self) { lang in
                            Text(lang).tag(lang)
                        }
                    }
                    .accessibilityLabel("Language selection")
                    .accessibilityValue("Currently: \(language)")
                }
                
                Section("Privacy & Data") {
                    Toggle("Enable Notifications", isOn: $notifications)
                        .accessibilityLabel("Notifications")
                        .accessibilityValue(notifications ? "Enabled" : "Disabled")
                    
                    Toggle("Analytics & Insights", isOn: $analyticsEnabled)
                        .accessibilityLabel("Analytics and insights")
                        .accessibilityValue(analyticsEnabled ? "Enabled" : "Disabled")
                    
                    Toggle("Auto-save Conversations", isOn: $autoSave)
                        .accessibilityLabel("Auto-save conversations")
                        .accessibilityValue(autoSave ? "Enabled" : "Disabled")
                }
            }
            .navigationTitle("Preferences")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    SecondaryActionButton(title: "Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    PrimaryActionButton(title: "Save") {
                        savePreferences()
                        dismiss()
                    }
                }
            }
        }
        .accessibilityLabel("Preferences modal")
        .presentationDetents([.large])
        .presentationDragIndicator(.visible)
    }
    
    private func savePreferences() {
        // Production preferences save logic would go here
        print("Preferences saved: Theme=\(appearance), Language=\(language)")
    }
}

// MARK: - Input Validation System

/// Text field with production validation
struct ValidatedTextField: View {
    let title: String
    @Binding var text: String
    let validator: (String) -> ValidationResult
    
    @State private var validationResult = ValidationResult.valid
    @FocusState private var isFocused: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField(title, text: $text)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .focused($isFocused)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(validationResult.borderColor, lineWidth: 1)
                )
                .onChange(of: text) { newValue in
                    validationResult = validator(newValue)
                }
                .accessibilityLabel(title)
                .accessibilityValue(text.isEmpty ? "Empty" : "Filled")
                .accessibilityHint(validationResult.accessibilityHint)
            
            if case .invalid(let message) = validationResult {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.circle.fill")
                        .foregroundColor(.red)
                        .font(.caption)
                    
                    Text(message)
                        .font(.caption)
                        .foregroundColor(.red)
                }
                .accessibilityLabel("Validation error: \(message)")
            }
        }
    }
}

/// Validation result for form inputs
enum ValidationResult {
    case valid
    case invalid(String)
    
    var borderColor: Color {
        switch self {
        case .valid: return .gray.opacity(0.3)
        case .invalid: return .red
        }
    }
    
    var accessibilityHint: String {
        switch self {
        case .valid: return "Input is valid"
        case .invalid(let message): return "Input is invalid: \(message)"
        }
    }
}

// MARK: - Animation Controllers

/// Modal transition animator
struct ModalTransitionAnimator {
    static func slideUpTransition() -> AnyTransition {
        .asymmetric(
            insertion: .move(edge: .bottom).combined(with: .opacity),
            removal: .move(edge: .bottom).combined(with: .opacity)
        )
    }
    
    static func fadeTransition() -> AnyTransition {
        .opacity
    }
    
    static func scaleTransition() -> AnyTransition {
        .scale(scale: 0.8).combined(with: .opacity)
    }
    
    static func springAnimation() -> Animation {
        .spring(response: 0.6, dampingFraction: 0.8, blendDuration: 0.3)
    }
}

// MARK: - Production UI Coordinator

/// Coordinates all production UI components
class ProductionUICoordinator: ObservableObject {
    @Published var modalStateManager = ModalStateManager()
    @Published var themeManager = ThemeManager()
    @Published var accessibilityManager = AccessibilityManager()
    
    init() {
        setupCoordination()
    }
    
    private func setupCoordination() {
        // Coordinate between different managers
        themeManager.objectWillChange
            .sink { [weak self] _ in
                self?.objectWillChange.send()
            }
            .store(in: &cancellables)
    }
    
    private var cancellables = Set<AnyCancellable>()
}

/// Theme management for production UI
class ThemeManager: ObservableObject {
    @Published var currentTheme: AppTheme = .auto
    @Published var accentColor: Color = .blue
    @Published var cornerRadius: CGFloat = 12
    
    enum AppTheme: String, CaseIterable {
        case light = "light"
        case dark = "dark"
        case auto = "auto"
        
        var displayName: String {
            switch self {
            case .light: return "Light"
            case .dark: return "Dark"
            case .auto: return "Auto"
            }
        }
    }
}

/// Accessibility management for production UI
class AccessibilityManager: ObservableObject {
    @Published var voiceOverEnabled = false
    @Published var dynamicTypeEnabled = true
    @Published var highContrastEnabled = false
    @Published var reduceMotionEnabled = false
    
    init() {
        updateAccessibilitySettings()
    }
    
    private func updateAccessibilitySettings() {
        voiceOverEnabled = UIAccessibility.isVoiceOverRunning
        reduceMotionEnabled = UIAccessibility.isReduceMotionEnabled
    }
}

// MARK: - Helper Extensions

extension View {
    func productionAccessibility(
        label: String,
        hint: String? = nil,
        value: String? = nil,
        traits: AccessibilityTraits = []
    ) -> some View {
        self
            .accessibilityLabel(label)
            .accessibilityHint(hint ?? "")
            .accessibilityValue(value ?? "")
            .accessibilityAddTraits(traits)
    }
}

// MARK: - Preview Providers

#if DEBUG
struct ProductionUIModals_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            SettingsModalView()
                .previewDisplayName("Settings Modal")
            
            PreferencesModalView()
                .previewDisplayName("Preferences Modal")
            
            VStack(spacing: 20) {
                PrimaryActionButton(title: "Primary Action") {}
                SecondaryActionButton(title: "Secondary Action") {}
                DestructiveActionButton(title: "Delete") {}
            }
            .padding()
            .previewDisplayName("Production Buttons")
        }
    }
}
#endif