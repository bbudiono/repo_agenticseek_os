//
// Strings.swift
// AgenticSeek
//
// Purpose: Centralizes all string literals used throughout the application to facilitate localization and consistency.
// Issues & Complexity: High (Centralization), Low (Technical)
// Ranking/Rating: 95% (Code), 90% (Problem)
//
// Key Complexity Drivers:
// - Logic Scope: ~50-100 LoC (initial)
// - Core Algorithm Complexity: N/A
// - Dependencies: N/A
// - State Management Complexity: N/A
// - Novelty/Uncertainty Factor: Low
//
// AI Pre-Task Self-Assessment (Est. Solution Difficulty for AI %): 85%
// Problem Estimate (Inherent Problem Difficulty %): 80% (Relative to other problems in _macOS/AgenticSeek/)
// Initial Code Complexity Estimate (Est. Code Difficulty %): 80% (Relative to other files in _macOS/AgenticSeek/)
// Justification for Estimates: Straightforward task of extracting and organizing strings.
//
// Last Updated: 2025-05-31
//
import Foundation

public struct Strings {
    public struct General {
        public static let ok = "OK"
        public static let cancel = "Cancel"
        public static let save = "Save"
        public static let dismiss = "Dismiss"
        public static let apply = "Apply"
        public static let update = "Update"
        public static let add = "Add"
        public static let selected = "Selected"
        public static let select = "Select"
        public static let details = "Details"
        public static let refreshModels = "Refresh Models"
        public static let refreshAll = "Refresh All"
        public static let loading = "Loading..."
        public static let active = "Active"
        public static let model = "Model"
        public static let noAPIKeys = "No API Keys"
        public static let apiKeysWillAppearHere = "API keys will appear here"
        public static let apiKey = "API Key"
        public static let enterYourApiKey = "Enter your API key"
        public static let yourApiKeyStoredSecurely = "Your API key will be stored securely."
        public static let configuration = "Configuration"
        public static let providers = "Providers"
        public static let apiKeys = "API Keys"
        public static let success = "Success"
        public static let failure = "Failure"
        public static let currentModel = "Current Model"
        public static let searchModels = "Search Models"
        public static let typeToSearchModels = "Type to search for specific models"
        public static let filterByCapability = "Filter by Capability"
        public static let filterModelsByCapabilities = "Filter models by their capabilities"
        public static let recommendations = "Recommendations"
        public static let showModelRecommendations = "Show model recommendations for current task"
        public static let loadingAvailableModels = "Loading Available Models..."
        public static let loadingModelsFromCatalog = "Loading models from catalog"
        public static let availableModelsGrid = "Available Models Grid"
        public static let recommendedModels = "Recommended Models"
        public static let forCapability = "For %@"
        public static let selectedModel = "Selected: %@"
        public static let noModelSelected = "No model selected"
        public static let applySelectedModel = "Apply the selected model to current agent"
        public static let byAuthor = "by %@"
        public static let memory = "Memory"
        public static let size = "Size"
        public static let framework = "Framework"
        public static let any = "Any"
    }

    public struct ErrorMessages {
        public static let invalidURL = "Invalid URL"
        public static let serverError = "Server error"
        public static let invalidResponse = "Invalid response"
        public static let failedToLoad = "Failed to load: %@"
        public static let failedToUpdateProvider = "Failed to update provider"
        public static let failedToUpdateAPIKey = "Failed to update API key"
        public static let failedToLoadModels = "Failed to load models: %@"
        public static let invalidResponseFormat = "Invalid response format for %@"
        public static let httpError = "HTTP error %d for %@"
        public static let error = "Error: %@"
        public static let failedToFetchModels = "Failed to fetch models"
        public static let modelValidationFailed = "Model validation failed: %@"
        public static let failedToDownloadModelMetadata = "Failed to download model metadata: %@"
        public static let networkError = "Network error: %@"
        public static let insufficientMemory = "Insufficient memory: requires %@GB, available %@GB"
        public static let incompatibleArchitecture = "Incompatible architecture: %@"
        public static let incompatibleFramework = "Incompatible framework: %@"
        public static let cacheError = "Cache error: %@"
    }

    public struct Accessibility {
        public static let modelSelectionInterface = "Model Selection Interface"
        public static let currentModel = "Current Model: %@"
        public static let searchModels = "Search Models"
        public static let filterByCapability = "Filter by Capability"
        public static let modelRecommendations = "Model Recommendations"
        public static let modelName = "Model: %@"
        public static let doubleTapToSelect = "Double tap to select, or swipe up for details"
        public static let viewDetailedModelInformation = "View detailed model information"
    }
    
    public struct Providers {
        public static let anthropic = "Anthropic"
        public static let openai = "OpenAI"
        public static let deepseek = "Deepseek"
        public static let google = "Google"
        public static let lmStudio = "LM Studio"
        public static let ollama = "Ollama"
    }
    
    public struct ModelNames {
        public static let llama3_2_3b = "llama3:2-3b"
        public static let qwen2_5_7b = "qwen2:5-7b"
        public static let dialoGPTMedium = "dialoGPT-medium"
        public static let dialoGPTLarge = "dialoGPT-large"
    }
    
    public struct DisplayNames {
        public static let llama3_2_3b = "Llama3 2.3B"
        public static let qwen2_5_7b = "Qwen2 5.7B"
        public static let dialoGPTMedium = "DialoGPT Medium"
        public static let dialoGPTLarge = "DialoGPT Large"
    }
    
    public struct ModelDescriptions {
        public static let llama3_2_3b = "A lightweight, fast, and powerful language model."
        public static let qwen2_5_7b = "A versatile model with strong performance in various tasks."
        public static let dialoGPTMedium = "A medium-sized conversational AI model."
        public static let dialoGPTLarge = "A larger, more capable conversational AI model."
    }
    
    public struct DownloadURLs {
        public static let ollamaLlama3_2_3b = "https://ollama.ai/download/llama3:2-3b"
        public static let ollamaQwen2_5_7b = "https://ollama.ai/download/qwen2:5-7b"
        public static let huggingfaceDialoGPTMedium = "https://huggingface.co/microsoft/DialoGPT-medium"
        public static let huggingfaceDialoGPTLarge = "https://huggingface.co/microsoft/DialoGPT-large"
    }
    
    public struct TestStatus {
        public static let pass = "Passed"
        public static let fail = "Failed"
    }
    
    public struct TestDescriptions {
        public static let backendConnection = "Backend Connection Test"
        public static let providerAPIs = "Provider API Tests"
        public static let modelLoading = "Model Loading Test"
        public static let apiKeyManagement = "API Key Management Test"
        public static let agentFunctionality = "Agent Functionality Test"
        public static let uiResponsiveness = "UI Responsiveness Test"
        public static let errorHandling = "Error Handling Test"
        public static let performanceMetrics = "Performance Metrics Collection"
    }
    
    public struct TestNumbers {
        public static let totalTests = 7
    }
    
    public struct Constants {
        public static let localServerURL = "http://127.0.0.1:8000"
    }
    
    public struct SystemTests {
        public static let systemTests = "System Tests"
    }
    
    public struct ConfigurationView {
        public static let clearCache = "Clear Cache"
        public static let deleteModel = "Delete Model"
        public static let downloadModel = "Download Model"
        public static let manageLocalModels = "Manage Local Models"
    }
    
    public struct Preferences {
        public static let assistantVoicePitch = "Assistant Voice Pitch"
        public static let assistantVoiceSpeed = "Assistant Voice Speed"
        public static let chatTemperature = "Chat Temperature"
        public static let maxTokens = "Max Tokens"
        public static let openAIApiKey = "OpenAI API Key"
        public static let ollamaApiUrl = "Ollama API URL"
        public static let deepseekApiKey = "Deepseek API Key"
        public static let googleApiKey = "Google API Key"
        public static let anthropicApiKey = "Anthropic API Key"
        public static let lmStudioApiUrl = "LM Studio API URL"
        public static let localLMStudioModels = "Local LM Studio Models"
        public static let ollamaModels = "Ollama Models"
        public static let clearDownloadedModels = "Clear Downloaded Models"
        public static let settings = "Settings"
    }

    public struct Chat {
        public static let chat = "Chat"
        public static let modelConfiguration = "Model Configuration"
        public static let sendMessage = "Send Message"
        public static let enterYourMessage = "Enter your message..."
        public static let responseGenerating = "Response generating..."
        public static let messageHistory = "Message History"
        public static let noAgentSelected = "No Agent Selected"
        public static let selectAnAgent = "Select an Agent to Begin Chatting"
        public static let startNewChat = "Start New Chat"
    }
} 
