import Foundation
import Combine

// MARK: - Model Catalog Service
// Business logic service extracted from ChatConfigurationManager (ContentView lines 193-494)
// Handles model discovery, validation, and catalog management
// Pure business logic - no UI dependencies

protocol ModelCatalogServiceProtocol {
    func fetchAvailableModels() async throws -> [ModelInfo]
    func validateModelCompatibility(_ model: ModelInfo) async throws -> Bool
    func downloadModelMetadata(_ modelId: String) async throws -> ModelMetadata
    func searchModels(query: String) async throws -> [ModelInfo]
    func getModelRecommendations(for capability: ModelCapability) async throws -> [ModelInfo]
}

@MainActor
class ModelCatalogService: ObservableObject, ModelCatalogServiceProtocol {
    
    // MARK: - Published State
    @Published private(set) var availableModels: [ModelInfo] = []
    @Published private(set) var isLoading: Bool = false
    @Published private(set) var lastError: Error?
    
    // MARK: - Private Properties
    private let networkManager: NetworkManager
    private let cacheManager: CacheManager
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Configuration
    private let catalogEndpoint = "https://api.agenticseek.com/models/catalog"
    private let cacheTimeout: TimeInterval = 300 // 5 minutes
    
    // MARK: - Initialization
    init(networkManager: NetworkManager = NetworkManager(), 
         cacheManager: CacheManager = CacheManager()) {
        self.networkManager = networkManager
        self.cacheManager = cacheManager
        
        setupAutoRefresh()
    }
    
    // MARK: - Public Methods
    
    /// Fetch all available models from catalog
    func fetchAvailableModels() async throws -> [ModelInfo] {
        isLoading = true
        lastError = nil
        
        defer { isLoading = false }
        
        do {
            // Check cache first
            if let cached = try await getCachedModels() {
                availableModels = cached
                return cached
            }
            
            // Fetch from network
            let models = try await fetchModelsFromNetwork()
            
            // Cache results
            try await cacheModels(models)
            
            availableModels = models
            return models
            
        } catch {
            lastError = error
            throw ModelCatalogError.fetchFailed(error)
        }
    }
    
    /// Validate if model is compatible with current system
    func validateModelCompatibility(_ model: ModelInfo) async throws -> Bool {
        do {
            let systemInfo = await getSystemInfo()
            
            // Check memory requirements
            guard systemInfo.availableMemoryGB >= model.minimumMemoryGB else {
                throw ModelCatalogError.insufficientMemory(
                    required: model.minimumMemoryGB,
                    available: systemInfo.availableMemoryGB
                )
            }
            
            // Check architecture compatibility
            guard systemInfo.supportedArchitectures.contains(model.architecture) else {
                throw ModelCatalogError.incompatibleArchitecture(model.architecture)
            }
            
            // Check framework compatibility
            guard systemInfo.supportedFrameworks.contains(model.framework) else {
                throw ModelCatalogError.incompatibleFramework(model.framework)
            }
            
            return true
            
        } catch {
            throw ModelCatalogError.validationFailed(error)
        }
    }
    
    /// Download detailed metadata for specific model
    func downloadModelMetadata(_ modelId: String) async throws -> ModelMetadata {
        let endpoint = "\(catalogEndpoint)/\(modelId)/metadata"
        
        do {
            let request = try await networkManager.createRequest(for: endpoint)
            let (data, response) = try await networkManager.execute(request)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  200...299 ~= httpResponse.statusCode else {
                throw ModelCatalogError.networkError("Invalid response")
            }
            
            let metadata = try JSONDecoder().decode(ModelMetadata.self, from: data)
            return metadata
            
        } catch {
            throw ModelCatalogError.metadataDownloadFailed(error)
        }
    }
    
    /// Search models by query string
    func searchModels(query: String) async throws -> [ModelInfo] {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return availableModels
        }
        
        let searchTerm = query.lowercased()
        
        return availableModels.filter { model in
            model.name.lowercased().contains(searchTerm) ||
            model.description.lowercased().contains(searchTerm) ||
            model.tags.contains { $0.lowercased().contains(searchTerm) } ||
            model.capabilities.contains { $0.rawValue.lowercased().contains(searchTerm) }
        }
    }
    
    /// Get model recommendations based on capability requirements
    func getModelRecommendations(for capability: ModelCapability) async throws -> [ModelInfo] {
        let compatibleModels = availableModels.filter { model in
            model.capabilities.contains(capability)
        }
        
        // Sort by performance score for the specific capability
        return compatibleModels.sorted { lhs, rhs in
            let lhsScore = lhs.performanceScores[capability] ?? 0
            let rhsScore = rhs.performanceScores[capability] ?? 0
            return lhsScore > rhsScore
        }.prefix(5).map { $0 } // Top 5 recommendations
    }
    
    // MARK: - Private Methods
    
    private func setupAutoRefresh() {
        // Auto-refresh every 30 minutes
        Timer.publish(every: 1800, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                Task {
                    try? await self?.fetchAvailableModels()
                }
            }
            .store(in: &cancellables)
    }
    
    private func getCachedModels() async throws -> [ModelInfo]? {
        let cacheKey = "available_models"
        
        guard let cachedData = try await cacheManager.retrieve(key: cacheKey),
              let models = try? JSONDecoder().decode([ModelInfo].self, from: cachedData),
              let cacheDate = try await cacheManager.getCacheDate(key: cacheKey),
              Date().timeIntervalSince(cacheDate) < cacheTimeout else {
            return nil
        }
        
        return models
    }
    
    private func cacheModels(_ models: [ModelInfo]) async throws {
        let cacheKey = "available_models"
        let data = try JSONEncoder().encode(models)
        try await cacheManager.store(data: data, key: cacheKey)
    }
    
    private func fetchModelsFromNetwork() async throws -> [ModelInfo] {
        let request = try await networkManager.createRequest(for: catalogEndpoint)
        let (data, response) = try await networkManager.execute(request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              200...299 ~= httpResponse.statusCode else {
            throw ModelCatalogError.networkError("Failed to fetch models")
        }
        
        let catalogResponse = try JSONDecoder().decode(ModelCatalogResponse.self, from: data)
        return catalogResponse.models
    }
    
    private func getSystemInfo() async -> SystemInfo {
        // Mock implementation - would get actual system info
        return SystemInfo(
            availableMemoryGB: 16.0,
            supportedArchitectures: [.arm64, .x86_64],
            supportedFrameworks: [.coreML, .onnx, .tensorFlow, .pytorch]
        )
    }
}

// MARK: - Model Information Types

struct ModelInfo: Codable, Identifiable, Equatable {
    let id: String
    let name: String
    let description: String
    let version: String
    let author: String
    let architecture: ModelArchitecture
    let framework: ModelFramework
    let minimumMemoryGB: Double
    let sizeGB: Double
    let capabilities: [ModelCapability]
    let tags: [String]
    let performanceScores: [ModelCapability: Double]
    let downloadURL: URL
    let licenseType: LicenseType
    let lastUpdated: Date
    
    static func == (lhs: ModelInfo, rhs: ModelInfo) -> Bool {
        lhs.id == rhs.id
    }
}

struct ModelMetadata: Codable {
    let modelId: String
    let technicalSpecs: TechnicalSpecs
    let benchmarkResults: [BenchmarkResult]
    let trainingDetails: TrainingDetails
    let usageExamples: [UsageExample]
}

struct TechnicalSpecs: Codable {
    let parameterCount: Int64
    let contextLength: Int
    let vocabularySize: Int
    let precision: ModelPrecision
    let quantizationLevel: QuantizationLevel?
}

struct BenchmarkResult: Codable {
    let benchmarkName: String
    let score: Double
    let details: String
    let dateRun: Date
}

struct TrainingDetails: Codable {
    let datasetSize: String
    let trainingDuration: String
    let computeUsed: String
    let specializations: [String]
}

struct UsageExample: Codable {
    let title: String
    let prompt: String
    let expectedOutput: String
    let capability: ModelCapability
}

struct SystemInfo {
    let availableMemoryGB: Double
    let supportedArchitectures: [ModelArchitecture]
    let supportedFrameworks: [ModelFramework]
}

struct ModelCatalogResponse: Codable {
    let models: [ModelInfo]
    let totalCount: Int
    let lastUpdated: Date
}

// MARK: - Enumerations

enum ModelCapability: String, Codable, CaseIterable {
    case textGeneration = "text_generation"
    case codeGeneration = "code_generation"
    case imageGeneration = "image_generation"
    case textToSpeech = "text_to_speech"
    case speechToText = "speech_to_text"
    case imageAnalysis = "image_analysis"
    case translation = "translation"
    case summarization = "summarization"
    case questionAnswering = "question_answering"
    case codeReview = "code_review"
    case webBrowsing = "web_browsing"
    case fileManagement = "file_management"
}

enum ModelArchitecture: String, Codable {
    case transformer = "transformer"
    case diffusion = "diffusion"
    case cnn = "cnn"
    case rnn = "rnn"
    case hybrid = "hybrid"
    case arm64 = "arm64"
    case x86_64 = "x86_64"
}

enum ModelFramework: String, Codable {
    case coreML = "coreml"
    case onnx = "onnx"
    case tensorFlow = "tensorflow"
    case pytorch = "pytorch"
    case llamaCpp = "llama_cpp"
    case ollama = "ollama"
}

enum ModelPrecision: String, Codable {
    case fp32 = "fp32"
    case fp16 = "fp16"
    case int8 = "int8"
    case int4 = "int4"
}

enum QuantizationLevel: String, Codable {
    case none = "none"
    case q4_0 = "q4_0"
    case q4_1 = "q4_1"
    case q5_0 = "q5_0"
    case q5_1 = "q5_1"
    case q8_0 = "q8_0"
}

enum LicenseType: String, Codable {
    case mit = "mit"
    case apache2 = "apache2"
    case gpl = "gpl"
    case commercial = "commercial"
    case custom = "custom"
}

// MARK: - Error Types

enum ModelCatalogError: Error, LocalizedError {
    case fetchFailed(Error)
    case validationFailed(Error)
    case metadataDownloadFailed(Error)
    case networkError(String)
    case insufficientMemory(required: Double, available: Double)
    case incompatibleArchitecture(ModelArchitecture)
    case incompatibleFramework(ModelFramework)
    case cacheError(String)
    
    var errorDescription: String? {
        switch self {
        case .fetchFailed(let error):
            return "Failed to fetch model catalog: \(error.localizedDescription)"
        case .validationFailed(let error):
            return "Model validation failed: \(error.localizedDescription)"
        case .metadataDownloadFailed(let error):
            return "Failed to download model metadata: \(error.localizedDescription)"
        case .networkError(let message):
            return "Network error: \(message)"
        case .insufficientMemory(let required, let available):
            return "Insufficient memory: requires \(required)GB, available \(available)GB"
        case .incompatibleArchitecture(let arch):
            return "Incompatible architecture: \(arch.rawValue)"
        case .incompatibleFramework(let framework):
            return "Incompatible framework: \(framework.rawValue)"
        case .cacheError(let message):
            return "Cache error: \(message)"
        }
    }
}

// MARK: - Mock Network and Cache Managers
// These would be implemented as separate services

class NetworkManager {
    func createRequest(for url: String) async throws -> URLRequest {
        guard let url = URL(string: url) else {
            throw URLError(.badURL)
        }
        return URLRequest(url: url)
    }
    
    func execute(_ request: URLRequest) async throws -> (Data, URLResponse) {
        // Mock implementation - would use actual networking
        let mockData = """
        {
            "models": [],
            "totalCount": 0,
            "lastUpdated": "2025-05-31T12:00:00Z"
        }
        """.data(using: .utf8)!
        
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: nil
        )!
        
        return (mockData, response)
    }
}

class CacheManager {
    func retrieve(key: String) async throws -> Data? {
        // Mock implementation
        return nil
    }
    
    func store(data: Data, key: String) async throws {
        // Mock implementation
    }
    
    func getCacheDate(key: String) async throws -> Date? {
        // Mock implementation
        return nil
    }
}