
import XCTest
@testable import AgenticSeek

class OllamaDetectorTest: XCTestCase {
    var detector: OllamaDetector!
    
    override func setUp() {
        super.setUp()
        detector = OllamaDetector()
    }
    
    func testOllamaInstallationDetection() {
        // Test should detect Ollama installation
        let isInstalled = detector.isOllamaInstalled()
        let installationPath = detector.getOllamaPath()
        
        // GREEN PHASE: Testing implementation
        XCTAssertTrue(isInstalled, "Should detect Ollama installation")
        XCTAssertNotNil(installationPath, "Should return installation path")
    }
    
    func testOllamaModelDiscovery() {
        // Test should discover available models
        let models = detector.discoverModels()
        
        XCTAssertGreaterThan(models.count, 0, "Should discover at least one model")
        
        for model in models {
            XCTAssertFalse(model.name.isEmpty, "Model should have a name")
            XCTAssertGreaterThan(model.size_gb, 0, "Model should have positive size")
        }
    }
    
    func testModelCompatibilityValidation() {
        // Test should validate model compatibility
        let models = detector.discoverModels()
        
        for model in models {
            let compatibility = detector.validateModelCompatibility(model)
            XCTAssertGreaterThanOrEqual(compatibility.score, 0.0, "Compatibility score should be non-negative")
            XCTAssertLessThanOrEqual(compatibility.score, 1.0, "Compatibility score should not exceed 1.0")
        }
    }
}
