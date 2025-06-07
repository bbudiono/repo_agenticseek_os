
import XCTest
@testable import AgenticSeek

class LMStudioDetectorTest: XCTestCase {
    var detector: LMStudioDetector!
    
    override func setUp() {
        super.setUp()
        detector = LMStudioDetector()
    }
    
    func testLMStudioInstallationDetection() {
        let isInstalled = detector.isLMStudioInstalled()
        let installationPath = detector.getLMStudioPath()
        
        // Test detection capabilities
        if isInstalled {
            XCTAssertNotNil(installationPath, "Should return installation path when installed")
        }
    }
    
    func testLMStudioModelDirectory() {
        let modelDirectory = detector.getModelDirectory()
        
        if detector.isLMStudioInstalled() {
            XCTAssertNotNil(modelDirectory, "Should find model directory")
            XCTAssertTrue(FileManager.default.fileExists(atPath: modelDirectory?.path ?? ""), "Model directory should exist")
        }
    }
    
    func testLMStudioAPIAvailability() {
        let apiAvailable = detector.isAPIAvailable()
        let apiEndpoint = detector.getAPIEndpoint()
        
        if detector.isLMStudioInstalled() {
            XCTAssertNotNil(apiEndpoint, "Should provide API endpoint")
        }
    }
}
