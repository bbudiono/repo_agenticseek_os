
import XCTest
@testable import AgenticSeek

class SingleAgentModeViewTest: XCTestCase {
    var singleAgentView: SingleAgentModeView!
    
    override func setUp() {
        super.setUp()
        singleAgentView = SingleAgentModeView()
    }
    
    func testUIComponentsExist() {
        // Test UI components are properly initialized
        XCTAssertNotNil(singleAgentView.modeToggle, "Mode toggle should exist")
        XCTAssertNotNil(singleAgentView.modelSelector, "Model selector should exist")
        XCTAssertNotNil(singleAgentView.performanceMonitor, "Performance monitor should exist")
    }
    
    func testModeToggleFunctionality() {
        // Test mode switching
        let initialMode = singleAgentView.getCurrentMode()
        singleAgentView.toggleMode()
        let newMode = singleAgentView.getCurrentMode()
        
        XCTAssertNotEqual(initialMode, newMode, "Mode should change when toggled")
    }
    
    func testModelSelectionInterface() {
        // Test model selection UI
        let availableModels = singleAgentView.getAvailableModels()
        XCTAssertGreaterThan(availableModels.count, 0, "Should show available models")
        
        if let firstModel = availableModels.first {
            singleAgentView.selectModel(firstModel)
            let selectedModel = singleAgentView.getSelectedModel()
            XCTAssertEqual(selectedModel?.name, firstModel.name, "Should select the specified model")
        }
    }
    
    func testPerformanceMonitoringDisplay() {
        // Test performance monitoring UI
        let performanceData = singleAgentView.getPerformanceData()
        
        XCTAssertNotNil(performanceData.cpuUsage, "Should display CPU usage")
        XCTAssertNotNil(performanceData.memoryUsage, "Should display memory usage")
        XCTAssertNotNil(performanceData.responseTime, "Should display response time")
    }
}
