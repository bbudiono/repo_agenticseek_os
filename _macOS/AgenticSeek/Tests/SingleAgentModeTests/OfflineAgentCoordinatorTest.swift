
import XCTest
@testable import AgenticSeek

class OfflineAgentCoordinatorTest: XCTestCase {
    var coordinator: OfflineAgentCoordinator!
    
    override func setUp() {
        super.setUp()
        coordinator = OfflineAgentCoordinator()
    }
    
    func testOfflineOperationCapability() {
        // Test complete offline operation
        let canOperateOffline = coordinator.canOperateOffline()
        XCTAssertTrue(canOperateOffline, "Should be able to operate offline")
    }
    
    func testSingleAgentCoordination() {
        // Test single agent management
        let agent = coordinator.createSingleAgent()
        XCTAssertNotNil(agent, "Should create single agent")
        
        let response = coordinator.processQuery("Hello, can you help me?", with: agent)
        XCTAssertFalse(response.isEmpty, "Should provide non-empty response")
    }
    
    func testComplexQueryHandling() {
        // Test handling of complex queries
        let complexQuery = "Explain quantum computing and its applications in cryptography"
        let agent = coordinator.createSingleAgent()
        
        let response = coordinator.processQuery(complexQuery, with: agent)
        XCTAssertGreaterThan(response.count, 100, "Should provide detailed response for complex query")
    }
    
    func testResponseQuality() {
        // Test response quality metrics
        let agent = coordinator.createSingleAgent()
        let query = "What is artificial intelligence?"
        
        let qualityMetrics = coordinator.evaluateResponseQuality(query: query, agent: agent)
        XCTAssertGreaterThan(qualityMetrics.coherence, 0.7, "Response should be coherent")
        XCTAssertGreaterThan(qualityMetrics.relevance, 0.8, "Response should be relevant")
    }
}
