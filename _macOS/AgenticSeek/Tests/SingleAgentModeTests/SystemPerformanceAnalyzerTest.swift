
import XCTest
@testable import AgenticSeek

class SystemPerformanceAnalyzerTest: XCTestCase {
    var analyzer: SystemPerformanceAnalyzer!
    
    override func setUp() {
        super.setUp()
        analyzer = SystemPerformanceAnalyzer()
    }
    
    func testSystemCapabilityDetection() {
        let capabilities = analyzer.analyzeSystemCapabilities()
        
        XCTAssertGreaterThan(capabilities.cpu_cores, 0, "Should detect CPU cores")
        XCTAssertGreaterThan(capabilities.total_ram_gb, 0, "Should detect RAM")
        XCTAssertFalse(capabilities.cpu_brand.isEmpty, "Should detect CPU brand")
    }
    
    func testPerformanceScoreCalculation() {
        let capabilities = analyzer.analyzeSystemCapabilities()
        let performanceScore = analyzer.calculatePerformanceScore(capabilities)
        
        XCTAssertGreaterThanOrEqual(performanceScore, 0.0, "Performance score should be non-negative")
        XCTAssertLessThanOrEqual(performanceScore, 1.0, "Performance score should not exceed 1.0")
    }
    
    func testModelRecommendations() {
        let capabilities = analyzer.analyzeSystemCapabilities()
        let recommendations = analyzer.recommendModels(for: capabilities)
        
        XCTAssertGreaterThan(recommendations.count, 0, "Should provide model recommendations")
        
        for recommendation in recommendations {
            XCTAssertFalse(recommendation.modelName.isEmpty, "Recommendation should have model name")
            XCTAssertGreaterThan(recommendation.suitabilityScore, 0, "Should have positive suitability score")
        }
    }
}
