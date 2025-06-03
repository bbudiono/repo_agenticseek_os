//
// PerformanceAnalyticsUITests.swift
// AgenticSeek Enhanced macOS
//
// UI Integration Tests for Performance Analytics Dashboard
//

import XCTest
import SwiftUI

@testable import AgenticSeek

class PerformanceAnalyticsUITests: XCTestCase {
    
    func testPerformanceTabExists() {
        // Test that the Performance tab exists in AppTab enum
        let performanceTab = AppTab.performance
        XCTAssertEqual(performanceTab.rawValue, "Performance")
        XCTAssertEqual(performanceTab.icon, "chart.line.uptrend.xyaxis")
        XCTAssertEqual(performanceTab.description, "Real-time performance analytics")
    }
    
    func testPerformanceTabIsInAllCases() {
        // Test that Performance tab is included in all cases
        let allTabs = AppTab.allCases
        XCTAssertTrue(allTabs.contains(.performance))
        
        // Verify the correct number of tabs (6 total including Performance)
        XCTAssertEqual(allTabs.count, 6)
    }
    
    func testPerformanceAnalyticsViewCreation() {
        // Test that PerformanceAnalyticsView can be created
        let performanceView = PerformanceAnalyticsView()
        XCTAssertNotNil(performanceView)
    }
    
    func testPerformanceAnalyticsManagerInitialization() {
        // Test that PerformanceAnalyticsManager initializes correctly
        let manager = PerformanceAnalyticsManager()
        XCTAssertTrue(manager.isLoading)
        XCTAssertEqual(manager.systemHealthScore, 0.75)
        XCTAssertEqual(manager.uptime, "0:00:00")
        XCTAssertEqual(manager.alertCount, 0)
        XCTAssertEqual(manager.metricsCollected, 0)
        XCTAssertTrue(manager.realTimeMetrics.isEmpty)
        XCTAssertTrue(manager.frameworkComparisons.isEmpty)
        XCTAssertTrue(manager.trendAnalyses.isEmpty)
        XCTAssertTrue(manager.activeBottlenecks.isEmpty)
        XCTAssertTrue(manager.recommendations.isEmpty)
    }
    
    func testTimeRangeEnumValues() {
        // Test TimeRange enum values
        let timeRanges = TimeRange.allCases
        XCTAssertEqual(timeRanges.count, 4)
        XCTAssertTrue(timeRanges.contains(.last1Hour))
        XCTAssertTrue(timeRanges.contains(.last6Hours))
        XCTAssertTrue(timeRanges.contains(.last24Hours))
        XCTAssertTrue(timeRanges.contains(.last7Days))
    }
    
    func testFrameworkFilterEnumValues() {
        // Test FrameworkFilter enum values
        let frameworks = FrameworkFilter.allCases
        XCTAssertEqual(frameworks.count, 4)
        XCTAssertTrue(frameworks.contains(.all))
        XCTAssertTrue(frameworks.contains(.langchain))
        XCTAssertTrue(frameworks.contains(.langgraph))
        XCTAssertTrue(frameworks.contains(.hybrid))
    }
    
    func testFrameworkComparisonDataStructure() {
        // Test FrameworkComparisonData structure
        let comparison = FrameworkComparisonData(
            comparisonId: "test_id",
            frameworkA: "LangGraph",
            frameworkB: "LangChain",
            performanceRatio: 1.25,
            recommendation: "Test recommendation"
        )
        
        XCTAssertEqual(comparison.comparisonId, "test_id")
        XCTAssertEqual(comparison.frameworkA, "LangGraph")
        XCTAssertEqual(comparison.frameworkB, "LangChain")
        XCTAssertEqual(comparison.performanceRatio, 1.25)
        XCTAssertEqual(comparison.recommendation, "Test recommendation")
    }
    
    func testTrendAnalysisDataStructure() {
        // Test TrendAnalysisData structure
        let trend = TrendAnalysisData(
            trendId: "test_trend",
            framework: "LangGraph",
            metricType: "Execution Time",
            direction: "Improving",
            slope: -1.5,
            trendStrength: 0.85
        )
        
        XCTAssertEqual(trend.trendId, "test_trend")
        XCTAssertEqual(trend.framework, "LangGraph")
        XCTAssertEqual(trend.metricType, "Execution Time")
        XCTAssertEqual(trend.direction, "Improving")
        XCTAssertEqual(trend.slope, -1.5)
        XCTAssertEqual(trend.trendStrength, 0.85)
    }
    
    func testBottleneckDataStructure() {
        // Test BottleneckData structure
        let bottleneck = BottleneckData(
            bottleneckId: "test_bottleneck",
            bottleneckType: "cpu_bound",
            severity: 0.75,
            rootCause: "High CPU usage detected",
            suggestedResolution: "Optimize algorithms"
        )
        
        XCTAssertEqual(bottleneck.bottleneckId, "test_bottleneck")
        XCTAssertEqual(bottleneck.bottleneckType, "cpu_bound")
        XCTAssertEqual(bottleneck.severity, 0.75)
        XCTAssertEqual(bottleneck.rootCause, "High CPU usage detected")
        XCTAssertEqual(bottleneck.suggestedResolution, "Optimize algorithms")
    }
    
    func testKeyboardShortcutForPerformanceTab() {
        // Test that Performance tab has correct keyboard shortcut (Cmd+5)
        // This test verifies the keyboard shortcut mapping in ProductionComponents.swift
        let allTabs = AppTab.allCases
        let performanceIndex = allTabs.firstIndex(of: .performance)
        XCTAssertNotNil(performanceIndex)
        // Performance should be at index 4 (0-based), so Cmd+5 shortcut is correct
        XCTAssertEqual(performanceIndex, 4)
    }
    
    // MARK: - UI Integration Tests
    
    func testPerformanceAnalyticsManagerStartAnalytics() {
        let expectation = self.expectation(description: "Analytics should start and load data")
        let manager = PerformanceAnalyticsManager()
        
        // Start analytics
        manager.startAnalytics()
        
        // Wait for loading to complete
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            XCTAssertFalse(manager.isLoading)
            XCTAssertGreaterThan(manager.systemHealthScore, 0.0)
            XCTAssertNotEqual(manager.uptime, "0:00:00")
            XCTAssertGreaterThan(manager.metricsCollected, 0)
            XCTAssertFalse(manager.realTimeMetrics.isEmpty)
            expectation.fulfill()
        }
        
        waitForExpectations(timeout: 2.0, handler: nil)
    }
    
    func testPerformanceAnalyticsManagerRefreshData() {
        let manager = PerformanceAnalyticsManager()
        
        // Start with initial data
        manager.startAnalytics()
        
        // Wait for initial load
        let expectation = self.expectation(description: "Data should refresh")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            let initialMetricsCount = manager.metricsCollected
            
            // Refresh data
            manager.refreshData()
            
            // Verify data was updated
            XCTAssertGreaterThan(manager.metricsCollected, initialMetricsCount)
            expectation.fulfill()
        }
        
        waitForExpectations(timeout: 2.0, handler: nil)
    }
    
    func testMockDataGeneration() {
        let manager = PerformanceAnalyticsManager()
        
        // Start analytics to trigger mock data generation
        manager.startAnalytics()
        
        let expectation = self.expectation(description: "Mock data should be generated")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            // Verify mock data was generated correctly
            XCTAssertFalse(manager.frameworkComparisons.isEmpty)
            XCTAssertFalse(manager.trendAnalyses.isEmpty)
            XCTAssertFalse(manager.recommendations.isEmpty)
            
            // Verify framework comparisons have correct data
            if let firstComparison = manager.frameworkComparisons.first {
                XCTAssertFalse(firstComparison.comparisonId.isEmpty)
                XCTAssertFalse(firstComparison.frameworkA.isEmpty)
                XCTAssertFalse(firstComparison.frameworkB.isEmpty)
                XCTAssertGreaterThan(firstComparison.performanceRatio, 0.0)
                XCTAssertFalse(firstComparison.recommendation.isEmpty)
            }
            
            // Verify trend analyses have correct data
            if let firstTrend = manager.trendAnalyses.first {
                XCTAssertFalse(firstTrend.trendId.isEmpty)
                XCTAssertFalse(firstTrend.framework.isEmpty)
                XCTAssertFalse(firstTrend.metricType.isEmpty)
                XCTAssertFalse(firstTrend.direction.isEmpty)
                XCTAssertGreaterThanOrEqual(firstTrend.trendStrength, 0.0)
                XCTAssertLessThanOrEqual(firstTrend.trendStrength, 1.0)
            }
            
            expectation.fulfill()
        }
        
        waitForExpectations(timeout: 2.0, handler: nil)
    }
}

// MARK: - UI Accessibility Tests

extension PerformanceAnalyticsUITests {
    
    func testPerformanceTabAccessibility() {
        // Test that Performance tab has proper accessibility attributes
        let performanceTab = AppTab.performance
        
        // Verify accessibility information is available
        XCTAssertFalse(performanceTab.rawValue.isEmpty)
        XCTAssertFalse(performanceTab.description.isEmpty)
        XCTAssertFalse(performanceTab.icon.isEmpty)
        
        // Verify the description is user-friendly
        XCTAssertTrue(performanceTab.description.contains("performance"))
        XCTAssertTrue(performanceTab.description.contains("analytics"))
    }
}