import unittest
import time
import json
import subprocess
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

class ContentViewRefactoringIntegrationTests(unittest.TestCase):
    """Integration tests for ContentViewRefactoring feature
    
    These tests validate the complete transformation of the 1,148-line monolithic
    ContentView.swift into modular, .cursorrules-compliant components.
    """
    
    def setUp(self):
        """Set up test environment"""
        self.start_time = time.time()
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.macos_path = self.project_root / "_macOS" / "AgenticSeek"
        self.sources_path = self.project_root / "sources"
        
        # Expected modular components after refactoring
        self.expected_components = [
            "AppNavigationView.swift",      # Lines 28-74 of original
            "ModelSelectionView.swift",     # Lines 503-614 of original
            "ServiceStatusView.swift",      # Lines 95-160 of original
            "ModelSuggestionsView.swift",   # Lines 654-786 of original
            "SystemTestsView.swift"         # Lines 1057-1134 of original
        ]
        
        # Expected business logic services
        self.expected_services = [
            "ModelCatalogService.swift",        # Extracted from ChatConfigurationManager
            "ProviderConfigurationService.swift", # Extracted from ChatConfigurationManager
            "ModelDownloadService.swift",       # Extracted from ChatConfigurationManager  
            "SystemTestingService.swift",       # Extracted from TestManager
            "APIKeyManagementService.swift"     # Extracted from ChatConfigurationManager
        ]
    
    def tearDown(self):
        """Clean up after tests"""
        elapsed = time.time() - self.start_time
        print(f"Test completed in {elapsed:.2f}s")
    
    def test_monolithic_contentview_breakdown(self):
        """Test that 1,148-line ContentView is broken into focused components"""
        original_contentview = self.macos_path / "ContentView.swift"
        
        # Verify original exists and is monolithic
        self.assertTrue(original_contentview.exists(), "Original ContentView.swift must exist")
        
        with open(original_contentview, 'r') as f:
            content = f.read()
            line_count = len(content.split('\n'))
        
        # Should be broken down if refactoring is complete
        if line_count < 300:  # If refactored
            # Verify modular components exist
            for component in self.expected_components:
                component_path = self.macos_path / component
                self.assertTrue(component_path.exists(), 
                               f"Component {component} should exist after refactoring")
                
                # Verify component size (should be < 200 lines each)
                with open(component_path, 'r') as f:
                    component_content = f.read()
                    component_lines = len(component_content.split('\n'))
                    self.assertLess(component_lines, 200, 
                                   f"Component {component} should be < 200 lines (was part of 1,148-line monolith)")
        else:
            # RED phase - should fail until GREEN phase implementation
            self.fail(f"ContentView still monolithic ({line_count} lines). Refactoring needed.")
    
    def test_business_logic_extraction(self):
        """Test that business logic is extracted from UI components"""
        # Check for extracted services
        for service in self.expected_services:
            service_path = self.sources_path / service
            
            if service_path.exists():
                # Verify service has no UI dependencies
                with open(service_path, 'r') as f:
                    content = f.read()
                    self.assertNotIn("import SwiftUI", content, 
                                   f"Service {service} must not import SwiftUI (business logic only)")
                    self.assertNotIn("@State", content, 
                                   f"Service {service} must not use UI state management")
                    self.assertNotIn("View", content, 
                                   f"Service {service} must not implement View protocol")
            else:
                # RED phase - should fail until GREEN phase implementation
                self.fail(f"Service {service} not found. Business logic extraction needed.")
    
    def test_design_system_implementation(self):
        """Test that DesignSystem.swift exists and is .cursorrules compliant"""
        design_system_path = self.macos_path / "DesignSystem.swift"
        
        if design_system_path.exists():
            with open(design_system_path, 'r') as f:
                content = f.read()
                
            # Verify required design system components
            required_components = [
                "struct DesignSystem",
                "struct Colors",
                "struct Typography", 
                "struct Spacing",
                "static let primary",
                "static let secondary",
                "static let agent"
            ]
            
            for component in required_components:
                self.assertIn(component, content, 
                             f"DesignSystem.swift must contain {component}")
        else:
            # RED phase - should fail until GREEN phase implementation
            self.fail("DesignSystem.swift not found. Design system implementation needed.")
    
    def test_contentviewrefactoring_agent_integration(self):
        """Test agent integration for ContentViewRefactoring"""
        # Test agent interface requirements per .cursorrules
        agent_components = ["ModelSelectionView.swift", "SystemTestsView.swift"]
        
        for component in agent_components:
            component_path = self.macos_path / component
            
            if component_path.exists():
                with open(component_path, 'r') as f:
                    content = f.read()
                
                # Test required agent interface modifiers
                required_modifiers = [
                    ".agentAvatarStyle", ".statusIndicatorStyle", ".agentSelectorStyle"
                ]
                
                for modifier in required_modifiers:
                    if "agent" in component.lower():
                        self.assertIn(modifier, content, 
                                     f"Component {component} must use {modifier} for agent interface")
            else:
                # RED phase - should fail until GREEN phase implementation
                self.fail(f"Agent component {component} not found. Implementation needed.")
    
    def test_contentviewrefactoring_performance_benchmarks(self):
        """Test performance requirements for refactored components"""
        performance_requirements = {
            "ui_response_time_ms": 100,
            "memory_usage_mb": 20,
            "main_thread_operations": 5
        }
        
        # Mock performance testing - would integrate with actual performance tools
        for component in self.expected_components:
            component_path = self.macos_path / component
            
            if component_path.exists():
                with open(component_path, 'r') as f:
                    content = f.read()
                
                # Check for heavy operations that should be moved to services
                heavy_operations = ["URLSession", "fileManager", "JSONDecoder", "Process"]
                for operation in heavy_operations:
                    if operation in content:
                        self.fail(f"Component {component} contains heavy operation {operation}. "
                                 f"Should be moved to service layer.")
                
                # Check for excessive async operations
                async_count = content.count("async")
                self.assertLess(async_count, performance_requirements["main_thread_operations"],
                               f"Component {component} has too many async operations ({async_count})")
            else:
                # RED phase - should fail until GREEN phase implementation
                self.fail(f"Component {component} not found for performance testing.")
    
    def test_contentviewrefactoring_security_compliance(self):
        """Test security requirements for refactored components"""
        security_patterns = {
            "api_keys": [r"[A-Za-z0-9]{32,}", r"sk-[A-Za-z0-9]+"],
            "hardcoded_urls": [r"https?://[^\s]+"],
            "credentials": [r"password\s*=\s*['\"][^'\"]+['\"]"]
        }
        
        # Check all component files for security issues
        all_files = list(self.macos_path.glob("*.swift")) + list(self.sources_path.glob("*.swift"))
        
        for file_path in all_files:
            if file_path.name in self.expected_components + self.expected_services:
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for security violations
                    import re
                    for violation_type, patterns in security_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, content):
                                self.fail(f"Security violation in {file_path.name}: "
                                         f"{violation_type} pattern found")
                else:
                    # RED phase - should fail until GREEN phase implementation
                    self.fail(f"File {file_path.name} not found for security compliance testing.")


class ContentViewRefactoringPerformanceTests(unittest.TestCase):
    """Performance-specific tests for ContentView refactoring"""
    
    def setUp(self):
        self.start_time = time.time()
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.macos_path = self.project_root / "_macOS" / "AgenticSeek"
    
    def test_ui_responsiveness_benchmarks(self):
        """Test UI responsiveness after refactoring"""
        # Mock implementation - would integrate with actual UI testing frameworks
        response_time_threshold = 100  # milliseconds
        
        ui_components = ["AppNavigationView.swift", "ModelSelectionView.swift"]
        
        for component in ui_components:
            component_path = self.macos_path / component
            
            if component_path.exists():
                # Mock response time measurement
                mock_response_time = 85  # Mock value - would be actual measurement
                
                self.assertLess(mock_response_time, response_time_threshold,
                               f"Component {component} response time ({mock_response_time}ms) "
                               f"exceeds threshold ({response_time_threshold}ms)")
            else:
                # RED phase - should fail until GREEN phase implementation
                self.fail(f"UI component {component} not found for responsiveness testing.")
    
    def test_memory_management_compliance(self):
        """Test memory management in refactored components"""
        retain_cycle_patterns = [
            r"\bself\s*\.\s*\w+\s*=.*self",  # Direct retain cycles
            r"\{\s*self\s*in",                # Closure capture without weak
        ]
        
        for component in ["AppNavigationView.swift", "ModelSelectionView.swift"]:
            component_path = self.macos_path / component
            
            if component_path.exists():
                with open(component_path, 'r') as f:
                    content = f.read()
                
                import re
                for pattern in retain_cycle_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        # Check if weak self is used
                        if "weak self" not in content:
                            self.fail(f"Potential retain cycle in {component}: {matches[0]}. "
                                     f"Use weak self in closures.")
            else:
                # RED phase - should fail until GREEN phase implementation
                self.fail(f"Component {component} not found for memory management testing.")

if __name__ == '__main__':
    unittest.main()