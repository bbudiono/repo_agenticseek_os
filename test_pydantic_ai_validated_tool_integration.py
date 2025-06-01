#!/usr/bin/env python3
"""
Comprehensive Test Suite for Pydantic AI Validated Tool Integration Framework

Tests all aspects of the validated tool integration including:
- Tool registration and capability validation
- Tier-based access control and parameter validation
- Tool execution with structured outputs
- Performance metrics and analytics
- Integration with agent factory and communication systems
- Error handling and security validation
"""

import asyncio
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.append('.')

try:
    from sources.pydantic_ai_validated_tool_integration import (
        ValidatedToolIntegrationFramework, create_validated_tool_framework,
        ToolCategory, ToolAccessLevel, ToolValidationLevel, ToolExecutionStatus,
        ToolCapability, ToolParameter, ToolExecutionRequest, quick_tool_integration_test
    )
    from sources.pydantic_ai_tier_aware_agent_factory import (
        AgentSpecialization, AgentTier, AgentCapability
    )
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"❌ Framework import failed: {e}")
    FRAMEWORK_AVAILABLE = False

def print_test_header(test_name: str):
    """Print formatted test header"""
    print(f"\n{'='*70}")
    print(f"🧪 {test_name}")
    print(f"{'='*70}")

def print_test_result(test_name: str, success: bool, details: str = "", execution_time: float = 0.0):
    """Print formatted test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    time_info = f" ({execution_time:.3f}s)" if execution_time > 0 else ""
    print(f"{status:12} {test_name}{time_info}")
    if details:
        print(f"    Details: {details}")

class ValidatedToolIntegrationTestSuite:
    """Comprehensive test suite for Validated Tool Integration Framework"""
    
    def __init__(self):
        self.framework = None
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.start_time = time.time()
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print("🧪 Pydantic AI Validated Tool Integration Framework - Comprehensive Test Suite")
        print("="*70)
        
        tests = [
            ("Import and Initialization", self.test_import_and_initialization),
            ("Framework Configuration", self.test_framework_configuration),
            ("Tool Registration", self.test_tool_registration),
            ("Tool Access Validation", self.test_tool_access_validation),
            ("Parameter Validation", self.test_parameter_validation),
            ("Tool Execution - SUCCESS", self.test_tool_execution_success),
            ("Tool Execution - TIMEOUT", self.test_tool_execution_timeout),
            ("Tool Execution - ACCESS_DENIED", self.test_tool_execution_access_denied),
            ("Tier-Based Access Control", self.test_tier_based_access),
            ("Available Tools Filtering", self.test_available_tools_filtering),
            ("Performance Metrics", self.test_performance_metrics),
            ("Tool Analytics", self.test_tool_analytics),
            ("Execution History", self.test_execution_history),
            ("Error Handling", self.test_error_handling),
            ("Integration Points", self.test_integration_points)
        ]
        
        for test_name, test_func in tests:
            await self.run_single_test(test_name, test_func)
        
        return self.generate_final_report()
    
    async def run_single_test(self, test_name: str, test_func):
        """Run a single test with error handling and timing"""
        self.total_tests += 1
        start_time = time.time()
        
        try:
            result = await test_func()
            execution_time = time.time() - start_time
            
            if result.get('success', False):
                self.passed_tests += 1
                print_test_result(test_name, True, result.get('details', ''), execution_time)
            else:
                print_test_result(test_name, False, result.get('error', 'Unknown error'), execution_time)
            
            self.test_results[test_name] = {
                'success': result.get('success', False),
                'execution_time': execution_time,
                'details': result.get('details', ''),
                'error': result.get('error', '')
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{str(e)}"
            print_test_result(test_name, False, error_msg, execution_time)
            
            self.test_results[test_name] = {
                'success': False,
                'execution_time': execution_time,
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    async def test_import_and_initialization(self) -> Dict[str, Any]:
        """Test framework import and initialization"""
        try:
            if not FRAMEWORK_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Framework components not available for import'
                }
            
            # Test framework creation
            self.framework = create_validated_tool_framework()
            
            if not self.framework:
                return {
                    'success': False,
                    'error': 'Framework creation returned None'
                }
            
            # Verify framework attributes
            required_attributes = [
                'framework_id', 'version', 'registered_tools', 
                'tool_executors', 'execution_history', 'performance_metrics'
            ]
            
            missing_attributes = [attr for attr in required_attributes 
                                if not hasattr(self.framework, attr)]
            
            if missing_attributes:
                return {
                    'success': False,
                    'error': f'Missing framework attributes: {missing_attributes}'
                }
            
            # Test basic framework properties
            framework_info = {
                'framework_id': self.framework.framework_id,
                'version': self.framework.version,
                'registered_tools_count': len(self.framework.registered_tools),
                'tool_executors_count': len(self.framework.tool_executors)
            }
            
            return {
                'success': True,
                'details': f"Framework initialized: {framework_info}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Initialization failed: {str(e)}'
            }
    
    async def test_framework_configuration(self) -> Dict[str, Any]:
        """Test framework configuration and default tools"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            # Test that core tools are registered
            expected_core_tools = [
                "python_interpreter",
                "web_search", 
                "file_operations",
                "browser_automation",
                "mcp_integration"
            ]
            
            registered_tool_ids = list(self.framework.registered_tools.keys())
            missing_tools = [tool_id for tool_id in expected_core_tools 
                           if tool_id not in registered_tool_ids]
            
            if missing_tools:
                return {
                    'success': False,
                    'error': f'Missing core tools: {missing_tools}'
                }
            
            # Verify each tool has proper configuration
            configuration_issues = []
            for tool_id, tool in self.framework.registered_tools.items():
                if not tool.name or not tool.category:
                    configuration_issues.append(f"{tool_id}: missing name or category")
                
                if tool_id not in self.framework.tool_executors:
                    configuration_issues.append(f"{tool_id}: missing executor")
            
            if configuration_issues:
                return {
                    'success': False,
                    'error': f'Configuration issues: {configuration_issues}'
                }
            
            config_summary = {
                'core_tools_registered': len(expected_core_tools),
                'total_tools_registered': len(registered_tool_ids),
                'executors_available': len(self.framework.tool_executors)
            }
            
            return {
                'success': True,
                'details': f"Configuration valid: {config_summary}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Configuration test failed: {str(e)}'
            }
    
    async def test_tool_registration(self) -> Dict[str, Any]:
        """Test tool registration functionality"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            # Create a test tool
            test_tool = ToolCapability(
                tool_id="test_tool_001",
                name="Test Tool",
                category=ToolCategory.CUSTOM,
                access_level=ToolAccessLevel.PUBLIC,
                required_capabilities=[AgentCapability.BASIC_REASONING],
                parameters=[
                    ToolParameter(
                        name="test_param",
                        type="str",
                        description="Test parameter"
                    )
                ],
                description="A test tool for validation"
            )
            
            # Create a mock executor
            async def test_executor(parameters):
                return {"result": f"Test executed with: {parameters}"}
            
            # Test tool registration
            registration_success = self.framework.register_tool(test_tool, test_executor)
            
            if not registration_success:
                return {
                    'success': False,
                    'error': 'Tool registration failed'
                }
            
            # Verify tool is registered
            if test_tool.tool_id not in self.framework.registered_tools:
                return {
                    'success': False,
                    'error': 'Tool not found in registry after registration'
                }
            
            # Test duplicate registration (should update)
            duplicate_registration = self.framework.register_tool(test_tool, test_executor)
            
            registration_summary = {
                'initial_registration': registration_success,
                'duplicate_registration': duplicate_registration,
                'tool_in_registry': test_tool.tool_id in self.framework.registered_tools,
                'executor_available': test_tool.tool_id in self.framework.tool_executors
            }
            
            return {
                'success': all(registration_summary.values()),
                'details': f"Tool registration: {registration_summary}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Tool registration test failed: {str(e)}'
            }
    
    async def test_tool_access_validation(self) -> Dict[str, Any]:
        """Test tool access validation logic"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            validation_tests = []
            
            # Test 1: FREE tier accessing public tool
            access_valid, message = self.framework.validate_tool_access(
                "web_search",
                "test_agent_free",
                AgentTier.FREE,
                [AgentCapability.WEB_BROWSING]
            )
            validation_tests.append({
                'test': 'FREE_tier_public_tool',
                'passed': access_valid,
                'message': message
            })
            
            # Test 2: FREE tier accessing restricted tool (should fail)
            access_valid, message = self.framework.validate_tool_access(
                "python_interpreter",
                "test_agent_free",
                AgentTier.FREE,
                [AgentCapability.CODE_GENERATION]
            )
            validation_tests.append({
                'test': 'FREE_tier_restricted_tool',
                'passed': not access_valid,  # Should fail
                'message': message
            })
            
            # Test 3: PRO tier accessing restricted tool
            access_valid, message = self.framework.validate_tool_access(
                "python_interpreter",
                "test_agent_pro",
                AgentTier.PRO,
                [AgentCapability.CODE_GENERATION]
            )
            validation_tests.append({
                'test': 'PRO_tier_restricted_tool',
                'passed': access_valid,
                'message': message
            })
            
            # Test 4: Missing capabilities (should fail)
            access_valid, message = self.framework.validate_tool_access(
                "python_interpreter",
                "test_agent_pro",
                AgentTier.PRO,
                [AgentCapability.BASIC_REASONING]  # Missing CODE_GENERATION
            )
            validation_tests.append({
                'test': 'missing_capabilities',
                'passed': not access_valid,  # Should fail
                'message': message
            })
            
            # Test 5: ENTERPRISE tier accessing premium tool
            access_valid, message = self.framework.validate_tool_access(
                "mcp_integration",
                "test_agent_enterprise",
                AgentTier.ENTERPRISE,
                [AgentCapability.MCP_INTEGRATION, AgentCapability.ADVANCED_REASONING]
            )
            validation_tests.append({
                'test': 'ENTERPRISE_tier_premium_tool',
                'passed': access_valid,
                'message': message
            })
            
            passed_validations = sum(1 for test in validation_tests if test['passed'])
            total_validations = len(validation_tests)
            
            return {
                'success': passed_validations == total_validations,
                'details': f"Access validation: {passed_validations}/{total_validations} tests passed"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Access validation test failed: {str(e)}'
            }
    
    async def test_parameter_validation(self) -> Dict[str, Any]:
        """Test parameter validation functionality"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            parameter_tests = []
            
            # Test 1: Valid parameters
            params_valid, details = self.framework.validate_tool_parameters(
                "web_search",
                {"query": "test search", "max_results": 10},
                ToolValidationLevel.STANDARD
            )
            parameter_tests.append({
                'test': 'valid_parameters',
                'passed': params_valid,
                'validation_errors': len(details.get('validation_errors', []))
            })
            
            # Test 2: Missing required parameter
            params_valid, details = self.framework.validate_tool_parameters(
                "web_search",
                {"max_results": 10},  # Missing 'query'
                ToolValidationLevel.STANDARD
            )
            parameter_tests.append({
                'test': 'missing_required_parameter',
                'passed': not params_valid,  # Should fail
                'validation_errors': len(details.get('validation_errors', []))
            })
            
            # Test 3: Security validation (strict mode)
            params_valid, details = self.framework.validate_tool_parameters(
                "python_interpreter",
                {"code": "import os; os.system('rm -rf /')", "timeout": 30},
                ToolValidationLevel.STRICT
            )
            parameter_tests.append({
                'test': 'security_validation_strict',
                'passed': not params_valid,  # Should fail due to dangerous code
                'security_issues': len(details.get('security_issues', []))
            })
            
            # Test 4: Path traversal detection
            params_valid, details = self.framework.validate_tool_parameters(
                "file_operations",
                {"operation": "read", "path": "../../../etc/passwd"},
                ToolValidationLevel.STRICT
            )
            parameter_tests.append({
                'test': 'path_traversal_detection',
                'passed': not params_valid,  # Should fail due to path traversal
                'security_issues': len(details.get('security_issues', []))
            })
            
            passed_parameter_tests = sum(1 for test in parameter_tests if test['passed'])
            total_parameter_tests = len(parameter_tests)
            
            return {
                'success': passed_parameter_tests == total_parameter_tests,
                'details': f"Parameter validation: {passed_parameter_tests}/{total_parameter_tests} tests passed"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Parameter validation test failed: {str(e)}'
            }
    
    async def test_tool_execution_success(self) -> Dict[str, Any]:
        """Test successful tool execution"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            # Create execution request
            request = ToolExecutionRequest(
                tool_id="web_search",
                agent_id="test_agent_001",
                parameters={"query": "test search", "max_results": 5},
                validation_level=ToolValidationLevel.STANDARD
            )
            
            # Execute tool
            result = await self.framework.execute_tool(request)
            
            execution_success = (
                result.output.status == ToolExecutionStatus.SUCCESS and
                result.output.result is not None and
                result.output.execution_time_ms > 0
            )
            
            if execution_success:
                execution_details = {
                    'status': result.output.status.value,
                    'execution_time_ms': result.output.execution_time_ms,
                    'validation_score': result.output.validation_score,
                    'has_result': result.output.result is not None
                }
                
                return {
                    'success': True,
                    'details': f"Tool execution successful: {execution_details}"
                }
            else:
                return {
                    'success': False,
                    'error': f'Tool execution failed: status={result.output.status.value}, error={result.output.error_message}'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Tool execution test failed: {str(e)}'
            }
    
    async def test_tool_execution_timeout(self) -> Dict[str, Any]:
        """Test tool execution timeout handling"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            # Create a custom tool with very short timeout for testing
            timeout_tool = ToolCapability(
                tool_id="timeout_test_tool",
                name="Timeout Test Tool",
                category=ToolCategory.CUSTOM,
                access_level=ToolAccessLevel.PUBLIC,
                timeout_seconds=1,  # Very short timeout
                description="Tool for testing timeout behavior"
            )
            
            # Create executor that takes longer than timeout
            async def slow_executor(parameters):
                await asyncio.sleep(2)  # Longer than timeout
                return {"result": "Should not complete"}
            
            # Register the timeout test tool
            self.framework.register_tool(timeout_tool, slow_executor)
            
            # Create execution request
            request = ToolExecutionRequest(
                tool_id="timeout_test_tool",
                agent_id="test_agent_001",
                parameters={},
                validation_level=ToolValidationLevel.BASIC
            )
            
            # Execute tool (should timeout)
            result = await self.framework.execute_tool(request)
            
            timeout_handled = result.output.status == ToolExecutionStatus.TIMEOUT
            
            return {
                'success': timeout_handled,
                'details': f"Timeout handling: status={result.output.status.value}, time={result.output.execution_time_ms:.1f}ms"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Timeout test failed: {str(e)}'
            }
    
    async def test_tool_execution_access_denied(self) -> Dict[str, Any]:
        """Test tool execution access denial"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            # Create execution request for restricted tool with FREE tier agent
            request = ToolExecutionRequest(
                tool_id="python_interpreter",  # Restricted tool
                agent_id="test_agent_free",
                parameters={"code": "print('hello')"},
                validation_level=ToolValidationLevel.STANDARD
            )
            
            # Execute tool (should be denied)
            result = await self.framework.execute_tool(request)
            
            access_denied = result.output.status == ToolExecutionStatus.FAILED
            has_access_error = "Access denied" in (result.output.error_message or "")
            
            access_control_working = access_denied and has_access_error
            
            return {
                'success': access_control_working,
                'details': f"Access control: denied={access_denied}, error_message_correct={has_access_error}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Access control test failed: {str(e)}'
            }
    
    async def test_tier_based_access(self) -> Dict[str, Any]:
        """Test tier-based access control comprehensively"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            tier_tests = []
            
            # Test available tools for each tier
            for tier in [AgentTier.FREE, AgentTier.PRO, AgentTier.ENTERPRISE]:
                available_tools = self.framework.get_available_tools(
                    agent_tier=tier,
                    agent_capabilities=[cap for cap in AgentCapability]  # All capabilities
                )
                
                # Check expected access patterns
                public_tools = [t for t in available_tools if t.access_level == ToolAccessLevel.PUBLIC]
                restricted_tools = [t for t in available_tools if t.access_level == ToolAccessLevel.RESTRICTED]
                premium_tools = [t for t in available_tools if t.access_level == ToolAccessLevel.PREMIUM]
                
                tier_test_result = {
                    'tier': tier.value,
                    'total_available': len(available_tools),
                    'public_tools': len(public_tools),
                    'restricted_tools': len(restricted_tools),
                    'premium_tools': len(premium_tools)
                }
                
                # Validate access patterns
                if tier == AgentTier.FREE:
                    # Should have public tools only
                    tier_test_result['access_correct'] = (len(restricted_tools) == 0 and len(premium_tools) == 0)
                elif tier == AgentTier.PRO:
                    # Should have public and restricted tools
                    tier_test_result['access_correct'] = (len(premium_tools) == 0)
                else:  # ENTERPRISE
                    # Should have all tools
                    tier_test_result['access_correct'] = True
                
                tier_tests.append(tier_test_result)
            
            all_tier_tests_passed = all(test['access_correct'] for test in tier_tests)
            
            return {
                'success': all_tier_tests_passed,
                'details': f"Tier-based access: {len([t for t in tier_tests if t['access_correct']])}/{len(tier_tests)} tiers correct"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Tier-based access test failed: {str(e)}'
            }
    
    async def test_available_tools_filtering(self) -> Dict[str, Any]:
        """Test available tools filtering functionality"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            # Test 1: No filters (all tools)
            all_tools = self.framework.get_available_tools()
            
            # Test 2: Filter by tier
            free_tools = self.framework.get_available_tools(agent_tier=AgentTier.FREE)
            enterprise_tools = self.framework.get_available_tools(agent_tier=AgentTier.ENTERPRISE)
            
            # Test 3: Filter by capabilities
            basic_capability_tools = self.framework.get_available_tools(
                agent_capabilities=[AgentCapability.BASIC_REASONING]
            )
            code_capability_tools = self.framework.get_available_tools(
                agent_capabilities=[AgentCapability.CODE_GENERATION]
            )
            
            # Test 4: Combined filters
            free_basic_tools = self.framework.get_available_tools(
                agent_tier=AgentTier.FREE,
                agent_capabilities=[AgentCapability.BASIC_REASONING]
            )
            
            filtering_results = {
                'all_tools': len(all_tools),
                'free_tools': len(free_tools),
                'enterprise_tools': len(enterprise_tools),
                'basic_capability_tools': len(basic_capability_tools),
                'code_capability_tools': len(code_capability_tools),
                'free_basic_tools': len(free_basic_tools)
            }
            
            # Validate filtering logic
            filtering_valid = (
                len(free_tools) <= len(enterprise_tools) and  # Enterprise should have more tools
                len(free_basic_tools) <= len(free_tools) and  # Capability filter should reduce count
                len(enterprise_tools) <= len(all_tools)       # All tools should be the maximum
            )
            
            return {
                'success': filtering_valid,
                'details': f"Tool filtering: {filtering_results}, logic_valid={filtering_valid}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Tool filtering test failed: {str(e)}'
            }
    
    async def test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance metrics tracking"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            # Get initial metrics
            initial_metrics = self.framework.performance_metrics.copy()
            
            # Execute a few tools to generate metrics
            for i in range(3):
                request = ToolExecutionRequest(
                    tool_id="web_search",
                    agent_id=f"test_agent_{i}",
                    parameters={"query": f"test search {i}"}
                )
                await self.framework.execute_tool(request)
            
            # Get updated metrics
            updated_metrics = self.framework.performance_metrics
            
            # Verify metrics were updated
            metrics_updated = (
                updated_metrics['total_executions'] > initial_metrics['total_executions'] and
                updated_metrics['successful_executions'] > initial_metrics['successful_executions']
            )
            
            # Check metric structure
            required_metrics = [
                'total_executions', 'successful_executions', 'failed_executions',
                'average_execution_time', 'tool_usage_stats'
            ]
            
            metrics_complete = all(metric in updated_metrics for metric in required_metrics)
            
            # Check tool-specific stats
            web_search_stats = updated_metrics['tool_usage_stats'].get('web_search', {})
            tool_stats_valid = (
                'total_calls' in web_search_stats and
                'successful_calls' in web_search_stats and
                web_search_stats['total_calls'] >= 3
            )
            
            performance_tracking_working = metrics_updated and metrics_complete and tool_stats_valid
            
            return {
                'success': performance_tracking_working,
                'details': f"Performance metrics: updated={metrics_updated}, complete={metrics_complete}, tool_stats={tool_stats_valid}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Performance metrics test failed: {str(e)}'
            }
    
    async def test_tool_analytics(self) -> Dict[str, Any]:
        """Test tool analytics generation"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            # Get analytics
            analytics = self.framework.get_tool_analytics()
            
            # Verify analytics structure
            required_sections = [
                'framework_info',
                'performance_metrics',
                'tool_catalog',
                'category_distribution',
                'access_level_distribution'
            ]
            
            missing_sections = [section for section in required_sections 
                              if section not in analytics]
            
            if missing_sections:
                return {
                    'success': False,
                    'error': f'Missing analytics sections: {missing_sections}'
                }
            
            # Verify framework info
            framework_info = analytics['framework_info']
            framework_info_valid = (
                'framework_id' in framework_info and
                'version' in framework_info and
                'registered_tools' in framework_info and
                framework_info['registered_tools'] > 0
            )
            
            # Verify tool catalog
            tool_catalog = analytics['tool_catalog']
            catalog_valid = (
                len(tool_catalog) > 0 and
                all('name' in tool_info and 'category' in tool_info 
                    for tool_info in tool_catalog.values())
            )
            
            # Verify distributions
            category_dist = analytics['category_distribution']
            access_dist = analytics['access_level_distribution']
            distributions_valid = (
                len(category_dist) > 0 and len(access_dist) > 0 and
                sum(category_dist.values()) == sum(access_dist.values())
            )
            
            analytics_complete = framework_info_valid and catalog_valid and distributions_valid
            
            return {
                'success': analytics_complete,
                'details': f"Analytics: framework_info={framework_info_valid}, catalog={catalog_valid}, distributions={distributions_valid}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Tool analytics test failed: {str(e)}'
            }
    
    async def test_execution_history(self) -> Dict[str, Any]:
        """Test execution history tracking"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            # Execute some tools to create history
            for i in range(2):
                request = ToolExecutionRequest(
                    tool_id="web_search",
                    agent_id=f"test_agent_history_{i}",
                    parameters={"query": f"history test {i}"}
                )
                await self.framework.execute_tool(request)
            
            # Test getting all history
            all_history = self.framework.get_execution_history()
            
            # Test filtering by tool
            web_search_history = self.framework.get_execution_history(tool_id="web_search")
            
            # Test filtering by agent
            agent_history = self.framework.get_execution_history(agent_id="test_agent_history_0")
            
            # Test limit
            limited_history = self.framework.get_execution_history(limit=1)
            
            history_tests = {
                'all_history_count': len(all_history),
                'web_search_history_count': len(web_search_history),
                'agent_history_count': len(agent_history),
                'limited_history_count': len(limited_history)
            }
            
            # Verify history functionality
            history_working = (
                len(all_history) >= 2 and  # At least 2 executions
                len(web_search_history) >= 2 and  # All should be web_search
                len(agent_history) >= 1 and  # At least 1 for specific agent
                len(limited_history) <= 1  # Limit should work
            )
            
            return {
                'success': history_working,
                'details': f"Execution history: {history_tests}, functionality_working={history_working}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Execution history test failed: {str(e)}'
            }
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and edge cases"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            error_tests = []
            
            # Test 1: Non-existent tool
            request = ToolExecutionRequest(
                tool_id="non_existent_tool",
                agent_id="test_agent",
                parameters={}
            )
            result = await self.framework.execute_tool(request)
            error_tests.append({
                'test': 'non_existent_tool',
                'passed': result.output.status == ToolExecutionStatus.FAILED
            })
            
            # Test 2: Invalid agent validation
            access_valid, message = self.framework.validate_tool_access(
                "invalid_tool_id",
                "test_agent",
                AgentTier.FREE,
                []
            )
            error_tests.append({
                'test': 'invalid_tool_validation',
                'passed': not access_valid
            })
            
            # Test 3: Parameter validation for non-existent tool
            params_valid, details = self.framework.validate_tool_parameters(
                "non_existent_tool",
                {},
                ToolValidationLevel.STANDARD
            )
            error_tests.append({
                'test': 'non_existent_tool_parameters',
                'passed': not params_valid
            })
            
            # Test 4: Empty execution history filters
            empty_history = self.framework.get_execution_history(tool_id="non_existent_tool")
            error_tests.append({
                'test': 'empty_history_filter',
                'passed': len(empty_history) == 0
            })
            
            passed_error_tests = sum(1 for test in error_tests if test['passed'])
            total_error_tests = len(error_tests)
            
            return {
                'success': passed_error_tests == total_error_tests,
                'details': f"Error handling: {passed_error_tests}/{total_error_tests} tests passed"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error handling test failed: {str(e)}'
            }
    
    async def test_integration_points(self) -> Dict[str, Any]:
        """Test integration points and dependencies"""
        try:
            if not self.framework:
                return {'success': False, 'error': 'Framework not initialized'}
            
            integration_tests = []
            
            # Test 1: Agent factory injection
            try:
                mock_agent_factory = {"type": "mock", "status": "active"}
                self.framework.set_agent_factory(mock_agent_factory)
                
                has_agent_factory = hasattr(self.framework, 'agent_factory')
                integration_tests.append({
                    'test': 'agent_factory_injection',
                    'passed': has_agent_factory
                })
            except Exception as e:
                integration_tests.append({
                    'test': 'agent_factory_injection',
                    'passed': False,
                    'error': str(e)
                })
            
            # Test 2: Communication manager injection
            try:
                mock_comm_manager = {"type": "mock", "status": "active"}
                self.framework.set_communication_manager(mock_comm_manager)
                
                has_comm_manager = hasattr(self.framework, 'communication_manager')
                integration_tests.append({
                    'test': 'communication_manager_injection',
                    'passed': has_comm_manager
                })
            except Exception as e:
                integration_tests.append({
                    'test': 'communication_manager_injection',
                    'passed': False,
                    'error': str(e)
                })
            
            # Test 3: Quick integration test function
            try:
                quick_test_result = await quick_tool_integration_test()
                quick_test_success = (
                    isinstance(quick_test_result, dict) and 
                    'framework_initialized' in quick_test_result and
                    quick_test_result.get('framework_initialized', False)
                )
                
                integration_tests.append({
                    'test': 'quick_integration_test',
                    'passed': quick_test_success
                })
            except Exception as e:
                integration_tests.append({
                    'test': 'quick_integration_test',
                    'passed': False,
                    'error': str(e)
                })
            
            passed_integration_tests = sum(1 for test in integration_tests if test['passed'])
            total_integration_tests = len(integration_tests)
            
            return {
                'success': passed_integration_tests == total_integration_tests,
                'details': f"Integration points: {passed_integration_tests}/{total_integration_tests} tests passed"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Integration points test failed: {str(e)}'
            }
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_execution_time = time.time() - self.start_time
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        # Determine readiness level
        if success_rate >= 90:
            readiness = "🚀 READY FOR PRODUCTION"
        elif success_rate >= 80:
            readiness = "⚠️ READY FOR FURTHER DEVELOPMENT"
        elif success_rate >= 70:
            readiness = "🔧 REQUIRES ADDITIONAL WORK"
        else:
            readiness = "❌ NOT READY - SIGNIFICANT ISSUES"
        
        print(f"\n{'='*70}")
        print("📋 PYDANTIC AI VALIDATED TOOL INTEGRATION FRAMEWORK TEST RESULTS")
        print(f"{'='*70}")
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            time_info = f" ({result['execution_time']:.3f}s)"
            print(f"{status:12} {test_name}{time_info}")
            if result.get('details'):
                print(f"    Details: {result['details']}")
            if result.get('error'):
                print(f"    Error: {result['error']}")
        
        print(f"\n{'-'*70}")
        print(f"Tests Passed: {self.passed_tests}/{self.total_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Execution Time: {total_execution_time:.2f}s")
        
        print(f"\n🚀 READINESS ASSESSMENT")
        print(f"-----------------------------------")
        print(f"{readiness}")
        if success_rate >= 80:
            print("✅ Core validated tool integration framework functionality validated")
        if success_rate < 80:
            print("⚠️ Some components may need refinement")
        
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'success_rate': success_rate,
            'execution_time': total_execution_time,
            'readiness': readiness,
            'test_results': self.test_results
        }

# Main execution
async def main():
    """Run the comprehensive test suite"""
    test_suite = ValidatedToolIntegrationTestSuite()
    await test_suite.run_all_tests()
    
    print("\n🎉 Validated Tool Integration Framework test suite completed!")

if __name__ == "__main__":
    asyncio.run(main())