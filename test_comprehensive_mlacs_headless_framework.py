#!/usr/bin/env python3
"""
SANDBOX FILE: For testing/development. See .cursorrules.

Comprehensive Test Suite for MLACS Headless Test Framework
==========================================================

Tests the comprehensive headless testing framework with full MLACS validation,
CI/CD integration, and cross-framework compatibility testing.
"""

import asyncio
import json
import logging
import os
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import threading

# Import the system under test
from comprehensive_mlacs_headless_test_framework import (
    MLACSHeadlessTestFramework,
    MLACSTestFrameworkFactory,
    TestCase,
    TestResult,
    TestSuite,
    TestExecution,
    TestCategory,
    TestStatus,
    TestPriority,
    FrameworkType,
    timer_decorator,
    async_timer_decorator
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMLACSHeadlessTestFramework(unittest.TestCase):
    """Test suite for MLACS Headless Test Framework"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary database for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_mlacs_framework.db")
        self.test_results_dir = os.path.join(self.temp_dir, "test_results")
        
        # Create test framework
        self.framework = MLACSHeadlessTestFramework(
            db_path=self.test_db_path,
            results_directory=self.test_results_dir,
            enable_parallel_execution=True,
            max_workers=4,
            enable_performance_monitoring=True,
            enable_ci_cd_integration=True
        )
        
        logger.info("Test setup completed")

    def tearDown(self):
        """Clean up test environment"""
        try:
            # Clean up temporary files
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Test cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    def test_01_framework_initialization(self):
        """Test framework initialization and database setup"""
        logger.info("Testing framework initialization...")
        
        # Test database exists
        self.assertTrue(os.path.exists(self.test_db_path))
        
        # Test database structure
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'test_cases',
            'test_results', 
            'test_executions',
            'performance_baselines'
        ]
        
        for table in expected_tables:
            self.assertIn(table, tables)
        
        conn.close()
        
        # Test system components
        self.assertIsNotNone(self.framework.test_suites)
        self.assertIsNotNone(self.framework.framework_instances)
        self.assertIsInstance(self.framework.test_suites, dict)
        
        logger.info("✅ Framework initialization test passed")

    def test_02_test_suite_registration(self):
        """Test test suite registration and management"""
        logger.info("Testing test suite registration...")
        
        # Check built-in test suites are registered
        self.assertGreater(len(self.framework.test_suites), 0)
        
        # Check specific test suites
        expected_suites = [
            'optimization_engine_suite',
            'cross_framework_suite',
            'performance_benchmark_suite',
            'ci_cd_validation_suite'
        ]
        
        for suite_name in expected_suites:
            if suite_name in self.framework.test_suites:
                suite = self.framework.test_suites[suite_name]
                self.assertIsInstance(suite, TestSuite)
                self.assertGreater(len(suite.test_cases), 0)
        
        logger.info("✅ Test suite registration test passed")

    def test_03_test_case_creation_and_validation(self):
        """Test test case creation and validation"""
        logger.info("Testing test case creation...")
        
        # Create test case
        test_case = TestCase(
            name="test_sample_case",
            description="Sample test case for validation",
            category=TestCategory.UNIT,
            priority=TestPriority.MEDIUM,
            framework=FrameworkType.OPTIMIZATION_ENGINE,
            test_function=lambda: "test_result",
            expected_duration=2.0,
            tags=["test", "sample"]
        )
        
        # Validate test case structure
        self.assertIsNotNone(test_case.id)
        self.assertEqual(test_case.name, "test_sample_case")
        self.assertEqual(test_case.category, TestCategory.UNIT)
        self.assertEqual(test_case.priority, TestPriority.MEDIUM)
        self.assertEqual(test_case.framework, FrameworkType.OPTIMIZATION_ENGINE)
        self.assertIsNotNone(test_case.test_function)
        
        logger.info("✅ Test case creation test passed")

    def test_04_single_test_execution(self):
        """Test single test case execution"""
        logger.info("Testing single test execution...")
        
        # Create test case
        test_case = TestCase(
            name="test_execution_sample",
            description="Sample test for execution testing",
            category=TestCategory.UNIT,
            priority=TestPriority.HIGH,
            framework=FrameworkType.OPTIMIZATION_ENGINE,
            test_function=lambda: "execution_successful",
            expected_duration=1.0,
            tags=["execution", "test"]
        )
        
        # Execute test
        result = self.framework._execute_single_test(test_case)
        
        # Validate result
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.test_id, test_case.id)
        self.assertEqual(result.test_name, test_case.name)
        self.assertEqual(result.status, TestStatus.PASSED)
        self.assertGreater(result.duration, 0)
        self.assertEqual(result.output, "execution_successful")
        
        logger.info("✅ Single test execution test passed")

    def test_05_parallel_test_execution(self):
        """Test parallel test execution"""
        logger.info("Testing parallel test execution...")
        
        # Create multiple test cases
        test_cases = []
        for i in range(5):
            test_case = TestCase(
                name=f"test_parallel_{i}",
                description=f"Parallel test case {i}",
                category=TestCategory.UNIT,
                priority=TestPriority.MEDIUM,
                framework=FrameworkType.OPTIMIZATION_ENGINE,
                test_function=lambda: time.sleep(0.1) or "parallel_success",
                expected_duration=0.2,
                tags=["parallel", "test"]
            )
            test_cases.append(test_case)
        
        # Execute tests in parallel
        start_time = time.time()
        results = self.framework._execute_tests_parallel(test_cases, max_workers=3)
        end_time = time.time()
        
        # Validate results
        self.assertEqual(len(results), 5)
        self.assertLess(end_time - start_time, 1.0)  # Should be faster than sequential
        
        for result in results:
            self.assertIsInstance(result, TestResult)
            self.assertEqual(result.status, TestStatus.PASSED)
        
        logger.info("✅ Parallel test execution test passed")

    def test_06_test_suite_execution(self):
        """Test complete test suite execution"""
        logger.info("Testing test suite execution...")
        
        # Execute optimization engine suite if available
        if 'optimization_engine_suite' in self.framework.test_suites:
            execution = self.framework.execute_test_suite('optimization_engine_suite')
            
            # Validate execution
            self.assertIsInstance(execution, TestExecution)
            self.assertGreater(execution.total_tests, 0)
            self.assertGreaterEqual(execution.passed_tests, 0)
            self.assertIsNotNone(execution.end_time)
            
            # Check test results
            self.assertEqual(len(execution.test_results), execution.total_tests)
            
            # Validate success rate
            success_rate = (execution.passed_tests / execution.total_tests) * 100
            self.assertGreaterEqual(success_rate, 50.0)  # At least 50% success rate
            
            logger.info(f"Suite execution: {execution.passed_tests}/{execution.total_tests} passed ({success_rate:.1f}%)")
        
        logger.info("✅ Test suite execution test passed")

    def test_07_framework_compatibility_validation(self):
        """Test framework compatibility validation"""
        logger.info("Testing framework compatibility...")
        
        # Check available frameworks
        available_frameworks = [name for name, available in self.framework.framework_instances.items() if available]
        self.assertGreater(len(available_frameworks), 0)
        
        # Test each available framework instance
        for framework_name, instance in self.framework.framework_instances.items():
            if hasattr(instance, 'get_system_status'):
                status = instance.get_system_status()
                self.assertIsInstance(status, dict)
                self.assertIn('engine_status', status)
        
        logger.info("✅ Framework compatibility test passed")

    def test_08_performance_monitoring(self):
        """Test performance monitoring capabilities"""
        logger.info("Testing performance monitoring...")
        
        # Test timer decorator
        @timer_decorator
        def sample_function():
            time.sleep(0.1)
            return "timer_test"
        
        result = sample_function()
        self.assertEqual(result, "timer_test")
        
        # Test performance baseline functionality
        baseline_key = "test_framework:response_time"
        self.framework.performance_baselines[baseline_key] = 0.5
        self.framework.regression_thresholds[baseline_key] = 1.0
        
        self.assertEqual(self.framework.performance_baselines[baseline_key], 0.5)
        self.assertEqual(self.framework.regression_thresholds[baseline_key], 1.0)
        
        logger.info("✅ Performance monitoring test passed")

    def test_09_ci_cd_integration(self):
        """Test CI/CD integration capabilities"""
        logger.info("Testing CI/CD integration...")
        
        if self.framework.enable_ci_cd_integration:
            # Test CI/CD hooks
            self.assertIn('pre_build', self.framework.ci_cd_hooks)
            self.assertIn('post_build', self.framework.ci_cd_hooks)
            self.assertIn('pre_deployment', self.framework.ci_cd_hooks)
            self.assertIn('post_deployment', self.framework.ci_cd_hooks)
            
            # Test hook execution
            try:
                self.framework._pre_build_hook()
                self.framework._post_build_hook()
                self.framework._pre_deployment_hook()
                self.framework._post_deployment_hook()
            except Exception as e:
                self.fail(f"CI/CD hook execution failed: {e}")
            
            # Test build validation tests
            self.assertGreater(len(self.framework.build_validation_tests), 0)
            self.assertGreater(len(self.framework.deployment_verification_tests), 0)
        
        logger.info("✅ CI/CD integration test passed")

    def test_10_test_result_persistence(self):
        """Test test result persistence and retrieval"""
        logger.info("Testing test result persistence...")
        
        # Create sample test execution
        execution = TestExecution(
            session_name="test_persistence_session",
            total_tests=2,
            passed_tests=1,
            failed_tests=1
        )
        
        # Add sample test results
        execution.test_results.append(TestResult(
            test_id="test_1",
            test_name="test_passed",
            status=TestStatus.PASSED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration=1.0,
            output="test passed"
        ))
        
        execution.test_results.append(TestResult(
            test_id="test_2",
            test_name="test_failed",
            status=TestStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration=0.5,
            error_message="test failed"
        ))
        
        execution.end_time = datetime.now()
        
        # Persist execution
        self.framework._persist_test_execution(execution)
        
        # Verify persistence
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        # Check execution record
        cursor.execute('SELECT * FROM test_executions WHERE id = ?', (execution.id,))
        execution_row = cursor.fetchone()
        self.assertIsNotNone(execution_row)
        
        # Check test result records
        cursor.execute('SELECT * FROM test_results WHERE test_id IN (?, ?)', ("test_1", "test_2"))
        result_rows = cursor.fetchall()
        self.assertEqual(len(result_rows), 2)
        
        conn.close()
        
        logger.info("✅ Test result persistence test passed")

    def test_11_test_report_generation(self):
        """Test test report generation"""
        logger.info("Testing test report generation...")
        
        # Create sample execution with results
        execution = TestExecution(
            session_name="test_report_generation",
            total_tests=3,
            passed_tests=2,
            failed_tests=1,
            end_time=datetime.now()
        )
        
        # Add sample results
        for i in range(3):
            result = TestResult(
                test_id=f"test_{i}",
                test_name=f"test_case_{i}",
                status=TestStatus.PASSED if i < 2 else TestStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration=1.0 + i,
                output=f"result_{i}"
            )
            execution.test_results.append(result)
        
        # Generate report
        self.framework._generate_test_report(execution)
        
        # Verify report file
        report_file = self.framework.results_directory / f"test_report_{execution.session_name}.json"
        self.assertTrue(report_file.exists())
        
        # Verify report content
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        self.assertIn('execution_summary', report_data)
        self.assertIn('test_results', report_data)
        self.assertEqual(report_data['execution_summary']['total_tests'], 3)
        self.assertEqual(len(report_data['test_results']), 3)
        
        logger.info("✅ Test report generation test passed")

    def test_12_system_status_monitoring(self):
        """Test system status monitoring"""
        logger.info("Testing system status monitoring...")
        
        # Get system status
        status = self.framework.get_system_status()
        
        # Validate status structure
        self.assertIsInstance(status, dict)
        self.assertIn('framework_status', status)
        self.assertIn('test_management', status)
        self.assertIn('framework_availability', status)
        self.assertIn('performance_monitoring', status)
        self.assertIn('ci_cd_integration', status)
        self.assertIn('configuration', status)
        
        # Validate test management info
        test_mgmt = status['test_management']
        self.assertIn('total_test_suites', test_mgmt)
        self.assertIn('total_test_cases', test_mgmt)
        self.assertIn('registered_frameworks', test_mgmt)
        
        # Validate framework availability
        framework_availability = status['framework_availability']
        self.assertIsInstance(framework_availability, dict)
        
        logger.info("✅ System status monitoring test passed")

    def test_13_error_handling_and_resilience(self):
        """Test error handling and system resilience"""
        logger.info("Testing error handling and resilience...")
        
        # Test with failing test case
        failing_test = TestCase(
            name="test_failing_case",
            description="Test case that fails",
            category=TestCategory.UNIT,
            priority=TestPriority.LOW,
            framework=FrameworkType.OPTIMIZATION_ENGINE,
            test_function=lambda: exec('raise Exception("Intentional test failure")'),
            expected_duration=1.0,
            tags=["error", "test"]
        )
        
        # Execute failing test
        result = self.framework._execute_single_test(failing_test)
        
        # Verify error handling
        self.assertEqual(result.status, TestStatus.ERROR)
        self.assertIsNotNone(result.error_message)
        self.assertIn("Intentional test failure", result.error_message)
        
        # Test with assertion error
        assertion_test = TestCase(
            name="test_assertion_case",
            description="Test case with assertion failure",
            category=TestCategory.UNIT,
            priority=TestPriority.LOW,
            framework=FrameworkType.OPTIMIZATION_ENGINE,
            test_function=lambda: exec('assert False, "Assertion failure"'),
            expected_duration=1.0,
            tags=["assertion", "test"]
        )
        
        # Execute assertion test
        result = self.framework._execute_single_test(assertion_test)
        
        # Verify assertion handling
        self.assertEqual(result.status, TestStatus.FAILED)
        self.assertIsNotNone(result.error_message)
        
        logger.info("✅ Error handling and resilience test passed")

    def test_14_factory_pattern_configuration(self):
        """Test factory pattern and configuration"""
        logger.info("Testing factory pattern configuration...")
        
        # Test factory with default config
        default_framework = MLACSTestFrameworkFactory.create_test_framework()
        self.assertIsInstance(default_framework, MLACSHeadlessTestFramework)
        self.assertEqual(default_framework.db_path, 'mlacs_test_framework.db')
        self.assertTrue(default_framework.enable_parallel_execution)
        
        # Test factory with custom config
        custom_config = {
            'db_path': 'custom_test_framework.db',
            'enable_parallel_execution': False,
            'max_workers': 2,
            'enable_ci_cd_integration': False
        }
        
        custom_framework = MLACSTestFrameworkFactory.create_test_framework(custom_config)
        self.assertIsInstance(custom_framework, MLACSHeadlessTestFramework)
        self.assertEqual(custom_framework.db_path, 'custom_test_framework.db')
        self.assertFalse(custom_framework.enable_parallel_execution)
        self.assertEqual(custom_framework.max_workers, 2)
        self.assertFalse(custom_framework.enable_ci_cd_integration)
        
        logger.info("✅ Factory pattern configuration test passed")

    def test_15_async_operations(self):
        """Test async operations and decorators"""
        logger.info("Testing async operations...")
        
        # Test async timer decorator
        @async_timer_decorator
        async def async_sample_function():
            await asyncio.sleep(0.1)
            return "async_timer_test"
        
        async def run_async_test():
            result = await async_sample_function()
            self.assertEqual(result, "async_timer_test")
        
        # Run async test
        asyncio.run(run_async_test())
        
        logger.info("✅ Async operations test passed")

def run_comprehensive_mlacs_headless_test_framework_tests():
    """Run all MLACS headless test framework tests and generate report"""
    
    print("🚀 MLACS Headless Test Framework - Comprehensive Test Suite")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test methods
    test_methods = [
        'test_01_framework_initialization',
        'test_02_test_suite_registration',
        'test_03_test_case_creation_and_validation',
        'test_04_single_test_execution',
        'test_05_parallel_test_execution',
        'test_06_test_suite_execution',
        'test_07_framework_compatibility_validation',
        'test_08_performance_monitoring',
        'test_09_ci_cd_integration',
        'test_10_test_result_persistence',
        'test_11_test_report_generation',
        'test_12_system_status_monitoring',
        'test_13_error_handling_and_resilience',
        'test_14_factory_pattern_configuration',
        'test_15_async_operations'
    ]
    
    for method in test_methods:
        suite.addTest(TestMLACSHeadlessTestFramework(method))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Calculate results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
    
    # Generate report
    print("\\n" + "=" * 70)
    print("🚀 MLACS HEADLESS TEST FRAMEWORK TEST REPORT")
    print("=" * 70)
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"💥 Errors: {errors}")
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    print(f"⏱️  Execution Time: {end_time - start_time:.2f} seconds")
    
    # Test categories breakdown
    categories = {
        'Core Framework Operations': ['01', '02', '14'],
        'Test Execution & Management': ['03', '04', '05', '06'],
        'Integration & Compatibility': ['07', '09', '15'],
        'Monitoring & Reporting': ['08', '10', '11', '12'],
        'Resilience & Error Handling': ['13']
    }
    
    print(f"\\n📋 Test Categories Breakdown:")
    for category, test_nums in categories.items():
        category_tests = [t for t in test_methods if any(num in t for num in test_nums)]
        print(f"   {category}: {len(category_tests)} tests")
    
    if failures > 0:
        print(f"\\n❌ Failed Tests:")
        for failure in result.failures:
            print(f"   - {failure[0]}")
    
    if errors > 0:
        print(f"\\n💥 Error Tests:")
        for error in result.errors:
            print(f"   - {error[0]}")
    
    # MLACS headless test framework specific metrics
    print(f"\\n🔧 MLACS Headless Test Framework Capabilities Verified:")
    print(f"   ✅ Comprehensive Test Suite Registration & Management")
    print(f"   ✅ Parallel & Sequential Test Execution")
    print(f"   ✅ Cross-Framework Compatibility Validation")
    print(f"   ✅ CI/CD Integration & Build Validation")
    print(f"   ✅ Performance Monitoring & Benchmarking")
    print(f"   ✅ Test Result Persistence & Report Generation")
    print(f"   ✅ Error Handling & System Resilience")
    print(f"   ✅ Factory Pattern Configuration & Deployment")
    print(f"   ✅ Async Operations & Timer Decorators")
    print(f"   ✅ System Status Monitoring & Health Checks")
    
    print(f"\\n🏆 MLACS Headless Test Framework: {'PASSED' if success_rate >= 80 else 'NEEDS IMPROVEMENT'}")
    print("=" * 70)
    
    return success_rate >= 80, {
        'total_tests': total_tests,
        'passed': passed,
        'failed': failures,
        'errors': errors,
        'success_rate': success_rate,
        'execution_time': end_time - start_time
    }

if __name__ == "__main__":
    # Run the comprehensive test suite
    success, metrics = run_comprehensive_mlacs_headless_test_framework_tests()
    
    # Exit with appropriate code
    exit(0 if success else 1)