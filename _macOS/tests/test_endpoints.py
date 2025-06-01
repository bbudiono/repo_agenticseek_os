#!/usr/bin/env python3
"""
FILE-LEVEL TEST REVIEW & RATING

Purpose: Comprehensive API endpoint, performance, and security testing for AgenticSeek backend.

Issues & Complexity: This file implements a wide range of tests, including basic endpoint checks, performance/load/stress testing, and security validation (auth, input validation, edge cases). The structure is modular and covers many real-world scenarios, but some test logic is synthetic (e.g., random payloads, simulated errors) and may not fully reflect production usage patterns. There is a risk of 'reward hacking' if endpoints are only optimized to pass these specific tests rather than for true robustness or user value.

Ranking/Rating:
- Coverage: 9/10 (Excellent breadth, covers most API surface)
- Realism: 7/10 (Some tests are synthetic or focus on status codes over deep correctness)
- Usefulness: 8/10 (Very useful for regression and performance, but could be improved with more user-centric and adversarial scenarios)
- Reward Hacking Risk: Moderate (Tests could be gamed by hardcoding responses or optimizing for test pass/fail rather than real-world behavior)

Overall Test Quality Score: 8/10

Summary: This file provides a strong foundation for backend API validation, but should be periodically reviewed and updated to ensure tests remain aligned with evolving user needs and adversarial scenarios. Consider adding more fuzzing, negative testing, and user-journey-based API flows to further reduce reward hacking risk.

Comprehensive endpoint testing for AgenticSeek API
Enhanced with performance testing, security validation, and detailed analysis
"""
import requests
import json
import time
import unittest
import threading
import concurrent.futures
import logging
import statistics
import uuid
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Comprehensive test result data structure"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    response_size: int = 0
    headers: Dict[str, str] = field(default_factory=dict)
    response_data: Any = None
    test_type: str = "basic"
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceMetrics:
    """Performance testing metrics"""
    min_time: float
    max_time: float
    avg_time: float
    median_time: float
    percentile_95: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float

class ComprehensiveAPITester:
    """Comprehensive API testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.results: List[TestResult] = []
        self.performance_data: Dict[str, List[float]] = {}
        
        # Test configuration
        self.test_data = {
            "valid_chat": {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_tokens": 100,
                "temperature": 0.7
            },
            "invalid_chat": {
                "model": "",
                "messages": [],
                "max_tokens": -1
            },
            "large_payload": {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "A" * 10000}],
                "max_tokens": 1000
            }
        }
        
        # Authentication test data
        self.auth_tests = [
            ("valid_api_key", "sk-test123456789"),
            ("invalid_api_key", "invalid-key"),
            ("malformed_api_key", "malformed"),
            ("empty_api_key", ""),
            ("sql_injection_attempt", "'; DROP TABLE users; --"),
            ("xss_attempt", "<script>alert('xss')</script>"),
        ]
        
    def setup_session(self):
        """Setup session with default headers and configurations"""
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'ComprehensiveAPITester/1.0',
            'Accept': 'application/json',
        })
        
    def log_result(self, result: TestResult):
        """Log test result"""
        status = "✅ PASS" if result.success else "❌ FAIL"
        logger.info(f"{status} {result.method} {result.endpoint} - {result.response_time:.3f}s - {result.status_code}")
        if result.error_message:
            logger.error(f"Error: {result.error_message}")
        self.results.append(result)

class BasicEndpointTester(ComprehensiveAPITester):
    """Basic endpoint functionality testing"""
    
    def test_endpoint(self, method: str, endpoint: str, data: Any = None, headers: Dict = None) -> TestResult:
        """Test a single endpoint with comprehensive metrics"""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            request_headers = {**self.session.headers, **(headers or {})}
            
            if method.upper() == "GET":
                response = self.session.get(url, timeout=self.timeout, headers=request_headers)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=self.timeout, headers=request_headers)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, timeout=self.timeout, headers=request_headers)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=self.timeout, headers=request_headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = time.time() - start_time
            
            # Parse response data
            try:
                response_data = response.json() if response.content else None
            except json.JSONDecodeError:
                response_data = response.text
            
            result = TestResult(
                endpoint=endpoint,
                method=method.upper(),
                status_code=response.status_code,
                response_time=response_time,
                success=200 <= response.status_code < 300,
                response_size=len(response.content),
                headers=dict(response.headers),
                response_data=response_data,
                test_type="basic"
            )
            
        except requests.exceptions.Timeout:
            result = TestResult(
                endpoint=endpoint,
                method=method.upper(),
                status_code=408,
                response_time=self.timeout,
                success=False,
                error_message="Request timeout",
                test_type="basic"
            )
        except requests.exceptions.ConnectionError:
            result = TestResult(
                endpoint=endpoint,
                method=method.upper(),
                status_code=0,
                response_time=time.time() - start_time,
                success=False,
                error_message="Connection error - server not responding",
                test_type="basic"
            )
        except Exception as e:
            result = TestResult(
                endpoint=endpoint,
                method=method.upper(),
                status_code=0,
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e),
                test_type="basic"
            )
        
        self.log_result(result)
        return result

class PerformanceTester(ComprehensiveAPITester):
    """Performance and load testing capabilities"""
    
    def performance_test_endpoint(self, method: str, endpoint: str, 
                                concurrent_users: int = 10, 
                                requests_per_user: int = 10,
                                data: Any = None) -> PerformanceMetrics:
        """Run performance tests with multiple concurrent users"""
        
        def make_request():
            """Single request function for threading"""
            return self.test_endpoint(method, endpoint, data)
        
        print(f"🚀 Performance testing {method} {endpoint} with {concurrent_users} users, {requests_per_user} requests each...")
        
        all_response_times = []
        successful_requests = 0
        failed_requests = 0
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # Submit all requests
            futures = []
            for user in range(concurrent_users):
                for request_num in range(requests_per_user):
                    future = executor.submit(make_request)
                    futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    all_response_times.append(result.response_time)
                    if result.success:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                except Exception as e:
                    failed_requests += 1
                    logger.error(f"Performance test request failed: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        total_requests = len(all_response_times) + failed_requests
        
        if all_response_times:
            metrics = PerformanceMetrics(
                min_time=min(all_response_times),
                max_time=max(all_response_times),
                avg_time=statistics.mean(all_response_times),
                median_time=statistics.median(all_response_times),
                percentile_95=self._percentile(all_response_times, 95),
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                requests_per_second=total_requests / total_time if total_time > 0 else 0
            )
        else:
            metrics = PerformanceMetrics(
                min_time=0, max_time=0, avg_time=0, median_time=0, percentile_95=0,
                total_requests=total_requests, successful_requests=0, 
                failed_requests=failed_requests, requests_per_second=0
            )
        
        self._log_performance_metrics(endpoint, metrics)
        return metrics
    
    def load_test_suite(self, endpoints: List[Tuple[str, str]], 
                       load_levels: List[int] = [1, 5, 10, 20, 50]) -> Dict[str, Dict[int, PerformanceMetrics]]:
        """Run comprehensive load testing across multiple load levels"""
        
        results = {}
        
        print("🔥 COMPREHENSIVE LOAD TESTING SUITE")
        print("=" * 60)
        
        for method, endpoint in endpoints:
            endpoint_results = {}
            print(f"\n📊 Testing {method} {endpoint}")
            
            for load_level in load_levels:
                print(f"   Load Level: {load_level} concurrent users")
                metrics = self.performance_test_endpoint(method, endpoint, 
                                                       concurrent_users=load_level, 
                                                       requests_per_user=5)
                endpoint_results[load_level] = metrics
                
                # Brief pause between load levels
                time.sleep(1)
            
            results[f"{method} {endpoint}"] = endpoint_results
        
        self._generate_load_test_report(results)
        return results
    
    def stress_test_endpoint(self, method: str, endpoint: str, 
                           duration_seconds: int = 60,
                           max_concurrent_users: int = 100) -> Dict[str, Any]:
        """Stress test an endpoint to find breaking point"""
        
        print(f"💥 Stress testing {method} {endpoint} for {duration_seconds} seconds...")
        
        stress_results = {
            'start_time': datetime.now(),
            'duration': duration_seconds,
            'max_concurrent_users': max_concurrent_users,
            'breaking_point': None,
            'metrics_by_load': {},
            'errors': []
        }
        
        current_load = 1
        last_successful_load = 0
        
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time and current_load <= max_concurrent_users:
            try:
                metrics = self.performance_test_endpoint(method, endpoint, 
                                                       concurrent_users=current_load, 
                                                       requests_per_user=3)
                
                stress_results['metrics_by_load'][current_load] = metrics
                
                # Consider it successful if >80% of requests succeed and avg response time < 5s
                success_rate = metrics.successful_requests / metrics.total_requests if metrics.total_requests > 0 else 0
                
                if success_rate >= 0.8 and metrics.avg_time < 5.0:
                    last_successful_load = current_load
                    current_load = min(current_load * 2, max_concurrent_users)  # Exponential increase
                else:
                    stress_results['breaking_point'] = current_load
                    break
                    
            except Exception as e:
                stress_results['errors'].append(f"Load {current_load}: {str(e)}")
                stress_results['breaking_point'] = current_load
                break
        
        stress_results['last_successful_load'] = last_successful_load
        stress_results['end_time'] = datetime.now()
        
        self._log_stress_test_results(endpoint, stress_results)
        return stress_results
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _log_performance_metrics(self, endpoint: str, metrics: PerformanceMetrics):
        """Log performance metrics"""
        print(f"   📈 Performance Results for {endpoint}:")
        print(f"      Total Requests: {metrics.total_requests}")
        print(f"      Successful: {metrics.successful_requests} ({metrics.successful_requests/metrics.total_requests*100:.1f}%)")
        print(f"      Failed: {metrics.failed_requests}")
        print(f"      Min Response Time: {metrics.min_time:.3f}s")
        print(f"      Max Response Time: {metrics.max_time:.3f}s")
        print(f"      Avg Response Time: {metrics.avg_time:.3f}s")
        print(f"      Median Response Time: {metrics.median_time:.3f}s")
        print(f"      95th Percentile: {metrics.percentile_95:.3f}s")
        print(f"      Requests/Second: {metrics.requests_per_second:.2f}")
    
    def _generate_load_test_report(self, results: Dict[str, Dict[int, PerformanceMetrics]]):
        """Generate comprehensive load test report"""
        print("\n📋 LOAD TEST SUMMARY REPORT")
        print("=" * 60)
        
        for endpoint, load_results in results.items():
            print(f"\n🎯 {endpoint}")
            print("   Load Level | Success Rate | Avg Time | 95th % | RPS")
            print("   " + "-" * 50)
            
            for load_level, metrics in load_results.items():
                success_rate = metrics.successful_requests / metrics.total_requests * 100 if metrics.total_requests > 0 else 0
                print(f"   {load_level:9d} | {success_rate:10.1f}% | {metrics.avg_time:8.3f}s | {metrics.percentile_95:6.3f}s | {metrics.requests_per_second:5.1f}")
    
    def _log_stress_test_results(self, endpoint: str, results: Dict[str, Any]):
        """Log stress test results"""
        print(f"\n💥 STRESS TEST RESULTS for {endpoint}")
        print("=" * 50)
        print(f"Duration: {results['duration']}s")
        print(f"Last Successful Load: {results['last_successful_load']} users")
        if results['breaking_point']:
            print(f"Breaking Point: {results['breaking_point']} users")
        else:
            print("No breaking point found within test parameters")
        
        if results['errors']:
            print("Errors encountered:")
            for error in results['errors']:
                print(f"  - {error}")

class SecurityTester(ComprehensiveAPITester):
    """Security and authentication testing"""
    
    def test_authentication_security(self, auth_endpoints: List[str]) -> List[TestResult]:
        """Test authentication security across multiple attack vectors"""
        
        print("🔐 SECURITY & AUTHENTICATION TESTING")
        print("=" * 60)
        
        security_results = []
        
        for endpoint in auth_endpoints:
            print(f"\n🛡️  Testing security for {endpoint}")
            
            # Test various authentication scenarios
            for test_name, api_key in self.auth_tests:
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                
                result = self.test_endpoint("POST", endpoint, 
                                          data=self.test_data["valid_chat"], 
                                          headers=headers)
                result.test_type = f"security_{test_name}"
                security_results.append(result)
                
                # Log security test result
                if test_name == "valid_api_key" and not result.success:
                    logger.warning(f"Valid API key failed on {endpoint} - potential issue")
                elif test_name in ["sql_injection_attempt", "xss_attempt"] and result.success:
                    logger.critical(f"SECURITY VULNERABILITY: {test_name} succeeded on {endpoint}")
        
        return security_results
    
    def test_input_validation(self, endpoints: List[Tuple[str, str]]) -> List[TestResult]:
        """Test input validation and injection attacks"""
        
        print("\n🔍 INPUT VALIDATION TESTING")
        print("=" * 50)
        
        validation_results = []
        
        # Malicious payloads for testing
        malicious_payloads = [
            {"test": "'; DROP TABLE users; --"},  # SQL injection
            {"test": "<script>alert('xss')</script>"},  # XSS
            {"test": "{{7*7}}"},  # Template injection
            {"test": "../../../etc/passwd"},  # Path traversal
            {"test": "A" * 10000},  # Buffer overflow attempt
            {"test": None},  # Null injection
            {"test": {"nested": {"deep": {"very": "deep"}}}},  # Deep nesting
        ]
        
        for method, endpoint in endpoints:
            if method.upper() in ["POST", "PUT"]:
                for i, payload in enumerate(malicious_payloads):
                    result = self.test_endpoint(method, endpoint, data=payload)
                    result.test_type = f"input_validation_{i}"
                    validation_results.append(result)
                    
                    # Check for dangerous responses
                    if result.response_data and isinstance(result.response_data, str):
                        if any(danger in result.response_data.lower() for danger in 
                               ['error', 'exception', 'stack trace', 'sql', 'database']):
                            logger.warning(f"Potential information leakage in {endpoint}")
        
        return validation_results

class EdgeCaseTester(ComprehensiveAPITester):
    """Edge case and boundary testing"""
    
    def test_boundary_conditions(self, endpoints: List[Tuple[str, str]]) -> List[TestResult]:
        """Test boundary conditions and edge cases"""
        
        print("\n🎯 BOUNDARY & EDGE CASE TESTING")
        print("=" * 50)
        
        edge_results = []
        
        # Edge case test data
        edge_cases = [
            ("empty_request", {}),
            ("null_values", {"model": None, "messages": None}),
            ("zero_values", {"max_tokens": 0, "temperature": 0}),
            ("negative_values", {"max_tokens": -1, "temperature": -1}),
            ("extreme_values", {"max_tokens": 999999, "temperature": 100}),
            ("unicode_test", {"messages": [{"role": "user", "content": "🚀🔥💯 Testing émojis and ünïcödé"}]}),
            ("large_array", {"messages": [{"role": "user", "content": f"Message {i}"} for i in range(100)]}),
            ("circular_reference", {"test": "circular"}),  # Would be modified to create actual circular ref
        ]
        
        for method, endpoint in endpoints:
            if method.upper() in ["POST", "PUT"]:
                print(f"   Testing {method} {endpoint}")
                
                for case_name, data in edge_cases:
                    result = self.test_endpoint(method, endpoint, data=data)
                    result.test_type = f"edge_case_{case_name}"
                    edge_results.append(result)
                    
                    # Brief pause between tests
                    time.sleep(0.1)
        
        return edge_results
    
    def test_concurrent_access(self, endpoint: str, method: str = "GET", num_threads: int = 50) -> List[TestResult]:
        """Test concurrent access to the same resource"""
        
        print(f"\n⚡ CONCURRENT ACCESS TESTING - {num_threads} simultaneous requests")
        print("=" * 60)
        
        concurrent_results = []
        
        def make_concurrent_request():
            result = self.test_endpoint(method, endpoint)
            result.test_type = "concurrent_access"
            return result
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(make_concurrent_request) for _ in range(num_threads)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    concurrent_results.append(result)
                except Exception as e:
                    logger.error(f"Concurrent request failed: {e}")
        
        end_time = time.time()
        
        # Analyze concurrent access results
        successful = sum(1 for r in concurrent_results if r.success)
        failed = len(concurrent_results) - successful
        
        print(f"   Concurrent Access Results:")
        print(f"   Total Requests: {len(concurrent_results)}")
        print(f"   Successful: {successful} ({successful/len(concurrent_results)*100:.1f}%)")
        print(f"   Failed: {failed}")
        print(f"   Total Time: {end_time - start_time:.3f}s")
        
        return concurrent_results

class ComprehensiveTestSuite:
    """Main comprehensive testing orchestrator"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.basic_tester = BasicEndpointTester(base_url)
        self.performance_tester = PerformanceTester(base_url)
        self.security_tester = SecurityTester(base_url)
        self.edge_case_tester = EdgeCaseTester(base_url)
        
        self.all_results = []
        self.test_summary = {
            'start_time': None,
            'end_time': None,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_types': {},
            'performance_metrics': {},
            'security_issues': [],
            'errors': []
        }
    
    def run_comprehensive_test_suite(self, 
                                   run_performance: bool = True,
                                   run_security: bool = True,
                                   run_edge_cases: bool = True,
                                   run_load_testing: bool = False,
                                   run_stress_testing: bool = False) -> Dict[str, Any]:
        """Run the complete comprehensive test suite"""
        
        self.test_summary['start_time'] = datetime.now()
        
        print("🚀 COMPREHENSIVE API TEST SUITE EXECUTION")
        print("=" * 80)
        print(f"🎯 Target: {self.base_url}")
        print(f"⏰ Started: {self.test_summary['start_time']}")
        print("=" * 80)
        
        # Define all endpoints to test
        endpoints = [
            ("GET", "/health"),
            ("GET", "/config/providers"),
            ("GET", "/config/api-keys"),
            ("GET", "/config/storage"),
            ("GET", "/config/models/anthropic"),
            ("GET", "/config/models/openai"),
            ("GET", "/config/models/google"),
            ("GET", "/config/models/deepseek"),
            ("GET", "/config/models/lm_studio"),
            ("GET", "/config/models/ollama"),
            ("POST", "/config/provider"),
            ("GET", "/models/installed"),
            ("GET", "/models/available"),
            ("GET", "/models/catalog"),
            ("POST", "/models/download"),
            ("POST", "/chat/set-model"),
            ("POST", "/chat/completions"),
            ("POST", "/v1/chat/completions"),
            ("POST", "/chat/stream"),
            ("GET", "/system/status"),
            ("POST", "/backend/deploy")
        ]
        
        auth_endpoints = [
            "/chat/completions",
            "/v1/chat/completions",
            "/chat/stream"
        ]
        
        # 1. Basic endpoint testing
        print("\n🔧 PHASE 1: BASIC ENDPOINT TESTING")
        print("-" * 50)
        basic_results = self._run_basic_tests(endpoints)
        self.all_results.extend(basic_results)
        
        # 2. Performance testing
        if run_performance:
            print("\n🚀 PHASE 2: PERFORMANCE TESTING")
            print("-" * 50)
            performance_results = self._run_performance_tests(endpoints[:5])  # Test first 5 endpoints
            self.all_results.extend(performance_results)
        
        # 3. Security testing
        if run_security:
            print("\n🔐 PHASE 3: SECURITY TESTING")
            print("-" * 50)
            security_results = self._run_security_tests(auth_endpoints)
            self.all_results.extend(security_results)
        
        # 4. Edge case testing
        if run_edge_cases:
            print("\n🎯 PHASE 4: EDGE CASE TESTING")
            print("-" * 50)
            edge_results = self._run_edge_case_tests(endpoints)
            self.all_results.extend(edge_results)
        
        # 5. Load testing (optional)
        if run_load_testing:
            print("\n🔥 PHASE 5: LOAD TESTING")
            print("-" * 50)
            self._run_load_tests(endpoints[:3])  # Test first 3 endpoints
        
        # 6. Stress testing (optional)
        if run_stress_testing:
            print("\n💥 PHASE 6: STRESS TESTING")
            print("-" * 50)
            self._run_stress_tests(endpoints[0])  # Test health endpoint
        
        self.test_summary['end_time'] = datetime.now()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        self._export_results_to_json()
        
        return self.test_summary
    
    def _run_basic_tests(self, endpoints: List[Tuple[str, str]]) -> List[TestResult]:
        """Run basic endpoint tests"""
        results = []
        
        for method, endpoint in endpoints:
            print(f"🧪 Testing {method:4} {endpoint}")
            
            # Test with default data for POST requests
            test_data = None
            if method.upper() == "POST":
                if "chat" in endpoint:
                    test_data = self.basic_tester.test_data["valid_chat"]
                else:
                    test_data = {"test": True}
            
            result = self.basic_tester.test_endpoint(method, endpoint, test_data)
            results.append(result)
        
        return results
    
    def _run_performance_tests(self, endpoints: List[Tuple[str, str]]) -> List[TestResult]:
        """Run performance tests"""
        results = []
        
        for method, endpoint in endpoints:
            if method.upper() == "GET":  # Focus on GET endpoints for performance
                print(f"📊 Performance testing {method} {endpoint}")
                
                # Light performance test
                metrics = self.performance_tester.performance_test_endpoint(
                    method, endpoint, concurrent_users=5, requests_per_user=3
                )
                
                # Convert metrics to test result for consistency
                result = TestResult(
                    endpoint=endpoint,
                    method=method,
                    status_code=200 if metrics.successful_requests > 0 else 500,
                    response_time=metrics.avg_time,
                    success=metrics.successful_requests > metrics.failed_requests,
                    test_type="performance",
                    response_data={"metrics": metrics}
                )
                results.append(result)
        
        return results
    
    def _run_security_tests(self, auth_endpoints: List[str]) -> List[TestResult]:
        """Run security tests"""
        return self.security_tester.test_authentication_security(auth_endpoints)
    
    def _run_edge_case_tests(self, endpoints: List[Tuple[str, str]]) -> List[TestResult]:
        """Run edge case tests"""
        boundary_results = self.edge_case_tester.test_boundary_conditions(endpoints)
        
        # Test concurrent access on health endpoint
        concurrent_results = self.edge_case_tester.test_concurrent_access("/health", "GET", 10)
        
        return boundary_results + concurrent_results
    
    def _run_load_tests(self, endpoints: List[Tuple[str, str]]):
        """Run load tests"""
        self.performance_tester.load_test_suite(endpoints, load_levels=[1, 5, 10])
    
    def _run_stress_tests(self, endpoint: Tuple[str, str]):
        """Run stress tests"""
        method, endpoint_path = endpoint
        if method.upper() == "GET":
            self.performance_tester.stress_test_endpoint(method, endpoint_path, duration_seconds=30)
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        
        # Calculate summary statistics
        self.test_summary['total_tests'] = len(self.all_results)
        self.test_summary['passed_tests'] = sum(1 for r in self.all_results if r.success)
        self.test_summary['failed_tests'] = self.test_summary['total_tests'] - self.test_summary['passed_tests']
        
        # Group by test type
        for result in self.all_results:
            test_type = result.test_type
            if test_type not in self.test_summary['test_types']:
                self.test_summary['test_types'][test_type] = {'passed': 0, 'failed': 0}
            
            if result.success:
                self.test_summary['test_types'][test_type]['passed'] += 1
            else:
                self.test_summary['test_types'][test_type]['failed'] += 1
                if result.error_message:
                    self.test_summary['errors'].append({
                        'endpoint': result.endpoint,
                        'method': result.method,
                        'error': result.error_message,
                        'test_type': result.test_type
                    })
        
        # Identify security issues
        for result in self.all_results:
            if result.test_type.startswith('security_') and result.success:
                if any(vuln in result.test_type for vuln in ['sql_injection', 'xss_attempt']):
                    self.test_summary['security_issues'].append({
                        'endpoint': result.endpoint,
                        'vulnerability': result.test_type,
                        'details': 'Potential security vulnerability detected'
                    })
        
        # Print comprehensive report
        duration = self.test_summary['end_time'] - self.test_summary['start_time']
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST EXECUTION REPORT")
        print("="*80)
        print(f"🕐 Test Duration: {duration}")
        print(f"📡 Target Server: {self.base_url}")
        print(f"📈 Total Tests Executed: {self.test_summary['total_tests']}")
        print(f"✅ Passed: {self.test_summary['passed_tests']} ({self.test_summary['passed_tests']/self.test_summary['total_tests']*100:.1f}%)")
        print(f"❌ Failed: {self.test_summary['failed_tests']} ({self.test_summary['failed_tests']/self.test_summary['total_tests']*100:.1f}%)")
        
        print("\n📋 TEST BREAKDOWN BY TYPE:")
        print("-" * 40)
        for test_type, counts in self.test_summary['test_types'].items():
            total = counts['passed'] + counts['failed']
            success_rate = counts['passed'] / total * 100 if total > 0 else 0
            print(f"  {test_type:20} | {counts['passed']:3d} ✅ | {counts['failed']:3d} ❌ | {success_rate:5.1f}%")
        
        if self.test_summary['security_issues']:
            print("\n🚨 SECURITY ISSUES DETECTED:")
            print("-" * 40)
            for issue in self.test_summary['security_issues']:
                print(f"  ⚠️  {issue['endpoint']} - {issue['vulnerability']}")
        
        if self.test_summary['errors']:
            print(f"\n❌ TOP ERRORS ({min(5, len(self.test_summary['errors']))}):")
            print("-" * 40)
            for error in self.test_summary['errors'][:5]:
                print(f"  {error['method']} {error['endpoint']} - {error['error'][:50]}...")
        
        # Performance summary
        performance_results = [r for r in self.all_results if r.test_type == "performance"]
        if performance_results:
            avg_response_time = statistics.mean([r.response_time for r in performance_results])
            print(f"\n⚡ PERFORMANCE SUMMARY:")
            print(f"  Average Response Time: {avg_response_time:.3f}s")
            print(f"  Performance Tests: {len(performance_results)}")
        
        print("="*80)
        
        # Recommendations
        self._generate_recommendations()
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on test results"""
        
        print("\n💡 RECOMMENDATIONS & NEXT STEPS:")
        print("-" * 50)
        
        failed_rate = self.test_summary['failed_tests'] / self.test_summary['total_tests']
        
        if failed_rate > 0.3:
            print("🔴 HIGH PRIORITY: >30% test failure rate detected")
            print("   - Review server configuration and stability")
            print("   - Check if all services are properly running")
        elif failed_rate > 0.1:
            print("🟡 MEDIUM PRIORITY: >10% test failure rate")
            print("   - Investigate failing endpoints")
            print("   - Review error logs for patterns")
        else:
            print("🟢 GOOD: Low failure rate (<10%)")
        
        if self.test_summary['security_issues']:
            print("🚨 SECURITY: Critical security issues found")
            print("   - Immediate security review required")
            print("   - Implement input validation and sanitization")
        
        # Performance recommendations
        performance_results = [r for r in self.all_results if r.test_type == "performance"]
        if performance_results:
            slow_endpoints = [r for r in performance_results if r.response_time > 2.0]
            if slow_endpoints:
                print(f"⚡ PERFORMANCE: {len(slow_endpoints)} slow endpoints detected (>2s)")
                print("   - Optimize database queries and caching")
                print("   - Consider implementing rate limiting")
        
        print("   - Set up continuous monitoring")
        print("   - Implement automated testing in CI/CD pipeline")
        print("   - Regular security audits recommended")
    
    def _export_results_to_json(self):
        """Export detailed results to JSON file"""
        
        export_data = {
            'summary': self.test_summary,
            'detailed_results': []
        }
        
        for result in self.all_results:
            export_data['detailed_results'].append({
                'endpoint': result.endpoint,
                'method': result.method,
                'status_code': result.status_code,
                'response_time': result.response_time,
                'success': result.success,
                'error_message': result.error_message,
                'response_size': result.response_size,
                'test_type': result.test_type,
                'timestamp': result.timestamp.isoformat()
            })
        
        filename = f"comprehensive_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"\n💾 Detailed results exported to: {filename}")
        except Exception as e:
            logger.error(f"Failed to export results: {e}")

def test_all_endpoints():
    """Legacy function - maintained for backward compatibility"""
    print("ℹ️  Running legacy test function...")
    suite = ComprehensiveTestSuite()
    return suite.run_comprehensive_test_suite(
        run_performance=False,
        run_security=False,
        run_edge_cases=False
    )

def run_comprehensive_tests(base_url: str = "http://localhost:8001",
                          performance: bool = True,
                          security: bool = True,
                          edge_cases: bool = True,
                          load_testing: bool = False,
                          stress_testing: bool = False):
    """Run comprehensive API testing suite"""
    
    suite = ComprehensiveTestSuite(base_url)
    return suite.run_comprehensive_test_suite(
        run_performance=performance,
        run_security=security,
        run_edge_cases=edge_cases,
        run_load_testing=load_testing,
        run_stress_testing=stress_testing
    )

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive API Testing Suite')
    parser.add_argument('--url', default='http://localhost:8001', help='Base URL for API testing')
    parser.add_argument('--no-performance', action='store_true', help='Skip performance testing')
    parser.add_argument('--no-security', action='store_true', help='Skip security testing')
    parser.add_argument('--no-edge-cases', action='store_true', help='Skip edge case testing')
    parser.add_argument('--load-testing', action='store_true', help='Include load testing (intensive)')
    parser.add_argument('--stress-testing', action='store_true', help='Include stress testing (very intensive)')
    parser.add_argument('--legacy', action='store_true', help='Run legacy basic testing only')
    
    args = parser.parse_args()
    
    print("🚀 Starting Comprehensive API Testing Suite...")
    print(f"📡 Target URL: {args.url}")
    
    if args.legacy:
        print("ℹ️  Running in legacy mode...")
        test_all_endpoints()
    else:
        print("🔬 Running comprehensive test suite...")
        run_comprehensive_tests(
            base_url=args.url,
            performance=not args.no_performance,
            security=not args.no_security,
            edge_cases=not args.no_edge_cases,
            load_testing=args.load_testing,
            stress_testing=args.stress_testing
        )
    
    print("\n🏁 Testing complete!")
    print("📋 Check the generated JSON report for detailed results")
    print("📄 Check api_test_results.log for detailed logs")