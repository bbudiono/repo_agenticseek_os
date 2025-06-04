#!/usr/bin/env python3
"""
Performance Regression Testing Framework for AgenticSeek
======================================================

* Purpose: Detect performance regressions in test execution and system performance
* Features: Baseline comparison, trend analysis, automated alerts
* Integration: Works with TDD pipeline to catch performance degradation
"""

import json
import time
import statistics
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: float
    category: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None

@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    test_suite_duration: float
    individual_test_times: Dict[str, float]
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    timestamp: float
    git_commit: Optional[str] = None

@dataclass
class RegressionResult:
    """Result of regression analysis"""
    metric_name: str
    current_value: float
    baseline_value: float
    percentage_change: float
    severity: str  # "none", "warning", "critical"
    description: str

class PerformanceRegressionTester:
    """Automated performance regression testing"""
    
    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.results_dir = Path("performance_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Performance thresholds (percentage degradation)
        self.thresholds = {
            "test_duration": {"warning": 20.0, "critical": 50.0},
            "memory_usage": {"warning": 25.0, "critical": 100.0},
            "success_rate": {"warning": -5.0, "critical": -10.0}  # Negative because lower is worse
        }
    
    def run_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test"""
        logger.info("🚀 Running performance regression test...")
        
        start_time = time.time()
        
        # Run test suite with timing
        test_results = self._run_timed_test_suite()
        
        # Collect system metrics
        system_metrics = self._collect_system_metrics()
        
        # Get git information
        git_commit = self._get_git_commit()
        
        performance_data = {
            "timestamp": time.time(),
            "test_duration": time.time() - start_time,
            "test_results": test_results,
            "system_metrics": system_metrics,
            "git_commit": git_commit
        }
        
        # Save current performance data
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        current_file = self.results_dir / f"performance_{timestamp_str}.json"
        with open(current_file, "w") as f:
            json.dump(performance_data, f, indent=2)
        
        logger.info(f"💾 Performance data saved: {current_file}")
        
        return performance_data
    
    def _run_timed_test_suite(self) -> Dict[str, Any]:
        """Run test suite with detailed timing"""
        logger.info("🧪 Running timed test suite...")
        
        start_time = time.time()
        
        try:
            # Run comprehensive test suite
            result = subprocess.run(
                ["python", "comprehensive_test_suite.py"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            duration = time.time() - start_time
            
            # Parse test results
            test_report = Path("test_report.json")
            if test_report.exists():
                with open(test_report, "r") as f:
                    report_data = json.load(f)
                
                return {
                    "duration": duration,
                    "success_rate": report_data.get("summary", {}).get("success_rate", 0),
                    "total_tests": report_data.get("summary", {}).get("total_tests", 0),
                    "passed_tests": report_data.get("summary", {}).get("passed", 0),
                    "failed_tests": report_data.get("summary", {}).get("failed", 0),
                    "category_performance": report_data.get("performance_metrics", {}),
                    "return_code": result.returncode
                }
            else:
                return {
                    "duration": duration,
                    "success_rate": 0,
                    "return_code": result.returncode,
                    "error": "No test report generated"
                }
                
        except subprocess.TimeoutExpired:
            return {
                "duration": 600,
                "success_rate": 0,
                "return_code": -1,
                "error": "Test suite timeout"
            }
        except Exception as e:
            return {
                "duration": time.time() - start_time,
                "success_rate": 0,
                "return_code": -1,
                "error": str(e)
            }
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics"""
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # CPU usage (sample over 1 second)
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk usage
            disk = psutil.disk_usage('.')
            
            return {
                "memory_total_mb": memory.total / 1024 / 1024,
                "memory_used_mb": memory.used / 1024 / 1024,
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent,
                "disk_total_gb": disk.total / 1024 / 1024 / 1024,
                "disk_used_gb": disk.used / 1024 / 1024 / 1024,
                "disk_percent": (disk.used / disk.total) * 100
            }
        except ImportError:
            logger.warning("psutil not available - using basic metrics")
            return {"error": "psutil not available"}
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            return {"error": str(e)}
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def establish_baseline(self, performance_data: Dict[str, Any]) -> PerformanceBaseline:
        """Establish or update performance baseline"""
        logger.info("📊 Establishing performance baseline...")
        
        test_results = performance_data.get("test_results", {})
        system_metrics = performance_data.get("system_metrics", {})
        
        baseline = PerformanceBaseline(
            test_suite_duration=test_results.get("duration", 0),
            individual_test_times=test_results.get("category_performance", {}),
            memory_usage_mb=system_metrics.get("memory_used_mb", 0),
            cpu_usage_percent=system_metrics.get("cpu_percent", 0),
            success_rate=test_results.get("success_rate", 0),
            timestamp=performance_data.get("timestamp", time.time()),
            git_commit=performance_data.get("git_commit")
        )
        
        # Save baseline
        with open(self.baseline_file, "w") as f:
            json.dump({
                "test_suite_duration": baseline.test_suite_duration,
                "individual_test_times": baseline.individual_test_times,
                "memory_usage_mb": baseline.memory_usage_mb,
                "cpu_usage_percent": baseline.cpu_usage_percent,
                "success_rate": baseline.success_rate,
                "timestamp": baseline.timestamp,
                "git_commit": baseline.git_commit
            }, f, indent=2)
        
        logger.info(f"✅ Baseline established: {self.baseline_file}")
        return baseline
    
    def load_baseline(self) -> Optional[PerformanceBaseline]:
        """Load existing performance baseline"""
        if not self.baseline_file.exists():
            return None
        
        try:
            with open(self.baseline_file, "r") as f:
                data = json.load(f)
            
            return PerformanceBaseline(
                test_suite_duration=data.get("test_suite_duration", 0),
                individual_test_times=data.get("individual_test_times", {}),
                memory_usage_mb=data.get("memory_usage_mb", 0),
                cpu_usage_percent=data.get("cpu_usage_percent", 0),
                success_rate=data.get("success_rate", 0),
                timestamp=data.get("timestamp", 0),
                git_commit=data.get("git_commit")
            )
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
            return None
    
    def analyze_regression(self, current_data: Dict[str, Any], baseline: PerformanceBaseline) -> List[RegressionResult]:
        """Analyze performance regression against baseline"""
        logger.info("🔍 Analyzing performance regression...")
        
        results = []
        test_results = current_data.get("test_results", {})
        system_metrics = current_data.get("system_metrics", {})
        
        # Test duration regression
        current_duration = test_results.get("duration", 0)
        if baseline.test_suite_duration > 0:
            duration_change = ((current_duration - baseline.test_suite_duration) / baseline.test_suite_duration) * 100
            severity = self._determine_severity("test_duration", duration_change)
            
            results.append(RegressionResult(
                metric_name="Test Suite Duration",
                current_value=current_duration,
                baseline_value=baseline.test_suite_duration,
                percentage_change=duration_change,
                severity=severity,
                description=f"Test suite duration changed by {duration_change:.1f}%"
            ))
        
        # Success rate regression
        current_success_rate = test_results.get("success_rate", 0)
        if baseline.success_rate > 0:
            success_change = ((current_success_rate - baseline.success_rate) / baseline.success_rate) * 100
            severity = self._determine_severity("success_rate", success_change)
            
            results.append(RegressionResult(
                metric_name="Test Success Rate",
                current_value=current_success_rate,
                baseline_value=baseline.success_rate,
                percentage_change=success_change,
                severity=severity,
                description=f"Test success rate changed by {success_change:.1f}%"
            ))
        
        # Memory usage regression
        current_memory = system_metrics.get("memory_used_mb", 0)
        if baseline.memory_usage_mb > 0:
            memory_change = ((current_memory - baseline.memory_usage_mb) / baseline.memory_usage_mb) * 100
            severity = self._determine_severity("memory_usage", memory_change)
            
            results.append(RegressionResult(
                metric_name="Memory Usage",
                current_value=current_memory,
                baseline_value=baseline.memory_usage_mb,
                percentage_change=memory_change,
                severity=severity,
                description=f"Memory usage changed by {memory_change:.1f}%"
            ))
        
        return results
    
    def _determine_severity(self, metric_type: str, percentage_change: float) -> str:
        """Determine severity level of performance change"""
        thresholds = self.thresholds.get(metric_type, {"warning": 20.0, "critical": 50.0})
        
        # Handle negative changes for success rate (lower is worse)
        if metric_type == "success_rate":
            if percentage_change <= thresholds["critical"]:
                return "critical"
            elif percentage_change <= thresholds["warning"]:
                return "warning"
            else:
                return "none"
        else:
            # For other metrics, higher is worse
            if abs(percentage_change) >= thresholds["critical"]:
                return "critical"
            elif abs(percentage_change) >= thresholds["warning"]:
                return "warning"
            else:
                return "none"
    
    def generate_regression_report(self, regressions: List[RegressionResult], 
                                 current_data: Dict[str, Any]) -> str:
        """Generate human-readable regression report"""
        report = []
        report.append("🔍 PERFORMANCE REGRESSION ANALYSIS")
        report.append("=" * 50)
        
        # Summary
        critical_regressions = [r for r in regressions if r.severity == "critical"]
        warning_regressions = [r for r in regressions if r.severity == "warning"]
        
        report.append(f"📊 Analysis Summary:")
        report.append(f"   🔴 Critical regressions: {len(critical_regressions)}")
        report.append(f"   🟡 Warning regressions: {len(warning_regressions)}")
        report.append(f"   ✅ Total metrics analyzed: {len(regressions)}")
        
        # Current performance
        test_results = current_data.get("test_results", {})
        report.append(f"\n📈 Current Performance:")
        report.append(f"   ⏱️  Test Duration: {test_results.get('duration', 0):.1f}s")
        report.append(f"   🎯 Success Rate: {test_results.get('success_rate', 0):.1f}%")
        report.append(f"   📊 Tests Passed: {test_results.get('passed_tests', 0)}")
        
        # Detailed regressions
        if critical_regressions or warning_regressions:
            report.append(f"\n🚨 Regression Details:")
            
            for regression in sorted(regressions, key=lambda x: x.severity, reverse=True):
                if regression.severity != "none":
                    icon = "🔴" if regression.severity == "critical" else "🟡"
                    report.append(f"   {icon} {regression.metric_name}")
                    report.append(f"      Current: {regression.current_value:.2f}")
                    report.append(f"      Baseline: {regression.baseline_value:.2f}")
                    report.append(f"      Change: {regression.percentage_change:.1f}%")
                    report.append(f"      {regression.description}")
        else:
            report.append(f"\n✅ No significant performance regressions detected!")
        
        return "\n".join(report)
    
    def run_full_regression_test(self) -> Dict[str, Any]:
        """Run complete regression test with baseline comparison"""
        # Run performance test
        current_data = self.run_performance_test()
        
        # Load or establish baseline
        baseline = self.load_baseline()
        if baseline is None:
            logger.info("📊 No baseline found, establishing new baseline...")
            baseline = self.establish_baseline(current_data)
            return {
                "status": "baseline_established",
                "baseline": baseline,
                "current_data": current_data,
                "message": "New performance baseline established"
            }
        
        # Analyze regressions
        regressions = self.analyze_regression(current_data, baseline)
        
        # Generate report
        report = self.generate_regression_report(regressions, current_data)
        
        # Determine overall status
        critical_count = len([r for r in regressions if r.severity == "critical"])
        warning_count = len([r for r in regressions if r.severity == "warning"])
        
        if critical_count > 0:
            status = "critical_regression"
        elif warning_count > 0:
            status = "warning_regression"
        else:
            status = "no_regression"
        
        return {
            "status": status,
            "regressions": [
                {
                    "metric_name": r.metric_name,
                    "current_value": r.current_value,
                    "baseline_value": r.baseline_value,
                    "percentage_change": r.percentage_change,
                    "severity": r.severity,
                    "description": r.description
                }
                for r in regressions
            ],
            "report": report,
            "current_data": current_data,
            "baseline_timestamp": baseline.timestamp
        }

def main():
    """Run performance regression test"""
    tester = PerformanceRegressionTester()
    
    print("🚀 Starting Performance Regression Test...")
    
    # Run full regression test
    results = tester.run_full_regression_test()
    
    # Print results
    if "report" in results:
        print(results["report"])
    else:
        print(f"✅ {results.get('message', 'Performance test completed')}")
    
    # Save detailed results
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"performance_regression_report_{timestamp_str}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved: {results_file}")
    
    # Exit with appropriate code
    if results["status"] == "critical_regression":
        print("🔴 CRITICAL: Performance regression detected!")
        return 1
    elif results["status"] == "warning_regression":
        print("🟡 WARNING: Performance degradation detected")
        return 0
    else:
        print("✅ SUCCESS: No performance regression detected")
        return 0

if __name__ == "__main__":
    exit(main())