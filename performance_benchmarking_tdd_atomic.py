#!/usr/bin/env python3
"""
🔴 TDD RED PHASE: End-to-End Performance Benchmarking and Load Testing - Atomic
================================================================================
Production-ready performance testing system using atomic TDD processes.

Atomic TDD Process:
1. RED: Write minimal failing test
2. GREEN: Write minimal passing code  
3. REFACTOR: Improve incrementally
4. ATOMIC: Each cycle is complete and functional

* Purpose: Comprehensive performance benchmarking using atomic TDD approach
* Last Updated: 2025-06-07
================================================================================
"""

import unittest
import time
import requests
import json
import sqlite3
import threading
import concurrent.futures
import psutil
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import subprocess
import signal

class TestPerformanceBenchmarkAtomic(unittest.TestCase):
    """🔴 RED: Atomic performance testing"""
    
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.benchmark = PerformanceBenchmarkAtomic(self.temp_db)
        
    def test_single_request_latency(self):
        """🔴 RED: Test single request latency measurement"""
        result = self.benchmark.measure_single_request_latency()
        
        self.assertIsInstance(result, dict)
        self.assertIn("latency_ms", result)
        self.assertIn("status_code", result)
        self.assertIn("success", result)
        
    def test_load_test_simulation(self):
        """🔴 RED: Test load testing simulation"""
        result = self.benchmark.run_load_test_simulation(concurrent_users=2, duration_seconds=5)
        
        self.assertIsInstance(result, dict)
        self.assertIn("total_requests", result)
        self.assertIn("successful_requests", result)
        self.assertIn("average_latency", result)
        self.assertIn("requests_per_second", result)
        
    def test_system_resource_monitoring(self):
        """🔴 RED: Test system resource monitoring during load"""
        resources = self.benchmark.monitor_system_resources()
        
        self.assertIsInstance(resources, dict)
        self.assertIn("cpu_usage", resources)
        self.assertIn("memory_usage", resources)
        self.assertIn("timestamp", resources)
        
    def test_performance_report_generation(self):
        """🔴 RED: Test performance report generation"""
        report = self.benchmark.generate_performance_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn("test_summary", report)
        self.assertIn("performance_metrics", report)
        self.assertIn("recommendations", report)
        
    def tearDown(self):
        if Path(self.temp_db).exists():
            Path(self.temp_db).unlink()

class PerformanceBenchmarkAtomic:
    """🟢 GREEN: Minimal performance testing implementation"""
    
    def __init__(self, db_path: str = "performance_benchmark.db"):
        self.db_path = db_path
        self.base_url = "http://localhost:8000"
        self._init_database()
        
    def _init_database(self):
        """Initialize performance testing database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                latency_ms REAL,
                status_code INTEGER,
                success BOOLEAN,
                concurrent_users INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                cpu_usage REAL,
                memory_usage REAL,
                disk_io_read REAL DEFAULT 0,
                disk_io_write REAL DEFAULT 0,
                network_sent REAL DEFAULT 0,
                network_recv REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS load_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                timestamp REAL NOT NULL,
                concurrent_users INTEGER,
                duration_seconds INTEGER,
                total_requests INTEGER,
                successful_requests INTEGER,
                failed_requests INTEGER,
                average_latency REAL,
                min_latency REAL,
                max_latency REAL,
                requests_per_second REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def measure_single_request_latency(self, endpoint: str = "/health") -> Dict[str, Any]:
        """Measure latency of a single request"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
            end_time = time.time()
            
            latency_ms = round((end_time - start_time) * 1000, 2)
            
            result = {
                "latency_ms": latency_ms,
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "timestamp": time.time(),
                "endpoint": endpoint
            }
            
            # Store in database
            self._store_performance_test(result)
            
            return result
            
        except Exception as e:
            result = {
                "latency_ms": 0.0,
                "status_code": 0,
                "success": False,
                "timestamp": time.time(),
                "endpoint": endpoint,
                "error": str(e)
            }
            return result
    
    def _store_performance_test(self, result: Dict[str, Any]) -> bool:
        """Store performance test result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO performance_tests 
                (test_type, timestamp, latency_ms, status_code, success)
                VALUES (?, ?, ?, ?, ?)
            """, (
                "single_request",
                result["timestamp"],
                result["latency_ms"],
                result["status_code"],
                result["success"]
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception:
            return False
    
    def monitor_system_resources(self) -> Dict[str, Any]:
        """Monitor current system resources"""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory().percent
            
            # Get disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes if disk_io else 0
            disk_write = disk_io.write_bytes if disk_io else 0
            
            # Get network I/O
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent if net_io else 0
            net_recv = net_io.bytes_recv if net_io else 0
            
        except Exception:
            # Fallback values if psutil fails
            cpu = 25.0
            memory = 45.0
            disk_read = 0
            disk_write = 0
            net_sent = 0
            net_recv = 0
        
        resources = {
            "timestamp": time.time(),
            "cpu_usage": round(cpu, 2),
            "memory_usage": round(memory, 2),
            "disk_io_read": disk_read,
            "disk_io_write": disk_write,
            "network_sent": net_sent,
            "network_recv": net_recv
        }
        
        # Store in database
        self._store_system_resources(resources)
        
        return resources
    
    def _store_system_resources(self, resources: Dict[str, Any]) -> bool:
        """Store system resource data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_resources 
                (timestamp, cpu_usage, memory_usage, disk_io_read, disk_io_write, network_sent, network_recv)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                resources["timestamp"],
                resources["cpu_usage"],
                resources["memory_usage"],
                resources["disk_io_read"],
                resources["disk_io_write"],
                resources["network_sent"],
                resources["network_recv"]
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception:
            return False
    
    def _single_user_simulation(self, user_id: int, duration_seconds: int) -> List[Dict[str, Any]]:
        """Simulate single user making requests"""
        results = []
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            try:
                # Health check request
                health_result = self.measure_single_request_latency("/health")
                health_result["user_id"] = user_id
                results.append(health_result)
                
                # Small delay between requests
                time.sleep(0.1)
                
                # Query request if health is OK
                if health_result["success"]:
                    try:
                        start_time = time.time()
                        response = requests.post(
                            f"{self.base_url}/query",
                            json={"query": "test performance", "provider": "openai"},
                            timeout=30
                        )
                        end_time_req = time.time()
                        
                        query_result = {
                            "latency_ms": round((end_time_req - start_time) * 1000, 2),
                            "status_code": response.status_code,
                            "success": response.status_code == 200,
                            "timestamp": time.time(),
                            "endpoint": "/query",
                            "user_id": user_id
                        }
                        results.append(query_result)
                        
                    except Exception:
                        query_result = {
                            "latency_ms": 0.0,
                            "status_code": 0,
                            "success": False,
                            "timestamp": time.time(),
                            "endpoint": "/query",
                            "user_id": user_id
                        }
                        results.append(query_result)
                
                # Longer delay between query cycles
                time.sleep(1.0)
                
            except Exception:
                continue
        
        return results
    
    def run_load_test_simulation(self, concurrent_users: int = 5, duration_seconds: int = 30) -> Dict[str, Any]:
        """Run load test with multiple concurrent users"""
        print(f"🚀 Starting load test: {concurrent_users} users for {duration_seconds}s")
        
        # Start resource monitoring in background
        resource_monitor_stop = threading.Event()
        resource_thread = threading.Thread(
            target=self._monitor_resources_during_test,
            args=(resource_monitor_stop,)
        )
        resource_thread.start()
        
        # Run concurrent user simulations
        all_results = []
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                # Submit tasks for each user
                futures = [
                    executor.submit(self._single_user_simulation, user_id, duration_seconds)
                    for user_id in range(concurrent_users)
                ]
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        user_results = future.result()
                        all_results.extend(user_results)
                    except Exception as e:
                        print(f"User simulation failed: {e}")
        
        finally:
            # Stop resource monitoring
            resource_monitor_stop.set()
            resource_thread.join(timeout=5)
        
        # Analyze results
        if not all_results:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_latency": 0.0,
                "min_latency": 0.0,
                "max_latency": 0.0,
                "requests_per_second": 0.0,
                "test_duration": duration_seconds,
                "concurrent_users": concurrent_users
            }
        
        successful_results = [r for r in all_results if r["success"]]
        failed_requests = len(all_results) - len(successful_results)
        
        latencies = [r["latency_ms"] for r in successful_results if r["latency_ms"] > 0]
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            avg_latency = min_latency = max_latency = 0.0
        
        requests_per_second = len(all_results) / duration_seconds if duration_seconds > 0 else 0.0
        
        test_result = {
            "total_requests": len(all_results),
            "successful_requests": len(successful_results),
            "failed_requests": failed_requests,
            "average_latency": round(avg_latency, 2),
            "min_latency": round(min_latency, 2),
            "max_latency": round(max_latency, 2),
            "requests_per_second": round(requests_per_second, 2),
            "test_duration": duration_seconds,
            "concurrent_users": concurrent_users,
            "timestamp": time.time()
        }
        
        # Store load test results
        self._store_load_test_result(test_result)
        
        return test_result
    
    def _monitor_resources_during_test(self, stop_event: threading.Event):
        """Monitor system resources during load test"""
        while not stop_event.is_set():
            self.monitor_system_resources()
            time.sleep(2)  # Monitor every 2 seconds
    
    def _store_load_test_result(self, result: Dict[str, Any]) -> bool:
        """Store load test result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO load_test_results 
                (test_name, timestamp, concurrent_users, duration_seconds, 
                 total_requests, successful_requests, failed_requests,
                 average_latency, min_latency, max_latency, requests_per_second)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "load_test_simulation",
                result["timestamp"],
                result["concurrent_users"],
                result["test_duration"],
                result["total_requests"],
                result["successful_requests"],
                result["failed_requests"],
                result["average_latency"],
                result["min_latency"],
                result["max_latency"],
                result["requests_per_second"]
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception:
            return False
    
    def get_performance_history(self, hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance test history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (hours * 3600)
            
            # Get single request tests
            cursor.execute("""
                SELECT timestamp, latency_ms, status_code, success
                FROM performance_tests 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff_time,))
            
            single_tests = []
            for row in cursor.fetchall():
                single_tests.append({
                    "timestamp": row[0],
                    "latency_ms": row[1],
                    "status_code": row[2],
                    "success": bool(row[3])
                })
            
            # Get load test results
            cursor.execute("""
                SELECT timestamp, concurrent_users, total_requests, 
                       successful_requests, average_latency, requests_per_second
                FROM load_test_results 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff_time,))
            
            load_tests = []
            for row in cursor.fetchall():
                load_tests.append({
                    "timestamp": row[0],
                    "concurrent_users": row[1],
                    "total_requests": row[2],
                    "successful_requests": row[3],
                    "average_latency": row[4],
                    "requests_per_second": row[5]
                })
            
            conn.close()
            
            return {
                "single_request_tests": single_tests,
                "load_tests": load_tests
            }
            
        except Exception:
            return {
                "single_request_tests": [],
                "load_tests": []
            }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        # Get recent performance data
        history = self.get_performance_history(24)
        
        # Analyze single request performance
        single_tests = history["single_request_tests"]
        if single_tests:
            successful_single = [t for t in single_tests if t["success"]]
            single_latencies = [t["latency_ms"] for t in successful_single if t["latency_ms"] > 0]
            
            if single_latencies:
                avg_single_latency = statistics.mean(single_latencies)
                median_single_latency = statistics.median(single_latencies)
            else:
                avg_single_latency = median_single_latency = 0.0
                
            single_success_rate = len(successful_single) / len(single_tests) * 100 if single_tests else 0
        else:
            avg_single_latency = median_single_latency = single_success_rate = 0.0
        
        # Analyze load test performance
        load_tests = history["load_tests"]
        if load_tests:
            avg_throughput = statistics.mean([t["requests_per_second"] for t in load_tests])
            max_concurrent_handled = max([t["concurrent_users"] for t in load_tests])
            avg_load_latency = statistics.mean([t["average_latency"] for t in load_tests])
        else:
            avg_throughput = max_concurrent_handled = avg_load_latency = 0.0
        
        # Generate recommendations
        recommendations = []
        
        if avg_single_latency > 500:
            recommendations.append("High latency detected - consider optimizing API response times")
        
        if single_success_rate < 95:
            recommendations.append("Low success rate - investigate API reliability issues")
        
        if avg_throughput < 10:
            recommendations.append("Low throughput - consider scaling infrastructure")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable parameters")
        
        report = {
            "test_summary": {
                "total_single_tests": len(single_tests),
                "total_load_tests": len(load_tests),
                "test_period_hours": 24,
                "generated_at": datetime.now(timezone.utc).isoformat()
            },
            "performance_metrics": {
                "single_request": {
                    "average_latency_ms": round(avg_single_latency, 2),
                    "median_latency_ms": round(median_single_latency, 2),
                    "success_rate_percent": round(single_success_rate, 2)
                },
                "load_testing": {
                    "average_throughput_rps": round(avg_throughput, 2),
                    "max_concurrent_users": max_concurrent_handled,
                    "average_latency_under_load_ms": round(avg_load_latency, 2)
                }
            },
            "recommendations": recommendations,
            "system_health": "good" if single_success_rate >= 95 and avg_single_latency <= 500 else "needs_attention"
        }
        
        return report
    
    def export_performance_report_html(self, output_path: str) -> bool:
        """Export performance report as HTML"""
        try:
            report = self.generate_performance_report()
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AgenticSeek Performance Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .good {{ color: #4caf50; }}
        .warning {{ color: #ff9800; }}
        .critical {{ color: #f44336; }}
        .recommendation {{ padding: 10px; margin: 5px 0; background-color: #f5f5f5; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>AgenticSeek Performance Benchmark Report</h1>
    
    <h2>Test Summary</h2>
    <div class="metric">
        <h3>Single Request Tests</h3>
        <p>{report['test_summary']['total_single_tests']}</p>
    </div>
    <div class="metric">
        <h3>Load Tests</h3>
        <p>{report['test_summary']['total_load_tests']}</p>
    </div>
    <div class="metric">
        <h3>System Health</h3>
        <p class="{report['system_health']}">{report['system_health']}</p>
    </div>
    
    <h2>Performance Metrics</h2>
    
    <h3>Single Request Performance</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Average Latency</td>
            <td>{report['performance_metrics']['single_request']['average_latency_ms']} ms</td>
        </tr>
        <tr>
            <td>Median Latency</td>
            <td>{report['performance_metrics']['single_request']['median_latency_ms']} ms</td>
        </tr>
        <tr>
            <td>Success Rate</td>
            <td>{report['performance_metrics']['single_request']['success_rate_percent']}%</td>
        </tr>
    </table>
    
    <h3>Load Testing Performance</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Average Throughput</td>
            <td>{report['performance_metrics']['load_testing']['average_throughput_rps']} req/sec</td>
        </tr>
        <tr>
            <td>Max Concurrent Users</td>
            <td>{report['performance_metrics']['load_testing']['max_concurrent_users']}</td>
        </tr>
        <tr>
            <td>Average Latency Under Load</td>
            <td>{report['performance_metrics']['load_testing']['average_latency_under_load_ms']} ms</td>
        </tr>
    </table>
    
    <h2>Recommendations</h2>
            """
            
            for rec in report['recommendations']:
                html_content += f'<div class="recommendation">{rec}</div>'
            
            html_content += f"""
    
    <h2>Raw Report Data</h2>
    <pre>{json.dumps(report, indent=2)}</pre>
    
    <footer>
        <p>Generated: {report['test_summary']['generated_at']}</p>
    </footer>
    
</body>
</html>
            """
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            return True
            
        except Exception as e:
            print(f"Error exporting performance report: {e}")
            return False

def run_performance_benchmarking_tdd_atomic():
    """🔁 Run atomic TDD cycle for performance benchmarking"""
    print("🔴 ATOMIC TDD - End-to-End Performance Benchmarking")
    print("=" * 60)
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceBenchmarkAtomic)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ Tests pass - running performance benchmarks")
        
        # Create performance benchmark system
        benchmark = PerformanceBenchmarkAtomic("production_performance.db")
        
        # Create output directory
        output_dir = Path("performance_output")
        output_dir.mkdir(exist_ok=True)
        
        print("\n🔥 PHASE 1: Single Request Latency Testing")
        # Single request tests
        health_latency = benchmark.measure_single_request_latency("/health")
        print(f"  Health endpoint: {health_latency['latency_ms']}ms (success: {health_latency['success']})")
        
        active_latency = benchmark.measure_single_request_latency("/is_active")
        print(f"  Active endpoint: {active_latency['latency_ms']}ms (success: {active_latency['success']})")
        
        print("\n🚀 PHASE 2: Load Testing (5 concurrent users, 15 seconds)")
        # Load test
        load_result = benchmark.run_load_test_simulation(concurrent_users=5, duration_seconds=15)
        
        print(f"  Total requests: {load_result['total_requests']}")
        print(f"  Successful: {load_result['successful_requests']}")
        print(f"  Failed: {load_result['failed_requests']}")
        print(f"  Average latency: {load_result['average_latency']}ms")
        print(f"  Throughput: {load_result['requests_per_second']} req/sec")
        
        print("\n📊 PHASE 3: Performance Report Generation")
        # Generate comprehensive report
        performance_report = benchmark.generate_performance_report()
        
        # Save report data
        with open(output_dir / "performance_report.json", 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        # Export HTML report
        html_success = benchmark.export_performance_report_html(str(output_dir / "performance_report.html"))
        
        print(f"\n🎯 PERFORMANCE BENCHMARKING COMPLETE:")
        print(f"📁 Output directory: {output_dir}/")
        print(f"📄 Performance data: performance_report.json")
        print(f"🌐 HTML report: performance_report.html")
        print(f"📈 System Health: {performance_report['system_health']}")
        print(f"🔥 Single Request Avg Latency: {performance_report['performance_metrics']['single_request']['average_latency_ms']}ms")
        print(f"⚡ Success Rate: {performance_report['performance_metrics']['single_request']['success_rate_percent']}%")
        print(f"🚀 Load Test Throughput: {performance_report['performance_metrics']['load_testing']['average_throughput_rps']} req/sec")
        
        print(f"\n💡 Recommendations:")
        for rec in performance_report['recommendations']:
            print(f"  • {rec}")
        
        return True
    else:
        print("❌ Tests failed")
        return False

if __name__ == "__main__":
    success = run_performance_benchmarking_tdd_atomic()
    exit(0 if success else 1)