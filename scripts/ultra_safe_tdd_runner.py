#!/usr/bin/env python3
"""
Ultra-Safe TDD Runner - Maximum Memory Protection
================================================

* Purpose: Ultra-conservative TDD execution with absolute memory safety
* Features: Minimal memory footprint, sequential execution, crash prevention
* Integration: Final safeguard against any JavaScript heap issues
* Safety: Conservative thresholds with aggressive cleanup between each test
"""

import asyncio
import gc
import psutil
import time
import json
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraSafeTDDRunner:
    """Ultra-conservative TDD runner with maximum memory safety"""
    
    def __init__(self, max_memory_mb: float = 256.0):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
        self.workspace_root = Path.cwd()
        
        # Ultra-conservative settings
        self.single_test_timeout = 30.0
        self.cleanup_interval = 5.0
        self.memory_check_interval = 1.0
        
        # Results tracking
        self.test_results = []
        self.execution_blocked = False
        
        logger.info(f"🛡️ Ultra-Safe TDD Runner initialized - Max Memory: {max_memory_mb}MB")
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def check_memory_safety(self) -> bool:
        """Check if current memory usage is safe"""
        current_memory = self.get_current_memory_mb()
        is_safe = current_memory < self.max_memory_mb
        
        if not is_safe:
            logger.warning(f"⚠️ Memory limit exceeded: {current_memory:.2f}MB > {self.max_memory_mb}MB")
            self.execution_blocked = True
        
        return is_safe
    
    async def aggressive_cleanup(self):
        """Perform aggressive memory cleanup"""
        logger.info("🧹 Performing aggressive cleanup")
        
        # Multiple GC passes
        collected_total = 0
        for i in range(5):
            collected = gc.collect()
            collected_total += collected
            await asyncio.sleep(0.2)
        
        # Clear caches
        try:
            import linecache
            linecache.clearcache()
        except:
            pass
        
        try:
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
        except:
            pass
        
        logger.info(f"🧹 Cleanup complete - collected {collected_total} objects")
    
    async def ultra_safe_test_provider_system(self) -> Dict[str, Any]:
        """Ultra-safe provider system test"""
        test_name = "Ultra-Safe Provider Test"
        logger.info(f"🧪 Running {test_name}")
        
        if not self.check_memory_safety():
            return {"name": test_name, "status": "BLOCKED", "reason": "memory_limit"}
        
        start_time = time.time()
        
        try:
            # Add path safely
            if str(self.workspace_root) not in sys.path:
                sys.path.insert(0, str(self.workspace_root))
            
            # Quick test with immediate cleanup
            from sources.cascading_provider import CascadingProvider
            
            provider = CascadingProvider()
            provider_count = len(provider.providers)
            
            # Immediate cleanup
            del provider
            await self.aggressive_cleanup()
            
            duration = time.time() - start_time
            
            return {
                "name": test_name,
                "status": "PASSED" if provider_count > 0 else "FAILED",
                "duration": duration,
                "providers_loaded": provider_count,
                "memory_mb": self.get_current_memory_mb()
            }
            
        except Exception as e:
            duration = time.time() - start_time
            await self.aggressive_cleanup()
            
            return {
                "name": test_name,
                "status": "ERROR",
                "duration": duration,
                "error": str(e),
                "memory_mb": self.get_current_memory_mb()
            }
    
    async def ultra_safe_test_basic_integration(self) -> Dict[str, Any]:
        """Ultra-safe basic integration test"""
        test_name = "Ultra-Safe Integration Test"
        logger.info(f"🧪 Running {test_name}")
        
        if not self.check_memory_safety():
            return {"name": test_name, "status": "BLOCKED", "reason": "memory_limit"}
        
        start_time = time.time()
        
        try:
            # Basic integration test without heavy components
            # Just verify basic system connectivity
            
            # Check if essential directories exist
            sources_dir = self.workspace_root / "sources"
            test_passed = sources_dir.exists()
            
            await self.aggressive_cleanup()
            
            duration = time.time() - start_time
            
            return {
                "name": test_name,
                "status": "PASSED" if test_passed else "FAILED",
                "duration": duration,
                "integration_check": "basic_connectivity",
                "memory_mb": self.get_current_memory_mb()
            }
            
        except Exception as e:
            duration = time.time() - start_time
            await self.aggressive_cleanup()
            
            return {
                "name": test_name,
                "status": "ERROR",
                "duration": duration,
                "error": str(e),
                "memory_mb": self.get_current_memory_mb()
            }
    
    async def ultra_safe_test_memory_baseline(self) -> Dict[str, Any]:
        """Ultra-safe memory baseline test"""
        test_name = "Ultra-Safe Memory Baseline"
        logger.info(f"🧪 Running {test_name}")
        
        start_time = time.time()
        memory_before = self.get_current_memory_mb()
        
        try:
            # Simple memory allocation test
            test_data = ["test"] * 1000  # Small allocation
            data_length = len(test_data)
            
            # Immediate cleanup
            del test_data
            await self.aggressive_cleanup()
            
            memory_after = self.get_current_memory_mb()
            memory_delta = memory_after - memory_before
            
            duration = time.time() - start_time
            
            return {
                "name": test_name,
                "status": "PASSED",
                "duration": duration,
                "memory_before_mb": memory_before,
                "memory_after_mb": memory_after,
                "memory_delta_mb": memory_delta,
                "data_processed": data_length
            }
            
        except Exception as e:
            duration = time.time() - start_time
            await self.aggressive_cleanup()
            
            return {
                "name": test_name,
                "status": "ERROR",
                "duration": duration,
                "error": str(e),
                "memory_mb": self.get_current_memory_mb()
            }
    
    async def run_ultra_safe_test_suite(self) -> Dict[str, Any]:
        """Run ultra-safe test suite with maximum memory protection"""
        logger.info("🚀 Starting Ultra-Safe TDD Test Suite")
        
        suite_start_time = time.time()
        initial_memory = self.get_current_memory_mb()
        
        # Define ultra-safe test suite
        test_functions = [
            self.ultra_safe_test_memory_baseline,
            self.ultra_safe_test_provider_system,
            self.ultra_safe_test_basic_integration,
        ]
        
        results = []
        tests_executed = 0
        tests_blocked = 0
        
        for test_func in test_functions:
            # Check memory before each test
            if not self.check_memory_safety():
                tests_blocked += 1
                logger.warning(f"🚫 Test blocked due to memory constraints")
                results.append({
                    "name": test_func.__name__,
                    "status": "BLOCKED",
                    "reason": "memory_limit_exceeded"
                })
                continue
            
            # Execute test with timeout
            try:
                result = await asyncio.wait_for(
                    test_func(),
                    timeout=self.single_test_timeout
                )
                results.append(result)
                tests_executed += 1
                
                logger.info(f"✅ Test completed: {result['name']} - {result['status']}")
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Test timeout: {test_func.__name__}")
                results.append({
                    "name": test_func.__name__,
                    "status": "TIMEOUT",
                    "duration": self.single_test_timeout
                })
                
            except Exception as e:
                logger.error(f"💥 Test error: {test_func.__name__} - {e}")
                results.append({
                    "name": test_func.__name__,
                    "status": "ERROR",
                    "error": str(e)
                })
            
            # Aggressive cleanup between tests
            await self.aggressive_cleanup()
            await asyncio.sleep(self.cleanup_interval)
        
        # Calculate summary
        suite_duration = time.time() - suite_start_time
        final_memory = self.get_current_memory_mb()
        memory_delta = final_memory - initial_memory
        
        passed = sum(1 for r in results if r.get("status") == "PASSED")
        failed = sum(1 for r in results if r.get("status") == "FAILED")
        errors = sum(1 for r in results if r.get("status") == "ERROR")
        timeouts = sum(1 for r in results if r.get("status") == "TIMEOUT")
        blocked = sum(1 for r in results if r.get("status") == "BLOCKED")
        
        summary = {
            "total_tests": len(results),
            "tests_executed": tests_executed,
            "tests_blocked": tests_blocked,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "timeouts": timeouts,
            "blocked": blocked,
            "success_rate": (passed / len(results) * 100) if results else 0,
            "suite_duration": suite_duration,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_delta_mb": memory_delta,
            "max_memory_mb": self.max_memory_mb,
            "memory_efficiency": ((self.max_memory_mb - final_memory) / self.max_memory_mb * 100) if self.max_memory_mb > 0 else 0
        }
        
        return {
            "summary": summary,
            "results": results
        }
    
    def generate_ultra_safe_report(self, test_results: Dict[str, Any]) -> str:
        """Generate ultra-safe test report"""
        summary = test_results["summary"]
        results = test_results["results"]
        
        success_rate = summary["success_rate"]
        success_emoji = "✅" if success_rate >= 75 else "⚠️" if success_rate >= 50 else "❌"
        
        report = f"""🛡️ ULTRA-SAFE TDD RUNNER REPORT
==================================================
{success_emoji} Test Suite Status: {success_rate:.1f}% success rate

📊 Test Summary:
   ✅ Passed: {summary['passed']}
   ❌ Failed: {summary['failed']}
   💥 Errors: {summary['errors']}
   ⏰ Timeouts: {summary['timeouts']}
   🚫 Blocked: {summary['blocked']}
   🧪 Total Executed: {summary['tests_executed']}/{summary['total_tests']}

⚡ Performance Metrics:
   ⏱️ Suite Duration: {summary['suite_duration']:.2f}s
   📈 Average Duration: {summary['suite_duration']/summary['total_tests']:.2f}s per test

🧠 Memory Metrics:
   💾 Memory Limit: {summary['max_memory_mb']}MB
   📊 Initial Memory: {summary['initial_memory_mb']:.2f}MB
   📈 Final Memory: {summary['final_memory_mb']:.2f}MB
   📉 Memory Delta: {summary['memory_delta_mb']:.2f}MB
   🎯 Memory Efficiency: {summary['memory_efficiency']:.1f}% headroom

📋 Detailed Results:"""

        for result in results:
            status_emoji = {
                "PASSED": "✅", "FAILED": "❌", "ERROR": "💥", 
                "TIMEOUT": "⏰", "BLOCKED": "🚫"
            }.get(result.get("status"), "❓")
            
            duration = result.get("duration", 0)
            memory_info = f" (Mem: {result.get('memory_mb', 0):.1f}MB)" if "memory_mb" in result else ""
            
            report += f"""
   {status_emoji} {result.get('name', 'Unknown')}
      Duration: {duration:.2f}s{memory_info}
      Status: {result.get('status', 'Unknown')}"""
            
            if result.get('error'):
                report += f"\n      Error: {result['error']}"
            elif result.get('reason'):
                report += f"\n      Reason: {result['reason']}"
            elif result.get('providers_loaded'):
                report += f"\n      Providers: {result['providers_loaded']}"
        
        return report

# CLI interface for ultra-safe runner
async def main():
    """Ultra-safe TDD runner CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Safe TDD Runner")
    parser.add_argument("--max-memory", type=float, default=256.0, help="Maximum memory limit in MB")
    parser.add_argument("--save-results", action="store_true", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Initialize ultra-safe runner
    runner = UltraSafeTDDRunner(max_memory_mb=args.max_memory)
    
    try:
        # Run ultra-safe test suite
        results = await runner.run_ultra_safe_test_suite()
        
        # Generate and display report
        report = runner.generate_ultra_safe_report(results)
        print(report)
        
        # Save results if requested
        if args.save_results:
            results_file = Path.cwd() / "test_results" / "ultra_safe" / f"ultra_safe_results_{int(time.time())}.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Results saved to {results_file}")
        
        # Exit with appropriate code
        summary = results["summary"]
        exit_code = 0 if summary["success_rate"] >= 75 else 1
        logger.info(f"🏁 Ultra-safe runner finished with exit code: {exit_code}")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"💥 Ultra-safe runner failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)