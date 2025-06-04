#!/usr/bin/env python3
"""
Production-Ready TDD Pipeline for AgenticSeek
=============================================

* Purpose: Complete TDD pipeline with atomic processes and memory safety
* Features: Automated testing, memory management, CI/CD integration, reporting
* Integration: Production deployment with comprehensive safeguards
* Safety: Memory leak prevention, resource management, crash protection
"""

import asyncio
import json
import time
import subprocess
import gc
import psutil
import tracemalloc
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field
import sys
import os
import uuid
from datetime import datetime

# Import our memory-safe frameworks
from memory_safe_atomic_tdd import MemorySafeAtomicTDDFramework, MemoryMetrics
from memory_safe_comprehensive_test import MemorySafeTestRunner

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler('production_tdd_pipeline.log', maxBytes=2*1024*1024, backupCount=5)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfiguration:
    """Configuration for production TDD pipeline"""
    memory_limit_mb: float = 2048.0
    test_timeout_seconds: float = 300.0
    max_concurrent_tests: int = 2
    enable_atomic_testing: bool = True
    enable_comprehensive_testing: bool = True
    enable_memory_monitoring: bool = True
    enable_performance_profiling: bool = True
    cleanup_interval_seconds: float = 30.0
    max_test_duration_seconds: float = 600.0

@dataclass
class PipelineResults:
    """Results from production TDD pipeline execution"""
    pipeline_id: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    atomic_results: Optional[Dict[str, Any]] = None
    comprehensive_results: Optional[Dict[str, Any]] = None
    memory_metrics: Optional[MemoryMetrics] = None
    performance_profile: Optional[Dict[str, Any]] = None
    success: bool = False
    error_details: Optional[str] = None

class ProductionTDDPipeline:
    """Production-ready TDD pipeline with comprehensive testing and memory safety"""
    
    def __init__(self, config: Optional[PipelineConfiguration] = None):
        self.config = config or PipelineConfiguration()
        self.pipeline_id = f"tdd_pipeline_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        self.workspace_root = Path.cwd()
        
        # Initialize memory monitoring
        self.process = psutil.Process()
        self.memory_monitor_active = False
        self.peak_memory_mb = 0
        
        # Initialize frameworks
        self.atomic_framework = None
        self.comprehensive_runner = None
        
        # Results storage
        self.results_dir = self.workspace_root / "test_results" / "production_pipeline"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Production TDD Pipeline initialized - ID: {self.pipeline_id}")
        logger.info(f"📊 Configuration: Memory limit {self.config.memory_limit_mb}MB, "
                   f"Timeout {self.config.test_timeout_seconds}s")
    
    def start_memory_monitoring(self):
        """Start comprehensive memory monitoring"""
        if self.config.enable_memory_monitoring:
            tracemalloc.start()
            self.memory_monitor_active = True
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            self.peak_memory_mb = initial_memory
            logger.info(f"📊 Memory monitoring started - Initial: {initial_memory:.2f}MB")
    
    def stop_memory_monitoring(self) -> MemoryMetrics:
        """Stop memory monitoring and return metrics"""
        if not self.memory_monitor_active:
            return MemoryMetrics(0, 0, 0, 0, 0, 0)
        
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        
        # Get tracemalloc stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Force garbage collection and measure
        gc_before = len(gc.get_objects())
        collected = gc.collect()
        gc_after = len(gc.get_objects())
        
        metrics = MemoryMetrics(
            peak_memory_mb=max(peak / 1024 / 1024, self.peak_memory_mb),
            current_memory_mb=current / 1024 / 1024,
            memory_percent=memory_percent,
            gc_collections=collected,
            start_time=time.time(),
            duration=0
        )
        
        self.memory_monitor_active = False
        logger.info(f"📊 Memory monitoring complete - Peak: {metrics.peak_memory_mb:.2f}MB, "
                   f"Collected: {collected} objects")
        
        return metrics
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resources before pipeline execution"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        cpu_percent = self.process.cpu_percent(interval=1)
        
        # Check disk space
        disk_usage = psutil.disk_usage(str(self.workspace_root))
        disk_free_gb = disk_usage.free / 1024 / 1024 / 1024
        
        resources = {
            "memory_mb": memory_mb,
            "memory_percent": memory_percent,
            "cpu_percent": cpu_percent,
            "disk_free_gb": disk_free_gb,
            "memory_available": memory_mb < self.config.memory_limit_mb,
            "disk_available": disk_free_gb > 1.0,  # At least 1GB free
            "system_ready": True
        }
        
        # Check resource constraints
        if not resources["memory_available"]:
            logger.warning(f"⚠️ Memory constraint: {memory_mb:.1f}MB > {self.config.memory_limit_mb}MB")
            resources["system_ready"] = False
        
        if not resources["disk_available"]:
            logger.warning(f"⚠️ Disk space constraint: {disk_free_gb:.1f}GB available")
            resources["system_ready"] = False
        
        return resources
    
    async def run_atomic_testing_phase(self) -> Optional[Dict[str, Any]]:
        """Run atomic testing phase with memory safety"""
        if not self.config.enable_atomic_testing:
            logger.info("⏭️ Atomic testing disabled in configuration")
            return None
        
        logger.info("🔬 Starting Atomic Testing Phase")
        
        try:
            # Initialize atomic framework with memory limits
            self.atomic_framework = MemorySafeAtomicTDDFramework(
                workspace_root=str(self.workspace_root),
                memory_limit_mb=self.config.memory_limit_mb * 0.6  # Use 60% of total memory
            )
            
            # Run atomic test suite
            atomic_results = await self.atomic_framework.run_memory_safe_test_suite()
            
            # Generate report
            report = self.atomic_framework.generate_memory_report(atomic_results)
            
            # Save atomic results
            atomic_results_file = self.results_dir / f"atomic_results_{self.pipeline_id}.json"
            with open(atomic_results_file, 'w') as f:
                json.dump({
                    "pipeline_id": self.pipeline_id,
                    "timestamp": datetime.now().isoformat(),
                    "results": {k: {
                        "test_id": v.test_id,
                        "status": v.status,
                        "duration": v.duration,
                        "isolation_verified": v.isolation_verified,
                        "dependencies_met": v.dependencies_met,
                        "state_preserved": v.state_preserved,
                        "cleanup_successful": v.cleanup_successful,
                        "memory_peak_mb": v.memory_metrics.peak_memory_mb if v.memory_metrics else 0
                    } for k, v in atomic_results.items()},
                    "report": report
                }, f, indent=2)
            
            logger.info(f"✅ Atomic testing completed - Results saved to {atomic_results_file}")
            
            return {
                "atomic_results": atomic_results,
                "report": report,
                "results_file": str(atomic_results_file)
            }
            
        except Exception as e:
            logger.error(f"💥 Atomic testing phase failed: {e}")
            return {"error": str(e), "phase": "atomic"}
        finally:
            # Cleanup atomic framework
            if self.atomic_framework:
                del self.atomic_framework
            gc.collect()
    
    async def run_comprehensive_testing_phase(self) -> Optional[Dict[str, Any]]:
        """Run comprehensive testing phase with memory safety"""
        if not self.config.enable_comprehensive_testing:
            logger.info("⏭️ Comprehensive testing disabled in configuration")
            return None
        
        logger.info("🧪 Starting Comprehensive Testing Phase")
        
        try:
            # Initialize comprehensive test runner
            self.comprehensive_runner = MemorySafeTestRunner(
                memory_limit_mb=self.config.memory_limit_mb * 0.8  # Use 80% of total memory
            )
            
            # Run comprehensive test suite
            comprehensive_results = await self.comprehensive_runner.run_comprehensive_memory_safe_suite()
            
            # Generate report
            report = self.comprehensive_runner.generate_memory_safe_report(comprehensive_results)
            
            # Save comprehensive results
            comprehensive_results_file = self.results_dir / f"comprehensive_results_{self.pipeline_id}.json"
            with open(comprehensive_results_file, 'w') as f:
                json.dump({
                    "pipeline_id": self.pipeline_id,
                    "timestamp": datetime.now().isoformat(),
                    "results": comprehensive_results,
                    "report": report
                }, f, indent=2)
            
            logger.info(f"✅ Comprehensive testing completed - Results saved to {comprehensive_results_file}")
            
            return {
                "comprehensive_results": comprehensive_results,
                "report": report,
                "results_file": str(comprehensive_results_file)
            }
            
        except Exception as e:
            logger.error(f"💥 Comprehensive testing phase failed: {e}")
            return {"error": str(e), "phase": "comprehensive"}
        finally:
            # Cleanup comprehensive runner
            if self.comprehensive_runner:
                del self.comprehensive_runner
            gc.collect()
    
    def analyze_pipeline_success(self, atomic_results: Optional[Dict], comprehensive_results: Optional[Dict]) -> Tuple[bool, str]:
        """Analyze overall pipeline success"""
        success_criteria = []
        
        # Atomic testing criteria
        if atomic_results and "atomic_results" in atomic_results:
            atomic_test_results = atomic_results["atomic_results"]
            atomic_passed = sum(1 for r in atomic_test_results.values() if r.status == "PASSED")
            atomic_total = len(atomic_test_results)
            atomic_success_rate = (atomic_passed / atomic_total * 100) if atomic_total > 0 else 0
            
            success_criteria.append(f"Atomic: {atomic_success_rate:.1f}% ({atomic_passed}/{atomic_total})")
            
            # Require at least 50% success rate for atomic tests
            if atomic_success_rate < 50:
                return False, f"Atomic testing below threshold: {atomic_success_rate:.1f}% < 50%"
        
        # Comprehensive testing criteria
        if comprehensive_results and "comprehensive_results" in comprehensive_results:
            comp_summary = comprehensive_results["comprehensive_results"]["summary"]
            comp_success_rate = comp_summary["success_rate"]
            
            success_criteria.append(f"Comprehensive: {comp_success_rate:.1f}%")
            
            # Require at least 75% success rate for comprehensive tests
            if comp_success_rate < 75:
                return False, f"Comprehensive testing below threshold: {comp_success_rate:.1f}% < 75%"
        
        return True, f"Pipeline successful - {', '.join(success_criteria)}"
    
    async def execute_production_pipeline(self) -> PipelineResults:
        """Execute complete production TDD pipeline"""
        start_time = datetime.now()
        logger.info(f"🚀 Starting Production TDD Pipeline - ID: {self.pipeline_id}")
        
        try:
            # Start memory monitoring
            self.start_memory_monitoring()
            
            # Check system resources
            resources = self.check_system_resources()
            if not resources["system_ready"]:
                raise RuntimeError(f"System not ready: {resources}")
            
            logger.info(f"✅ System resources verified - Memory: {resources['memory_mb']:.1f}MB, "
                       f"CPU: {resources['cpu_percent']:.1f}%, Disk: {resources['disk_free_gb']:.1f}GB")
            
            # Phase 1: Atomic Testing
            atomic_results = await self.run_atomic_testing_phase()
            
            # Memory cleanup between phases
            gc.collect()
            await asyncio.sleep(1)
            
            # Phase 2: Comprehensive Testing
            comprehensive_results = await self.run_comprehensive_testing_phase()
            
            # Stop memory monitoring
            memory_metrics = self.stop_memory_monitoring()
            
            # Analyze success
            success, success_message = self.analyze_pipeline_success(atomic_results, comprehensive_results)
            
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            # Create pipeline results
            pipeline_results = PipelineResults(
                pipeline_id=self.pipeline_id,
                start_time=start_time,
                end_time=end_time,
                total_duration=total_duration,
                atomic_results=atomic_results,
                comprehensive_results=comprehensive_results,
                memory_metrics=memory_metrics,
                success=success,
                error_details=None if success else success_message
            )
            
            # Save pipeline results
            pipeline_results_file = self.results_dir / f"pipeline_results_{self.pipeline_id}.json"
            with open(pipeline_results_file, 'w') as f:
                json.dump({
                    "pipeline_id": self.pipeline_id,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "total_duration": total_duration,
                    "success": success,
                    "success_message": success_message,
                    "memory_peak_mb": memory_metrics.peak_memory_mb,
                    "memory_current_mb": memory_metrics.current_memory_mb,
                    "gc_collections": memory_metrics.gc_collections,
                    "atomic_phase": atomic_results is not None,
                    "comprehensive_phase": comprehensive_results is not None
                }, f, indent=2)
            
            logger.info(f"🎯 Pipeline completed - Success: {success}")
            logger.info(f"📊 Duration: {total_duration:.2f}s, Peak Memory: {memory_metrics.peak_memory_mb:.2f}MB")
            logger.info(f"💾 Results saved to {pipeline_results_file}")
            
            return pipeline_results
            
        except Exception as e:
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            logger.error(f"💥 Pipeline execution failed: {e}")
            
            # Stop memory monitoring if still active
            memory_metrics = self.stop_memory_monitoring()
            
            return PipelineResults(
                pipeline_id=self.pipeline_id,
                start_time=start_time,
                end_time=end_time,
                total_duration=total_duration,
                memory_metrics=memory_metrics,
                success=False,
                error_details=str(e)
            )
        finally:
            # Final cleanup
            gc.collect()
    
    def generate_production_report(self, results: PipelineResults) -> str:
        """Generate comprehensive production pipeline report"""
        status_emoji = "✅" if results.success else "❌"
        
        report = f"""🚀 PRODUCTION TDD PIPELINE REPORT
==================================================
{status_emoji} Pipeline ID: {results.pipeline_id}
📅 Execution Time: {results.start_time.strftime('%Y-%m-%d %H:%M:%S')}
⏱️ Total Duration: {results.total_duration:.2f}s
🎯 Success: {results.success}

🧠 Memory Metrics:
   📊 Peak Usage: {results.memory_metrics.peak_memory_mb:.2f}MB
   💾 Current Usage: {results.memory_metrics.current_memory_mb:.2f}MB
   🧹 GC Collections: {results.memory_metrics.gc_collections}

📋 Phase Results:"""

        if results.atomic_results:
            if "atomic_results" in results.atomic_results:
                atomic_test_results = results.atomic_results["atomic_results"]
                atomic_passed = sum(1 for r in atomic_test_results.values() if r.status == "PASSED")
                atomic_total = len(atomic_test_results)
                atomic_success_rate = (atomic_passed / atomic_total * 100) if atomic_total > 0 else 0
                
                report += f"""
   🔬 Atomic Testing: {atomic_success_rate:.1f}% ({atomic_passed}/{atomic_total})"""
            else:
                report += f"""
   🔬 Atomic Testing: ❌ Failed - {results.atomic_results.get('error', 'Unknown error')}"""
        else:
            report += """
   🔬 Atomic Testing: ⏭️ Skipped"""

        if results.comprehensive_results:
            if "comprehensive_results" in results.comprehensive_results:
                comp_summary = results.comprehensive_results["comprehensive_results"]["summary"]
                comp_success_rate = comp_summary["success_rate"]
                comp_passed = comp_summary["passed"]
                comp_total = comp_summary["total_tests"]
                
                report += f"""
   🧪 Comprehensive Testing: {comp_success_rate:.1f}% ({comp_passed}/{comp_total})"""
            else:
                report += f"""
   🧪 Comprehensive Testing: ❌ Failed - {results.comprehensive_results.get('error', 'Unknown error')}"""
        else:
            report += """
   🧪 Comprehensive Testing: ⏭️ Skipped"""

        if not results.success and results.error_details:
            report += f"""

❌ Failure Details:
   {results.error_details}"""

        report += f"""

📁 Results Directory: {self.results_dir}
💾 Detailed results saved in individual phase files"""

        return report

# CLI interface for production pipeline
async def main():
    """Main CLI interface for production TDD pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production TDD Pipeline for AgenticSeek")
    parser.add_argument("--memory-limit", type=float, default=2048.0, help="Memory limit in MB")
    parser.add_argument("--timeout", type=float, default=300.0, help="Test timeout in seconds")
    parser.add_argument("--skip-atomic", action="store_true", help="Skip atomic testing phase")
    parser.add_argument("--skip-comprehensive", action="store_true", help="Skip comprehensive testing phase")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = PipelineConfiguration(
        memory_limit_mb=args.memory_limit,
        test_timeout_seconds=args.timeout,
        enable_atomic_testing=not args.skip_atomic,
        enable_comprehensive_testing=not args.skip_comprehensive
    )
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize and run pipeline
    pipeline = ProductionTDDPipeline(config)
    
    try:
        results = await pipeline.execute_production_pipeline()
        
        # Generate and display report
        report = pipeline.generate_production_report(results)
        print(report)
        
        # Exit with appropriate code
        exit_code = 0 if results.success else 1
        logger.info(f"🏁 Pipeline finished with exit code: {exit_code}")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"💥 Pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)