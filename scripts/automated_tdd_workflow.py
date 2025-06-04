#!/usr/bin/env python3
"""
Automated TDD Workflow Integration System
========================================

* Purpose: Fully automated TDD workflow with CI/CD integration and memory monitoring
* Features: Git hooks, automated testing, continuous monitoring, deployment gating
* Safety: Advanced memory protection with terminal crash prevention
* Integration: Production-ready workflow automation for AgenticSeek
"""

import asyncio
import json
import time
import subprocess
import os
import gc
import psutil
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
from contextlib import asynccontextmanager
import sqlite3
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
import yaml

# Configure automated workflow logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_tdd_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class WorkflowConfig:
    """Automated workflow configuration"""
    memory_limit_mb: float = 512.0
    test_timeout_seconds: float = 300.0
    critical_test_timeout: float = 60.0
    memory_check_interval: float = 5.0
    max_concurrent_tests: int = 3
    enable_git_hooks: bool = True
    enable_ci_cd_integration: bool = True
    enable_performance_monitoring: bool = True
    emergency_cleanup_threshold: float = 0.9  # 90% memory usage

@dataclass
class WorkflowResult:
    """Automated workflow execution result"""
    workflow_id: str
    status: str  # SUCCESS, FAILURE, TIMEOUT, MEMORY_EXCEEDED
    duration: float
    tests_executed: int
    tests_passed: int
    tests_failed: int
    peak_memory_mb: float
    git_commit: Optional[str]
    deployment_allowed: bool
    error_details: Optional[str] = None

class MemoryMonitor:
    """Advanced memory monitoring for automated workflows"""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.monitoring_active = False
        self.memory_samples = []
        self.process = psutil.Process()
        self.alert_callbacks = []
        self.emergency_callbacks = []
        
    def start_monitoring(self):
        """Start continuous memory monitoring"""
        self.monitoring_active = True
        self.memory_samples = []
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    memory_percent = memory_mb / self.config.memory_limit_mb
                    
                    self.memory_samples.append({
                        'timestamp': time.time(),
                        'memory_mb': memory_mb,
                        'memory_percent': memory_percent
                    })
                    
                    # Check thresholds
                    if memory_percent > self.config.emergency_cleanup_threshold:
                        logger.error(f"🚨 EMERGENCY: Memory usage {memory_mb:.1f}MB ({memory_percent:.1%})")
                        for callback in self.emergency_callbacks:
                            callback(memory_mb, memory_percent)
                    elif memory_percent > 0.8:  # 80% warning
                        logger.warning(f"⚠️ Memory warning: {memory_mb:.1f}MB ({memory_percent:.1%})")
                        for callback in self.alert_callbacks:
                            callback(memory_mb, memory_percent)
                    
                    # Keep only recent samples
                    if len(self.memory_samples) > 100:
                        self.memory_samples = self.memory_samples[-50:]
                    
                    time.sleep(self.config.memory_check_interval)
                    
                except Exception as e:
                    logger.error(f"❌ Memory monitoring error: {e}")
                    time.sleep(self.config.memory_check_interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("📊 Memory monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics"""
        self.monitoring_active = False
        
        if not self.memory_samples:
            return {"error": "No memory samples collected"}
        
        peak_memory = max(sample['memory_mb'] for sample in self.memory_samples)
        avg_memory = sum(sample['memory_mb'] for sample in self.memory_samples) / len(self.memory_samples)
        final_memory = self.memory_samples[-1]['memory_mb']
        
        return {
            "peak_memory_mb": peak_memory,
            "average_memory_mb": avg_memory,
            "final_memory_mb": final_memory,
            "sample_count": len(self.memory_samples),
            "monitoring_duration": self.memory_samples[-1]['timestamp'] - self.memory_samples[0]['timestamp']
        }
    
    def add_alert_callback(self, callback):
        """Add memory alert callback"""
        self.alert_callbacks.append(callback)
    
    def add_emergency_callback(self, callback):
        """Add emergency cleanup callback"""
        self.emergency_callbacks.append(callback)

class GitIntegration:
    """Git integration for automated TDD workflows"""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.hooks_dir = workspace_root / ".git" / "hooks"
        
    def install_git_hooks(self) -> bool:
        """Install automated TDD git hooks"""
        try:
            if not self.hooks_dir.exists():
                logger.error("❌ Git hooks directory not found")
                return False
            
            # Pre-commit hook
            pre_commit_script = """#!/bin/bash
# Automated TDD Pre-commit Hook
echo "🔍 Running automated TDD pre-commit checks..."
cd "$(git rev-parse --show-toplevel)"
python scripts/automated_tdd_workflow.py --mode=pre-commit
exit $?
"""
            
            pre_commit_path = self.hooks_dir / "pre-commit"
            with open(pre_commit_path, 'w') as f:
                f.write(pre_commit_script)
            pre_commit_path.chmod(0o755)
            
            # Pre-push hook
            pre_push_script = """#!/bin/bash
# Automated TDD Pre-push Hook
echo "🚀 Running automated TDD pre-push validation..."
cd "$(git rev-parse --show-toplevel)"
python scripts/automated_tdd_workflow.py --mode=pre-push
exit $?
"""
            
            pre_push_path = self.hooks_dir / "pre-push"
            with open(pre_push_path, 'w') as f:
                f.write(pre_push_script)
            pre_push_path.chmod(0o755)
            
            logger.info("✅ Git hooks installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to install git hooks: {e}")
            return False
    
    def get_current_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
                cwd=self.workspace_root
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def get_changed_files(self, base_commit: str = "HEAD~1") -> List[str]:
        """Get list of changed files"""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", base_commit],
                capture_output=True, text=True, timeout=10,
                cwd=self.workspace_root
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.split('\n') if f.strip()]
            return []
        except Exception:
            return []
    
    def create_workflow_branch(self, workflow_id: str) -> bool:
        """Create isolated workflow branch"""
        try:
            branch_name = f"tdd-workflow-{workflow_id}"
            result = subprocess.run(
                ["git", "checkout", "-b", branch_name],
                capture_output=True, text=True, timeout=10,
                cwd=self.workspace_root
            )
            return result.returncode == 0
        except Exception:
            return False

class AutomatedTDDWorkflow:
    """Comprehensive automated TDD workflow system"""
    
    def __init__(self, config: WorkflowConfig, workspace_root: str = "."):
        self.config = config
        self.workspace_root = Path(workspace_root)
        self.workflow_id = f"workflow_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        # Initialize components
        self.memory_monitor = MemoryMonitor(config)
        self.git_integration = GitIntegration(self.workspace_root)
        
        # State management
        self.workflow_db = self.workspace_root / "automated_tdd_workflow.db"
        self.results_dir = self.workspace_root / "workflow_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Emergency cleanup flag
        self.emergency_cleanup_triggered = False
        
        # Setup emergency callbacks
        self.memory_monitor.add_emergency_callback(self._emergency_cleanup)
        
        self._init_workflow_db()
        logger.info(f"🤖 Automated TDD Workflow initialized - ID: {self.workflow_id}")
    
    def _init_workflow_db(self):
        """Initialize workflow database"""
        with sqlite3.connect(self.workflow_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_executions (
                    workflow_id TEXT PRIMARY KEY,
                    status TEXT,
                    duration REAL,
                    tests_executed INTEGER,
                    tests_passed INTEGER,
                    tests_failed INTEGER,
                    peak_memory_mb REAL,
                    git_commit TEXT,
                    deployment_allowed BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_details TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    metric_type TEXT,
                    metric_value REAL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _emergency_cleanup(self, memory_mb: float, memory_percent: float):
        """Emergency memory cleanup procedure"""
        if self.emergency_cleanup_triggered:
            return  # Already triggered
        
        self.emergency_cleanup_triggered = True
        logger.error(f"🚨 EMERGENCY CLEANUP TRIGGERED - Memory: {memory_mb:.1f}MB ({memory_percent:.1%})")
        
        try:
            # Aggressive garbage collection
            for _ in range(5):
                collected = gc.collect()
                logger.info(f"🧹 Emergency GC: {collected} objects collected")
            
            # Clear any large data structures
            gc.disable()
            gc.enable()
            
            # Force memory release
            if hasattr(gc, 'set_threshold'):
                gc.set_threshold(700, 10, 10)  # More aggressive GC
            
            logger.info("🛡️ Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Emergency cleanup failed: {e}")
    
    async def execute_automated_workflow(self, mode: str = "full") -> WorkflowResult:
        """Execute automated TDD workflow"""
        start_time = time.time()
        
        logger.info(f"🚀 Starting automated TDD workflow - Mode: {mode}")
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        try:
            # Get git information
            current_commit = self.git_integration.get_current_commit()
            changed_files = self.git_integration.get_changed_files()
            
            logger.info(f"📋 Git commit: {current_commit}")
            logger.info(f"📁 Changed files: {len(changed_files)}")
            
            # Execute workflow based on mode
            if mode == "pre-commit":
                result = await self._execute_pre_commit_workflow(changed_files)
            elif mode == "pre-push":
                result = await self._execute_pre_push_workflow(changed_files)
            elif mode == "ci-cd":
                result = await self._execute_ci_cd_workflow()
            else:  # full
                result = await self._execute_full_workflow()
            
            # Stop monitoring and get metrics
            memory_metrics = self.memory_monitor.stop_monitoring()
            
            # Create workflow result
            workflow_result = WorkflowResult(
                workflow_id=self.workflow_id,
                status=result.get("status", "ERROR"),
                duration=time.time() - start_time,
                tests_executed=result.get("tests_executed", 0),
                tests_passed=result.get("tests_passed", 0),
                tests_failed=result.get("tests_failed", 0),
                peak_memory_mb=memory_metrics.get("peak_memory_mb", 0),
                git_commit=current_commit,
                deployment_allowed=result.get("deployment_allowed", False),
                error_details=result.get("error_details")
            )
            
            # Record workflow result
            await self._record_workflow_result(workflow_result)
            
            return workflow_result
            
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            
            # Stop monitoring
            memory_metrics = self.memory_monitor.stop_monitoring()
            
            return WorkflowResult(
                workflow_id=self.workflow_id,
                status="ERROR",
                duration=time.time() - start_time,
                tests_executed=0,
                tests_passed=0,
                tests_failed=0,
                peak_memory_mb=memory_metrics.get("peak_memory_mb", 0),
                git_commit=current_commit,
                deployment_allowed=False,
                error_details=str(e)
            )
    
    async def _execute_pre_commit_workflow(self, changed_files: List[str]) -> Dict[str, Any]:
        """Execute pre-commit workflow"""
        logger.info("🔍 Executing pre-commit workflow")
        
        try:
            # Fast critical tests only
            from enhanced_atomic_tdd_coverage import EnhancedAtomicTDDFramework
            framework = EnhancedAtomicTDDFramework(max_memory_mb=256.0)
            
            # Run critical tests
            results = await framework.run_enhanced_suite(priority_filter="critical")
            
            tests_passed = sum(1 for r in results.values() if r.status == "PASSED")
            tests_failed = len(results) - tests_passed
            
            status = "SUCCESS" if tests_passed == len(results) else "FAILURE"
            deployment_allowed = status == "SUCCESS"
            
            return {
                "status": status,
                "tests_executed": len(results),
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "deployment_allowed": deployment_allowed
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error_details": str(e),
                "deployment_allowed": False
            }
    
    async def _execute_pre_push_workflow(self, changed_files: List[str]) -> Dict[str, Any]:
        """Execute pre-push workflow"""
        logger.info("🚀 Executing pre-push workflow")
        
        try:
            # Comprehensive tests for push validation
            from enhanced_atomic_tdd_coverage import EnhancedAtomicTDDFramework
            framework = EnhancedAtomicTDDFramework(max_memory_mb=384.0)
            
            # Run high priority and critical tests
            critical_results = await framework.run_enhanced_suite(priority_filter="critical")
            high_results = await framework.run_enhanced_suite(priority_filter="high")
            
            all_results = {**critical_results, **high_results}
            
            tests_passed = sum(1 for r in all_results.values() if r.status == "PASSED")
            tests_failed = len(all_results) - tests_passed
            
            # Require 90% success rate for push
            success_rate = tests_passed / len(all_results) if all_results else 0
            status = "SUCCESS" if success_rate >= 0.9 else "FAILURE"
            deployment_allowed = status == "SUCCESS"
            
            return {
                "status": status,
                "tests_executed": len(all_results),
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "deployment_allowed": deployment_allowed
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error_details": str(e),
                "deployment_allowed": False
            }
    
    async def _execute_ci_cd_workflow(self) -> Dict[str, Any]:
        """Execute CI/CD workflow"""
        logger.info("🏭 Executing CI/CD workflow")
        
        try:
            # Full test suite for CI/CD
            from enhanced_atomic_tdd_coverage import EnhancedAtomicTDDFramework
            framework = EnhancedAtomicTDDFramework(max_memory_mb=512.0)
            
            # Run complete test suite
            results = await framework.run_enhanced_suite()
            
            tests_passed = sum(1 for r in results.values() if r.status == "PASSED")
            tests_failed = len(results) - tests_passed
            
            # Require 95% success rate for deployment
            success_rate = tests_passed / len(results) if results else 0
            status = "SUCCESS" if success_rate >= 0.95 else "FAILURE"
            deployment_allowed = status == "SUCCESS"
            
            return {
                "status": status,
                "tests_executed": len(results),
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "deployment_allowed": deployment_allowed
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error_details": str(e),
                "deployment_allowed": False
            }
    
    async def _execute_full_workflow(self) -> Dict[str, Any]:
        """Execute full automated workflow"""
        logger.info("🔄 Executing full automated workflow")
        
        try:
            # Complete workflow with enhanced framework
            from enhanced_atomic_tdd_coverage import EnhancedAtomicTDDFramework
            
            # Run comprehensive enhanced framework validation
            enhanced_framework = EnhancedAtomicTDDFramework(max_memory_mb=384.0)
            
            # Enhanced tests with all priorities
            enhanced_results = await enhanced_framework.run_enhanced_suite()
            
            # Get results metrics
            total_tests = len(enhanced_results)
            total_passed = sum(1 for r in enhanced_results.values() if r.status == "PASSED")
            total_failed = total_tests - total_passed
            
            # Require 90% success rate for full workflow
            success_rate = total_passed / total_tests if total_tests else 0
            status = "SUCCESS" if success_rate >= 0.9 else "FAILURE"
            deployment_allowed = status == "SUCCESS"
            
            return {
                "status": status,
                "tests_executed": total_tests,
                "tests_passed": total_passed,
                "tests_failed": total_failed,
                "deployment_allowed": deployment_allowed
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error_details": str(e),
                "deployment_allowed": False
            }
    
    async def _record_workflow_result(self, result: WorkflowResult):
        """Record workflow execution result"""
        try:
            with sqlite3.connect(self.workflow_db) as conn:
                conn.execute("""
                    INSERT INTO workflow_executions 
                    (workflow_id, status, duration, tests_executed, tests_passed, tests_failed,
                     peak_memory_mb, git_commit, deployment_allowed, completed_at, error_details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                """, (
                    result.workflow_id,
                    result.status,
                    result.duration,
                    result.tests_executed,
                    result.tests_passed,
                    result.tests_failed,
                    result.peak_memory_mb,
                    result.git_commit,
                    result.deployment_allowed,
                    result.error_details
                ))
            
            # Save detailed results
            result_file = self.results_dir / f"{result.workflow_id}_result.json"
            with open(result_file, 'w') as f:
                json.dump({
                    "workflow_id": result.workflow_id,
                    "status": result.status,
                    "duration": result.duration,
                    "tests_executed": result.tests_executed,
                    "tests_passed": result.tests_passed,
                    "tests_failed": result.tests_failed,
                    "peak_memory_mb": result.peak_memory_mb,
                    "git_commit": result.git_commit,
                    "deployment_allowed": result.deployment_allowed,
                    "error_details": result.error_details,
                    "timestamp": time.time()
                }, f, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Failed to record workflow result: {e}")
    
    def generate_workflow_report(self, result: WorkflowResult) -> str:
        """Generate comprehensive workflow report"""
        status_icon = {
            "SUCCESS": "✅",
            "FAILURE": "❌",
            "ERROR": "💥",
            "TIMEOUT": "⏰",
            "MEMORY_EXCEEDED": "🧠"
        }.get(result.status, "❓")
        
        success_rate = (result.tests_passed / result.tests_executed * 100) if result.tests_executed > 0 else 0
        memory_efficiency = ((512.0 - result.peak_memory_mb) / 512.0 * 100) if result.peak_memory_mb > 0 else 100
        
        report = [
            "🤖 AUTOMATED TDD WORKFLOW REPORT",
            "=" * 50,
            f"🆔 Workflow ID: {result.workflow_id}",
            f"{status_icon} Status: {result.status}",
            f"⏱️ Duration: {result.duration:.2f}s",
            f"📊 Tests: {result.tests_passed}/{result.tests_executed} passed ({success_rate:.1f}%)",
            f"🧠 Peak Memory: {result.peak_memory_mb:.1f}MB ({memory_efficiency:.1f}% efficient)",
            f"📝 Git Commit: {result.git_commit or 'Unknown'}",
            f"🚀 Deployment: {'✅ ALLOWED' if result.deployment_allowed else '❌ BLOCKED'}",
        ]
        
        if result.error_details:
            report.append(f"❌ Error: {result.error_details}")
        
        return "\n".join(report)

async def main():
    """Automated TDD workflow demonstration"""
    config = WorkflowConfig(
        memory_limit_mb=512.0,
        test_timeout_seconds=180.0,
        enable_git_hooks=True,
        enable_ci_cd_integration=True
    )
    
    workflow = AutomatedTDDWorkflow(config)
    
    print("🤖 Starting Automated TDD Workflow System")
    
    # Install git hooks
    if config.enable_git_hooks:
        hooks_installed = workflow.git_integration.install_git_hooks()
        print(f"🔗 Git hooks: {'✅ Installed' if hooks_installed else '❌ Failed'}")
    
    # Execute full automated workflow
    result = await workflow.execute_automated_workflow(mode="full")
    
    # Generate and display report
    report = workflow.generate_workflow_report(result)
    print(report)
    
    # Final cleanup
    gc.collect()
    print("\n🧹 Automated workflow completed")

if __name__ == "__main__":
    import sys
    
    # Handle command line arguments for git hooks
    if len(sys.argv) > 1:
        mode = sys.argv[1].replace("--mode=", "")
        config = WorkflowConfig(memory_limit_mb=256.0, test_timeout_seconds=60.0)
        workflow = AutomatedTDDWorkflow(config)
        result = asyncio.run(workflow.execute_automated_workflow(mode=mode))
        
        # Exit with appropriate code for git hooks
        exit_code = 0 if result.status == "SUCCESS" else 1
        print(f"🔗 Git hook result: {result.status}")
        sys.exit(exit_code)
    else:
        asyncio.run(main())