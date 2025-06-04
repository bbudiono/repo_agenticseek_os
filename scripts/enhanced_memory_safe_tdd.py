#!/usr/bin/env python3
"""
Enhanced Memory-Safe TDD Pipeline with Advanced Protection
=========================================================

* Purpose: TDD pipeline with advanced memory guardian and emergency recovery
* Features: Real-time leak detection, adaptive execution, emergency shutdown
* Integration: Complete protection against JavaScript heap crashes
* Safety: Multi-layered memory protection with intelligent recovery
"""

import asyncio
import time
import gc
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import sys
import os

# Import our advanced systems
from advanced_memory_guardian import AdvancedMemoryGuardian, MemoryThreshold, memory_protected_execution
from memory_safe_atomic_tdd import MemorySafeAtomicTDDFramework
from memory_safe_comprehensive_test import MemorySafeTestRunner

logger = logging.getLogger(__name__)

class EnhancedMemorySafeTDDPipeline:
    """Enhanced TDD pipeline with advanced memory protection"""
    
    def __init__(self, memory_limit_mb: float = 1024.0):
        self.memory_limit_mb = memory_limit_mb
        self.workspace_root = Path.cwd()
        
        # Initialize advanced memory guardian
        self.guardian = AdvancedMemoryGuardian(
            MemoryThreshold(
                warning_mb=memory_limit_mb * 0.4,    # 40% - warning
                critical_mb=memory_limit_mb * 0.6,   # 60% - critical  
                emergency_mb=memory_limit_mb * 0.8,  # 80% - emergency
                terminal_mb=memory_limit_mb * 0.95   # 95% - terminal
            )
        )
        
        # Register emergency cleanup actions
        self._register_emergency_actions()
        
        # Test execution tracking
        self.execution_stats = {
            "tests_executed": 0,
            "tests_blocked": 0,
            "emergency_recoveries": 0,
            "memory_violations": 0
        }
        
        # Results storage
        self.results_dir = self.workspace_root / "test_results" / "enhanced_memory_safe"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🛡️ Enhanced Memory-Safe TDD Pipeline initialized")
        logger.info(f"📊 Memory limit: {memory_limit_mb}MB with guardian protection")
    
    def _register_emergency_actions(self):
        """Register emergency cleanup actions with the guardian"""
        
        # Priority 1: Immediate cleanup
        self.guardian.emergency_recovery.register_cleanup_action(
            self._emergency_gc_cleanup,
            priority=1
        )
        
        # Priority 2: Clear Python caches
        self.guardian.emergency_recovery.register_cleanup_action(
            self._emergency_cache_cleanup,
            priority=2
        )
        
        # Priority 3: Clear module references
        self.guardian.emergency_recovery.register_cleanup_action(
            self._emergency_module_cleanup,
            priority=3
        )
        
        # Priority 4: Nuclear cleanup (last resort)
        self.guardian.emergency_recovery.register_cleanup_action(
            self._emergency_nuclear_cleanup,
            priority=4
        )
    
    async def _emergency_gc_cleanup(self) -> Dict[str, Any]:
        """Emergency garbage collection cleanup"""
        collected = 0
        for i in range(5):  # Multiple passes
            collected += gc.collect()
            await asyncio.sleep(0.1)
        
        return {"action": "gc_cleanup", "objects_collected": collected}
    
    async def _emergency_cache_cleanup(self) -> Dict[str, Any]:
        """Emergency cache cleanup"""
        caches_cleared = []
        
        try:
            import linecache
            linecache.clearcache()
            caches_cleared.append("linecache")
        except:
            pass
        
        try:
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
                caches_cleared.append("type_cache")
        except:
            pass
        
        return {"action": "cache_cleanup", "caches_cleared": caches_cleared}
    
    async def _emergency_module_cleanup(self) -> Dict[str, Any]:
        """Emergency module cleanup"""
        modules_before = len(sys.modules)
        
        # Remove test-specific modules
        modules_to_remove = []
        for module_name in list(sys.modules.keys()):
            if any(pattern in module_name for pattern in ['test_', '_test', 'sources.']):
                if module_name not in ['sources.logger', 'sources.utility']:  # Keep essential modules
                    modules_to_remove.append(module_name)
        
        for module_name in modules_to_remove:
            try:
                del sys.modules[module_name]
            except:
                pass
        
        modules_after = len(sys.modules)
        modules_removed = modules_before - modules_after
        
        return {"action": "module_cleanup", "modules_removed": modules_removed}
    
    async def _emergency_nuclear_cleanup(self) -> Dict[str, Any]:
        """Nuclear cleanup - last resort"""
        logger.critical("☢️ NUCLEAR CLEANUP INITIATED - CLEARING ALL NON-ESSENTIAL DATA")
        
        # Clear all local variables in caller frames
        import inspect
        frame = inspect.currentframe()
        cleanup_count = 0
        
        try:
            while frame:
                frame_locals = frame.f_locals
                keys_to_clear = [k for k in frame_locals.keys() 
                               if not k.startswith('__') and k not in ['self', 'logger']]
                
                for key in keys_to_clear:
                    try:
                        if key in frame_locals:
                            del frame_locals[key]
                            cleanup_count += 1
                    except:
                        pass
                
                frame = frame.f_back
        except:
            pass
        
        # Final aggressive GC
        for _ in range(10):
            gc.collect()
            await asyncio.sleep(0.05)
        
        return {"action": "nuclear_cleanup", "variables_cleared": cleanup_count}
    
    async def memory_safe_atomic_tests(self) -> Optional[Dict[str, Any]]:
        """Execute atomic tests with memory protection"""
        logger.info("🔬 Starting memory-protected atomic tests")
        
        # Check if execution is safe
        if not self.guardian.is_execution_safe(estimated_memory_mb=200):
            self.execution_stats["tests_blocked"] += 1
            logger.warning("🚫 Atomic tests blocked due to memory constraints")
            return {"status": "blocked", "reason": "memory_constraints"}
        
        try:
            with memory_protected_execution(self.guardian, estimated_memory_mb=200):
                # Initialize atomic framework with reduced memory
                atomic_framework = MemorySafeAtomicTDDFramework(
                    workspace_root=str(self.workspace_root),
                    memory_limit_mb=self.memory_limit_mb * 0.3  # Use only 30% for atomic tests
                )
                
                # Execute tests
                results = await atomic_framework.run_memory_safe_test_suite()
                self.execution_stats["tests_executed"] += len(results)
                
                # Cleanup framework
                del atomic_framework
                gc.collect()
                
                return {
                    "status": "completed",
                    "results": results,
                    "tests_executed": len(results)
                }
                
        except MemoryError as e:
            self.execution_stats["memory_violations"] += 1
            logger.error(f"💥 Memory error in atomic tests: {e}")
            return {"status": "memory_error", "error": str(e)}
        except Exception as e:
            logger.error(f"💥 Error in atomic tests: {e}")
            return {"status": "error", "error": str(e)}
    
    async def memory_safe_comprehensive_tests(self) -> Optional[Dict[str, Any]]:
        """Execute comprehensive tests with memory protection"""
        logger.info("🧪 Starting memory-protected comprehensive tests")
        
        # Check if execution is safe
        if not self.guardian.is_execution_safe(estimated_memory_mb=300):
            self.execution_stats["tests_blocked"] += 1
            logger.warning("🚫 Comprehensive tests blocked due to memory constraints")
            return {"status": "blocked", "reason": "memory_constraints"}
        
        try:
            with memory_protected_execution(self.guardian, estimated_memory_mb=300):
                # Initialize comprehensive runner with reduced memory
                comprehensive_runner = MemorySafeTestRunner(
                    memory_limit_mb=self.memory_limit_mb * 0.5  # Use 50% for comprehensive tests
                )
                
                # Execute tests
                results = await comprehensive_runner.run_comprehensive_memory_safe_suite()
                self.execution_stats["tests_executed"] += results["summary"]["total_tests"]
                
                # Cleanup runner
                del comprehensive_runner
                gc.collect()
                
                return {
                    "status": "completed",
                    "results": results,
                    "tests_executed": results["summary"]["total_tests"]
                }
                
        except MemoryError as e:
            self.execution_stats["memory_violations"] += 1
            logger.error(f"💥 Memory error in comprehensive tests: {e}")
            return {"status": "memory_error", "error": str(e)}
        except Exception as e:
            logger.error(f"💥 Error in comprehensive tests: {e}")
            return {"status": "error", "error": str(e)}
    
    async def execute_enhanced_pipeline(self) -> Dict[str, Any]:
        """Execute enhanced memory-safe TDD pipeline"""
        start_time = datetime.now()
        pipeline_id = f"enhanced_pipeline_{int(time.time())}"
        
        logger.info(f"🚀 Starting Enhanced Memory-Safe TDD Pipeline - ID: {pipeline_id}")
        
        # Start memory guardian monitoring
        self.guardian.start_monitoring(interval=0.5)
        
        try:
            # Initial memory check
            initial_report = self.guardian.get_memory_report()
            logger.info(f"📊 Initial memory: {initial_report['current_memory_mb']:.2f}MB")
            
            results = {
                "pipeline_id": pipeline_id,
                "start_time": start_time.isoformat(),
                "initial_memory_mb": initial_report['current_memory_mb'],
                "memory_limit_mb": self.memory_limit_mb,
                "atomic_results": None,
                "comprehensive_results": None,
                "execution_stats": None,
                "memory_guardian_report": None,
                "success": False
            }
            
            # Phase 1: Atomic tests with memory protection
            atomic_results = await self.memory_safe_atomic_tests()
            results["atomic_results"] = atomic_results
            
            # Memory recovery pause
            await asyncio.sleep(2)
            gc.collect()
            
            # Check if emergency recovery was triggered
            if self.guardian.emergency_recovery.emergency_active:
                self.execution_stats["emergency_recoveries"] += 1
                logger.warning("⚠️ Emergency recovery was triggered during atomic tests")
            
            # Phase 2: Comprehensive tests with memory protection
            comprehensive_results = await self.memory_safe_comprehensive_tests()
            results["comprehensive_results"] = comprehensive_results
            
            # Final memory report
            final_report = self.guardian.get_memory_report()
            results["memory_guardian_report"] = final_report
            results["execution_stats"] = self.execution_stats
            
            # Determine success
            atomic_success = atomic_results and atomic_results.get("status") == "completed"
            comprehensive_success = comprehensive_results and comprehensive_results.get("status") == "completed"
            results["success"] = atomic_success and comprehensive_success
            
            # Calculate duration
            end_time = datetime.now()
            results["end_time"] = end_time.isoformat()
            results["total_duration"] = (end_time - start_time).total_seconds()
            
            # Save results
            results_file = self.results_dir / f"{pipeline_id}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"🎯 Enhanced pipeline completed - Success: {results['success']}")
            logger.info(f"📊 Final memory: {final_report['current_memory_mb']:.2f}MB")
            logger.info(f"💾 Results saved to {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"💥 Enhanced pipeline failed: {e}")
            return {
                "pipeline_id": pipeline_id,
                "error": str(e),
                "success": False,
                "execution_stats": self.execution_stats
            }
        finally:
            # Stop memory guardian
            self.guardian.stop_monitoring()
    
    def generate_enhanced_report(self, results: Dict[str, Any]) -> str:
        """Generate enhanced pipeline report"""
        success_emoji = "✅" if results.get("success") else "❌"
        
        report = f"""🛡️ ENHANCED MEMORY-SAFE TDD PIPELINE REPORT
==================================================
{success_emoji} Pipeline ID: {results.get('pipeline_id', 'Unknown')}
📅 Execution Time: {results.get('start_time', 'Unknown')}
⏱️ Total Duration: {results.get('total_duration', 0):.2f}s
🎯 Success: {results.get('success', False)}

🧠 Memory Protection:
   💾 Memory Limit: {results.get('memory_limit_mb', 0)}MB
   📊 Initial Memory: {results.get('initial_memory_mb', 0):.2f}MB"""

        if results.get('memory_guardian_report'):
            guardian_report = results['memory_guardian_report']
            report += f"""
   📈 Final Memory: {guardian_report.get('current_memory_mb', 0):.2f}MB
   🔍 Memory Snapshots: {guardian_report.get('snapshots_count', 0)}
   🚨 Leaks Detected: {guardian_report.get('detected_leaks', 0)}
   🛡️ Execution Blocked: {guardian_report.get('execution_blocked', False)}"""

        if results.get('execution_stats'):
            stats = results['execution_stats']
            report += f"""

📊 Execution Statistics:
   🧪 Tests Executed: {stats.get('tests_executed', 0)}
   🚫 Tests Blocked: {stats.get('tests_blocked', 0)}
   🚨 Emergency Recoveries: {stats.get('emergency_recoveries', 0)}
   💥 Memory Violations: {stats.get('memory_violations', 0)}"""

        # Atomic results
        if results.get('atomic_results'):
            atomic = results['atomic_results']
            if atomic.get('status') == 'completed':
                atomic_results = atomic.get('results', {})
                passed = sum(1 for r in atomic_results.values() if r.status == "PASSED")
                total = len(atomic_results)
                report += f"""

🔬 Atomic Tests: ✅ {passed}/{total} passed"""
            else:
                report += f"""

🔬 Atomic Tests: ❌ {atomic.get('status', 'failed')}"""

        # Comprehensive results
        if results.get('comprehensive_results'):
            comp = results['comprehensive_results']
            if comp.get('status') == 'completed':
                comp_summary = comp.get('results', {}).get('summary', {})
                passed = comp_summary.get('passed', 0)
                total = comp_summary.get('total_tests', 0)
                report += f"""
🧪 Comprehensive Tests: ✅ {passed}/{total} passed"""
            else:
                report += f"""
🧪 Comprehensive Tests: ❌ {comp.get('status', 'failed')}"""

        if not results.get('success'):
            report += f"""

❌ Failure Details:
   {results.get('error', 'Multiple test failures or memory constraints')}"""

        return report

# CLI interface for enhanced pipeline
async def main():
    """Enhanced memory-safe TDD pipeline CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Memory-Safe TDD Pipeline")
    parser.add_argument("--memory-limit", type=float, default=1024.0, help="Memory limit in MB")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize and run enhanced pipeline
    pipeline = EnhancedMemorySafeTDDPipeline(memory_limit_mb=args.memory_limit)
    
    try:
        results = await pipeline.execute_enhanced_pipeline()
        
        # Generate and display report
        report = pipeline.generate_enhanced_report(results)
        print(report)
        
        # Exit with appropriate code
        exit_code = 0 if results.get("success") else 1
        logger.info(f"🏁 Enhanced pipeline finished with exit code: {exit_code}")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"💥 Enhanced pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)