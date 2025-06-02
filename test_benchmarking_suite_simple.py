#!/usr/bin/env python3
"""
SANDBOX FILE: For testing/development. See .cursorrules.

Simple test for MLACS Performance Benchmarking Suite
====================================================
"""

import sys
import time
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append('/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek')

try:
    from mlacs_comprehensive_performance_benchmarking_suite import (
        MLACSPerformanceBenchmarkingSuite,
        MLACSBenchmarkingSuiteFactory
    )
    
    def simple_benchmarking_demo():
        """Simple benchmarking demonstration"""
        
        print("🚀 MLACS Performance Benchmarking Suite - Simple Demo")
        print("=" * 60)
        
        # Create benchmarking suite with basic config
        config = {
            'results_directory': 'simple_benchmark_results',
            'benchmark_iterations': 3,  # Reduced for demo
            'warmup_iterations': 1,
            'enable_memory_monitoring': True,
            'enable_cpu_monitoring': True,
            'enable_regression_testing': True
        }
        
        try:
            suite = MLACSBenchmarkingSuiteFactory.create_benchmarking_suite(config)
            print("✅ Benchmarking suite created successfully")
            
            # Get system status
            status = suite.get_system_status()
            print(f"✅ Framework status: {status['benchmarking_status']}")
            print(f"✅ Available frameworks: {sum(1 for v in status['framework_availability'].values() if v)}")
            print(f"✅ Registered benchmark suites: {status['benchmark_management']['registered_suites']}")
            
            # Test optimization engine benchmarks if available
            if 'optimization_engine_performance' in suite.benchmark_suites:
                print("\n📊 Running optimization engine performance benchmarks...")
                
                # Execute without decorator complications
                suite_obj = suite.benchmark_suites['optimization_engine_performance']
                print(f"   Suite: {suite_obj.name}")
                print(f"   Scenarios: {len(suite_obj.test_scenarios)}")
                print(f"   Framework: {suite_obj.framework}")
                
                # Run a simple benchmark test
                if 'optimization_engine' in suite.framework_instances:
                    instance = suite.framework_instances['optimization_engine']
                    print(f"   Instance available: ✅")
                    
                    # Test system status response time
                    start_time = time.time()
                    status = instance.get_system_status()
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    print(f"   System status response: {response_time:.4f}s")
                    print(f"   Status keys: {list(status.keys())}")
                else:
                    print("   Instance not available: ❌")
            
            # Test cross-framework benchmarks
            if 'cross_framework_performance' in suite.benchmark_suites:
                print("\n🔄 Cross-framework compatibility check...")
                cross_suite = suite.benchmark_suites['cross_framework_performance']
                print(f"   Cross-framework scenarios: {len(cross_suite.test_scenarios)}")
                
                # Test framework initialization time
                start_time = time.time()
                available_count = sum(1 for available in status['framework_availability'].values() if available)
                end_time = time.time()
                init_time = end_time - start_time
                
                print(f"   Framework check time: {init_time:.4f}s")
                print(f"   Available frameworks: {available_count}")
            
            # Generate optimization recommendations
            print("\n💡 Generating optimization recommendations...")
            recommendations = suite.generate_optimization_recommendations()
            print(f"   Generated recommendations: {len(recommendations)}")
            
            if recommendations:
                print("   Sample recommendations:")
                for i, rec in enumerate(recommendations[:3]):
                    print(f"     {i+1}. {rec}")
            
            # Final status check
            print("\n📈 Final performance summary...")
            final_status = suite.get_system_status()
            perf_monitoring = final_status['performance_monitoring']
            
            print(f"   Memory monitoring: {'✅' if perf_monitoring['memory_monitoring_enabled'] else '❌'}")
            print(f"   CPU monitoring: {'✅' if perf_monitoring['cpu_monitoring_enabled'] else '❌'}")
            print(f"   Regression testing: {'✅' if perf_monitoring['regression_testing_enabled'] else '❌'}")
            
            print("\n🎉 Simple benchmarking demo completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error during benchmarking demo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    if __name__ == "__main__":
        success = simple_benchmarking_demo()
        print(f"\nDemo result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are available")