#!/usr/bin/env python3
"""
Test MLACS Performance Optimization Integration
Validates performance improvements against original benchmarks
"""

import asyncio
import time
import json
from typing import Dict, List, Any

from mlacs_performance_optimizer import MLACSPerformanceOptimizer, OptimizationStrategy

class MLACSPerformanceTest:
    """Test suite for MLACS performance optimization"""
    
    def __init__(self):
        self.optimizer = MLACSPerformanceOptimizer(enable_caching=True)
        self.baseline_metrics = {
            'coordination_overhead': 133.1,  # From benchmark
            'avg_response_time': 20.75,      # From benchmark
            'quality_score': 1.10            # From benchmark
        }
    
    async def run_performance_comparison(self):
        """Run performance comparison against baseline"""
        print("🚀 MLACS Performance Optimization Test")
        print("="*60)
        
        # Test scenarios based on original benchmark
        test_scenarios = [
            {
                'name': 'Complex Research Query',
                'query': 'Analyze the long-term implications of artificial intelligence on global economic structures, considering technological displacement, new job creation, and regulatory frameworks',
                'providers': ['anthropic', 'openai', 'google'],
                'expected_complexity': 'high'
            },
            {
                'name': 'Multi-Domain Analysis',
                'query': 'Compare and contrast renewable energy adoption strategies across developed and developing nations',
                'providers': ['anthropic', 'openai'],
                'expected_complexity': 'medium'
            },
            {
                'name': 'Simple Factual Query',
                'query': 'What are the main differences between Python and JavaScript?',
                'providers': ['openai', 'anthropic'],
                'expected_complexity': 'low'
            }
        ]
        
        total_optimized_time = 0
        total_baseline_time = 0
        optimization_results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📋 Test {i}: {scenario['name']}")
            print(f"   Query: {scenario['query'][:80]}...")
            
            # Run optimized version
            start_time = time.time()
            result = await self.optimizer.optimize_multi_llm_query(
                scenario['query'], 
                scenario['providers']
            )
            optimized_time = time.time() - start_time
            
            # Calculate baseline simulation (sequential execution)
            baseline_time = await self._simulate_baseline_execution(scenario)
            
            total_optimized_time += optimized_time
            total_baseline_time += baseline_time
            
            # Performance metrics
            metrics = result['performance_metrics']
            improvement = ((baseline_time - optimized_time) / baseline_time) * 100
            
            optimization_results.append({
                'scenario': scenario['name'],
                'baseline_time': baseline_time,
                'optimized_time': optimized_time,
                'improvement_percent': improvement,
                'coordination_overhead': metrics['coordination_overhead_percent'],
                'parallel_execution': metrics['parallel_execution'],
                'synthesis_skipped': metrics['synthesis_skipped'],
                'cache_hit': metrics['cache_hit']
            })
            
            print(f"   ⏱️ Baseline time: {baseline_time:.2f}s")
            print(f"   ⚡ Optimized time: {optimized_time:.2f}s")
            print(f"   📈 Improvement: {improvement:.1f}%")
            print(f"   🔄 Parallel execution: {metrics['parallel_execution']}")
            print(f"   📊 Coordination overhead: {metrics['coordination_overhead_percent']:.1f}%")
        
        # Overall performance summary
        total_improvement = ((total_baseline_time - total_optimized_time) / total_baseline_time) * 100
        avg_coordination_overhead = sum(r['coordination_overhead'] for r in optimization_results) / len(optimization_results)
        
        print(f"\n🎯 Overall Performance Summary")
        print(f"   📊 Total baseline time: {total_baseline_time:.2f}s")
        print(f"   ⚡ Total optimized time: {total_optimized_time:.2f}s")
        print(f"   📈 Overall improvement: {total_improvement:.1f}%")
        print(f"   📉 Avg coordination overhead: {avg_coordination_overhead:.1f}%")
        print(f"   🎯 Coordination overhead reduction: {133.1 - avg_coordination_overhead:.1f} percentage points")
        
        # Generate performance report
        performance_report = self.optimizer.get_performance_report()
        
        print(f"\n📊 Optimization Report")
        print(f"   💾 Cache hit rate: {performance_report['cache_hit_rate']:.1%}")
        print(f"   ⚡ Parallel execution rate: {performance_report['parallel_execution_rate']:.1%}")
        print(f"   🎯 Synthesis skip rate: {performance_report['synthesis_skip_rate']:.1%}")
        print(f"   🚀 Estimated time savings: {performance_report['performance_improvements']['estimated_time_savings']:.1%}")
        
        # Test caching effectiveness
        await self._test_caching_performance()
        
        return {
            'total_improvement_percent': total_improvement,
            'coordination_overhead_reduction': 133.1 - avg_coordination_overhead,
            'scenarios': optimization_results,
            'performance_report': performance_report
        }
    
    async def _simulate_baseline_execution(self, scenario: Dict[str, Any]) -> float:
        """Simulate baseline sequential execution time"""
        providers = scenario['providers']
        
        # Simulate sequential execution (original MLACS behavior)
        total_time = 0
        
        for provider in providers:
            # Provider-specific baseline times (from benchmark data)
            provider_times = {
                'anthropic': 9.54,  # Claude baseline
                'openai': 8.26,     # GPT baseline
                'google': 8.5,      # Estimated Gemini
                'deepseek': 7.8     # Estimated DeepSeek
            }
            
            provider_time = provider_times.get(provider, 8.5)
            total_time += provider_time
        
        # Add synthesis time (from benchmark: additional 20-30%)
        synthesis_time = total_time * 0.25
        total_time += synthesis_time
        
        return total_time
    
    async def _test_caching_performance(self):
        """Test caching performance benefits"""
        print(f"\n💾 Testing Cache Performance")
        
        # Test query
        test_query = "What are the benefits of renewable energy?"
        providers = ['openai', 'anthropic']
        
        # First execution (no cache)
        start_time = time.time()
        await self.optimizer.optimize_multi_llm_query(test_query, providers)
        first_execution_time = time.time() - start_time
        
        # Second execution (with cache)
        start_time = time.time()
        cached_result = await self.optimizer.optimize_multi_llm_query(test_query, providers)
        cached_execution_time = time.time() - start_time
        
        cache_improvement = ((first_execution_time - cached_execution_time) / first_execution_time) * 100
        
        print(f"   🔄 First execution: {first_execution_time:.3f}s")
        print(f"   💾 Cached execution: {cached_execution_time:.3f}s")
        print(f"   📈 Cache improvement: {cache_improvement:.1f}%")
        print(f"   ✅ Cache hit: {cached_result.get('cached', False)}")

async def main():
    """Run the performance optimization test"""
    test_suite = MLACSPerformanceTest()
    results = await test_suite.run_performance_comparison()
    
    print(f"\n🎉 Performance Optimization Test Complete!")
    print(f"🚀 Achieved {results['total_improvement_percent']:.1f}% overall performance improvement")
    print(f"📉 Reduced coordination overhead by {results['coordination_overhead_reduction']:.1f} percentage points")

if __name__ == "__main__":
    asyncio.run(main())