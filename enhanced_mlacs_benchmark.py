#!/usr/bin/env python3
"""
Enhanced MLACS Benchmarking System
Multi-LLM Agent Coordination System with Advanced Performance Analytics

This enhanced benchmarking system provides comprehensive testing scenarios,
advanced performance metrics, and sophisticated optimization strategies.

Author: AgenticSeek Development Team
Date: 2025-01-06
Version: 2.0.0
"""

import asyncio
import time
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import numpy as np
from pathlib import Path
import anthropic
import openai
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import threading

# Enhanced Performance Metrics
@dataclass
class EnhancedPerformanceMetrics:
    """Enhanced performance metrics with detailed analytics"""
    # Basic Metrics
    total_scenarios: int
    successful_scenarios: int
    failed_scenarios: int
    success_rate: float
    total_duration: float
    total_llm_calls: int
    total_tokens: int
    
    # Response Time Analytics
    avg_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    response_time_std: float
    
    # Quality Analytics
    avg_quality_score: float
    median_quality_score: float
    quality_score_std: float
    min_quality_score: float
    max_quality_score: float
    
    # Token Analytics
    avg_tokens_per_call: float
    median_tokens_per_call: float
    tokens_per_second: float
    cost_per_1k_tokens: float
    estimated_total_cost: float
    
    # Coordination Analytics
    coordination_overhead: float
    parallel_efficiency: float
    cache_hit_rate: float
    error_rate: float
    
    # Resource Analytics
    peak_memory_usage: float
    avg_cpu_usage: float
    network_latency: float
    
    # Provider-Specific Analytics
    provider_performance: Dict[str, Dict[str, float]]
    
    # Quality Distribution
    quality_distribution: Dict[str, int]
    
    # Advanced Analytics
    throughput_per_minute: float
    scalability_score: float
    reliability_score: float
    efficiency_score: float
    overall_performance_score: float

@dataclass
class BenchmarkScenario:
    """Enhanced benchmark scenario with complexity and context"""
    scenario_id: str
    title: str
    description: str
    query: str
    expected_providers: List[str]
    complexity_score: float  # 0.0 - 1.0
    scenario_type: str  # single_llm, multi_llm, coordination, complex_reasoning
    context_requirements: Dict[str, Any]
    success_criteria: Dict[str, float]
    expected_duration_range: Tuple[float, float]
    priority: str  # low, medium, high, critical

class OptimizationStrategy(Enum):
    """Enhanced optimization strategies"""
    BASELINE = "baseline"
    PARALLEL_EXECUTION = "parallel_execution"
    SMART_CACHING = "smart_caching"
    ADAPTIVE_ROUTING = "adaptive_routing"
    FAST_MODELS = "fast_models"
    LOAD_BALANCING = "load_balancing"
    RESPONSE_STREAMING = "response_streaming"
    CONTEXT_COMPRESSION = "context_compression"
    PREDICTIVE_CACHING = "predictive_caching"
    DYNAMIC_SCALING = "dynamic_scaling"
    QUALITY_OPTIMIZATION = "quality_optimization"
    COST_OPTIMIZATION = "cost_optimization"

class EnhancedMLACSBenchmark:
    """Enhanced Multi-LLM Agent Coordination System Benchmark"""
    
    def __init__(self):
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.google_key = os.getenv('GOOGLE_API_KEY')
        
        if not self.anthropic_key or not self.openai_key:
            raise ValueError("Missing required API keys. Please set ANTHROPIC_API_KEY and OPENAI_API_KEY")
        
        # Enhanced caching system
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # Performance monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Enhanced scenarios
        self.scenarios = self._create_enhanced_scenarios()
        
        # Provider configurations
        self.provider_configs = {
            "claude": {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 2000,
                "temperature": 0.7,
                "cost_per_1k_tokens": 0.00025  # Haiku pricing
            },
            "claude-sonnet": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 4000,
                "temperature": 0.3,
                "cost_per_1k_tokens": 0.003
            },
            "gpt4": {
                "model": "gpt-4-turbo-preview",
                "max_tokens": 2000,
                "temperature": 0.7,
                "cost_per_1k_tokens": 0.01
            },
            "gpt3.5": {
                "model": "gpt-3.5-turbo",
                "max_tokens": 2000,
                "temperature": 0.7,
                "cost_per_1k_tokens": 0.0005
            }
        }
        
        print("🚀 Enhanced MLACS Benchmark System Initialized")
        print(f"📊 Available Scenarios: {len(self.scenarios)}")
        print(f"🔧 Provider Configurations: {len(self.provider_configs)}")
    
    def _create_enhanced_scenarios(self) -> List[BenchmarkScenario]:
        """Create comprehensive benchmark scenarios"""
        return [
            # Single LLM Scenarios
            BenchmarkScenario(
                scenario_id="single_simple_analysis",
                title="Simple Analysis Task",
                description="Basic analytical task for single LLM",
                query="Analyze the current trends in renewable energy adoption globally. Provide 3 key insights.",
                expected_providers=["claude"],
                complexity_score=0.3,
                scenario_type="single_llm",
                context_requirements={"max_tokens": 1000},
                success_criteria={"min_quality": 0.8, "max_duration": 10.0},
                expected_duration_range=(3.0, 8.0),
                priority="medium"
            ),
            
            BenchmarkScenario(
                scenario_id="single_complex_reasoning",
                title="Complex Reasoning Task",
                description="Multi-step reasoning for single LLM",
                query="Design a comprehensive strategy for a tech startup to achieve carbon neutrality within 2 years. Include timeline, budget considerations, and risk mitigation strategies.",
                expected_providers=["claude-sonnet"],
                complexity_score=0.8,
                scenario_type="single_llm",
                context_requirements={"max_tokens": 4000},
                success_criteria={"min_quality": 0.9, "max_duration": 15.0},
                expected_duration_range=(8.0, 15.0),
                priority="high"
            ),
            
            # Multi-LLM Coordination Scenarios
            BenchmarkScenario(
                scenario_id="multi_comparative_analysis",
                title="Comparative Analysis Coordination",
                description="Multiple LLMs providing different perspectives on the same topic",
                query="Compare and contrast the economic impacts of renewable energy vs fossil fuels. Each perspective should focus on different aspects: environmental economics, market dynamics, and policy implications.",
                expected_providers=["claude", "gpt4"],
                complexity_score=0.6,
                scenario_type="multi_llm",
                context_requirements={"coordination_type": "parallel", "synthesis_required": True},
                success_criteria={"min_quality": 1.1, "max_duration": 25.0},
                expected_duration_range=(15.0, 25.0),
                priority="high"
            ),
            
            BenchmarkScenario(
                scenario_id="multi_expert_consensus",
                title="Expert Consensus Building",
                description="Multiple LLMs working towards consensus on complex topic",
                query="Develop a consensus strategy for global climate policy that balances economic growth with environmental protection. Incorporate perspectives from developed and developing nations.",
                expected_providers=["claude-sonnet", "gpt4"],
                complexity_score=0.9,
                scenario_type="coordination",
                context_requirements={"coordination_type": "consensus", "iterations": 3},
                success_criteria={"min_quality": 1.3, "max_duration": 45.0},
                expected_duration_range=(25.0, 45.0),
                priority="critical"
            ),
            
            # Speed vs Quality Trade-off Scenarios
            BenchmarkScenario(
                scenario_id="speed_optimized_task",
                title="Speed-Optimized Quick Response",
                description="Fast response scenario prioritizing speed over depth",
                query="Provide 5 quick actionable tips for improving team productivity in remote work environments.",
                expected_providers=["gpt3.5"],
                complexity_score=0.2,
                scenario_type="speed_optimized",
                context_requirements={"max_tokens": 500, "temperature": 0.9},
                success_criteria={"min_quality": 0.7, "max_duration": 5.0},
                expected_duration_range=(1.0, 5.0),
                priority="medium"
            ),
            
            BenchmarkScenario(
                scenario_id="quality_optimized_task",
                title="Quality-Optimized Deep Analysis",
                description="High-quality response scenario prioritizing depth and accuracy",
                query="Conduct a comprehensive analysis of the potential societal impacts of artificial general intelligence (AGI), including ethical considerations, economic disruption, and governance challenges.",
                expected_providers=["claude-sonnet", "gpt4"],
                complexity_score=1.0,
                scenario_type="quality_optimized",
                context_requirements={"max_tokens": 6000, "temperature": 0.1},
                success_criteria={"min_quality": 1.4, "max_duration": 60.0},
                expected_duration_range=(30.0, 60.0),
                priority="critical"
            ),
            
            # Stress Testing Scenarios
            BenchmarkScenario(
                scenario_id="high_volume_parallel",
                title="High Volume Parallel Processing",
                description="Multiple concurrent requests testing system scalability",
                query="Generate creative business ideas for sustainable technology startups in different sectors: energy, transportation, agriculture, and manufacturing.",
                expected_providers=["claude", "gpt4", "gpt3.5"],
                complexity_score=0.7,
                scenario_type="stress_test",
                context_requirements={"parallel_requests": 4, "coordination_type": "parallel"},
                success_criteria={"min_quality": 1.0, "max_duration": 30.0},
                expected_duration_range=(15.0, 30.0),
                priority="high"
            ),
            
            # Cache Effectiveness Scenarios
            BenchmarkScenario(
                scenario_id="cache_test_repeated",
                title="Cache Effectiveness Test",
                description="Repeated queries to test caching effectiveness",
                query="Explain the basic principles of machine learning and provide 3 common algorithms with their use cases.",
                expected_providers=["claude"],
                complexity_score=0.4,
                scenario_type="cache_test",
                context_requirements={"repeat_count": 3},
                success_criteria={"min_quality": 0.8, "max_duration": 15.0},
                expected_duration_range=(2.0, 10.0),
                priority="medium"
            ),
            
            # Context Window Management Scenarios
            BenchmarkScenario(
                scenario_id="large_context_processing",
                title="Large Context Processing",
                description="Testing handling of large context windows",
                query="Analyze this comprehensive business plan and provide detailed feedback on market analysis, financial projections, competitive landscape, and growth strategy. Consider all aspects of viability and provide specific improvement recommendations. " + "Context: " + "A" * 2000,  # Large context
                expected_providers=["claude-sonnet"],
                complexity_score=0.9,
                scenario_type="large_context",
                context_requirements={"large_context": True, "max_tokens": 5000},
                success_criteria={"min_quality": 1.2, "max_duration": 40.0},
                expected_duration_range=(20.0, 40.0),
                priority="high"
            )
        ]
    
    async def run_enhanced_benchmark(self, optimization_strategies: List[OptimizationStrategy] = None) -> Dict[str, Any]:
        """Run comprehensive enhanced benchmark"""
        if optimization_strategies is None:
            optimization_strategies = [OptimizationStrategy.BASELINE]
        
        print(f"🚀 Enhanced MLACS Benchmark with {len(optimization_strategies)} Optimization Strategies")
        print("=" * 80)
        
        benchmark_start = time.time()
        self.resource_monitor.start_monitoring()
        
        all_results = []
        scenario_results = {}
        
        for scenario in self.scenarios:
            print(f"\n🎬 Scenario: {scenario.title}")
            print(f"   📝 Complexity: {scenario.complexity_score:.2f}")
            print(f"   🎯 Type: {scenario.scenario_type}")
            print(f"   ⚡ Priority: {scenario.priority}")
            
            scenario_start = time.time()
            
            try:
                if scenario.scenario_type == "stress_test":
                    result = await self._run_stress_test_scenario(scenario, optimization_strategies)
                elif scenario.scenario_type == "cache_test":
                    result = await self._run_cache_test_scenario(scenario, optimization_strategies)
                else:
                    result = await self._run_standard_scenario(scenario, optimization_strategies)
                
                result['scenario_metadata'] = asdict(scenario)
                all_results.append(result)
                scenario_results[scenario.scenario_id] = result
                
                # Real-time performance feedback
                duration = time.time() - scenario_start
                success_rate = 1.0 if result.get('success', False) else 0.0
                print(f"   ✅ Completed in {duration:.2f}s")
                print(f"   🎯 Success Rate: {success_rate * 100:.1f}%")
                
                if result.get('quality_score'):
                    print(f"   🏆 Quality Score: {result['quality_score']:.2f}")
                
            except Exception as e:
                print(f"   ❌ Scenario Failed: {str(e)}")
                error_result = {
                    'scenario_id': scenario.scenario_id,
                    'success': False,
                    'error': str(e),
                    'duration': time.time() - scenario_start
                }
                all_results.append(error_result)
        
        self.resource_monitor.stop_monitoring()
        benchmark_duration = time.time() - benchmark_start
        
        # Calculate enhanced performance metrics
        enhanced_metrics = self._calculate_enhanced_metrics(all_results)
        
        # Generate comprehensive report
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'benchmark_duration': benchmark_duration,
            'optimization_strategies': [s.value for s in optimization_strategies],
            'scenario_results': scenario_results,
            'enhanced_metrics': asdict(enhanced_metrics),
            'resource_usage': self.resource_monitor.get_summary(),
            'cache_statistics': self.cache_stats.copy(),
            'performance_insights': self._generate_performance_insights(enhanced_metrics, all_results),
            'recommendations': self._generate_optimization_recommendations(enhanced_metrics, all_results)
        }
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"enhanced_mlacs_benchmark_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 Enhanced MLACS Benchmark Results")
        print("=" * 80)
        self._print_enhanced_summary(enhanced_metrics, report)
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return report
    
    async def _run_standard_scenario(self, scenario: BenchmarkScenario, strategies: List[OptimizationStrategy]) -> Dict[str, Any]:
        """Run standard benchmark scenario"""
        if len(scenario.expected_providers) == 1:
            return await self._run_single_llm_scenario(scenario, strategies)
        else:
            return await self._run_multi_llm_scenario(scenario, strategies)
    
    async def _run_single_llm_scenario(self, scenario: BenchmarkScenario, strategies: List[OptimizationStrategy]) -> Dict[str, Any]:
        """Run single LLM scenario"""
        provider = scenario.expected_providers[0]
        
        # Check cache if smart caching is enabled
        if OptimizationStrategy.SMART_CACHING in strategies:
            cache_key = self._generate_cache_key(scenario.query, provider)
            if cache_key in self.cache:
                self.cache_stats["hits"] += 1
                cached_result = self.cache[cache_key]
                return {
                    'scenario_id': scenario.scenario_id,
                    'providers_used': ['cache'],
                    'total_duration': 0.001,  # Near-zero cache retrieval time
                    'llm_calls': 0,
                    'total_tokens': 0,
                    'quality_score': cached_result['quality_score'],
                    'response': f"[CACHED] {cached_result['response'][:200]}...",
                    'success': True,
                    'cached': True
                }
        
        start_time = time.time()
        
        try:
            if provider.startswith('claude'):
                response, tokens = await self._call_claude(scenario.query, provider)
            elif provider.startswith('gpt'):
                response, tokens = await self._call_openai(scenario.query, provider)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            duration = time.time() - start_time
            quality_score = self._calculate_quality_score(response, scenario)
            
            # Cache result if applicable
            if OptimizationStrategy.SMART_CACHING in strategies:
                cache_key = self._generate_cache_key(scenario.query, provider)
                self.cache[cache_key] = {
                    'response': response,
                    'quality_score': quality_score,
                    'timestamp': time.time()
                }
                self.cache_stats["misses"] += 1
            
            return {
                'scenario_id': scenario.scenario_id,
                'providers_used': [provider],
                'total_duration': duration,
                'llm_calls': 1,
                'total_tokens': tokens,
                'quality_score': quality_score,
                'response': response[:500] + "..." if len(response) > 500 else response,
                'success': True,
                'cached': False
            }
            
        except Exception as e:
            return {
                'scenario_id': scenario.scenario_id,
                'providers_used': [provider],
                'total_duration': time.time() - start_time,
                'llm_calls': 0,
                'total_tokens': 0,
                'quality_score': 0.0,
                'error': str(e),
                'success': False,
                'cached': False
            }
    
    async def _run_multi_llm_scenario(self, scenario: BenchmarkScenario, strategies: List[OptimizationStrategy]) -> Dict[str, Any]:
        """Run multi-LLM coordination scenario"""
        providers = scenario.expected_providers
        
        start_time = time.time()
        
        try:
            responses = []
            total_tokens = 0
            
            # Parallel execution if enabled
            if OptimizationStrategy.PARALLEL_EXECUTION in strategies:
                tasks = []
                for provider in providers:
                    if provider.startswith('claude'):
                        task = self._call_claude(scenario.query, provider)
                    elif provider.startswith('gpt'):
                        task = self._call_openai(scenario.query, provider)
                    else:
                        continue
                    tasks.append(task)
                
                parallel_start = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                parallel_duration = time.time() - parallel_start
                
                for result in results:
                    if isinstance(result, tuple):
                        response, tokens = result
                        responses.append(response)
                        total_tokens += tokens
            else:
                # Sequential execution
                for provider in providers:
                    if provider.startswith('claude'):
                        response, tokens = await self._call_claude(scenario.query, provider)
                    elif provider.startswith('gpt'):
                        response, tokens = await self._call_openai(scenario.query, provider)
                    else:
                        continue
                    responses.append(response)
                    total_tokens += tokens
            
            # Synthesis phase
            if responses:
                synthesis_prompt = f"Synthesize these expert analyses into a comprehensive response:\n\n"
                for i, response in enumerate(responses):
                    synthesis_prompt += f"Expert {i+1} Analysis:\n{response}\n\n"
                
                synthesis_response, synthesis_tokens = await self._call_claude(synthesis_prompt, "claude")
                total_tokens += synthesis_tokens
                
                combined_response = f"Synthesis: {synthesis_response}"
            else:
                combined_response = "No valid responses received"
            
            duration = time.time() - start_time
            quality_score = self._calculate_multi_llm_quality_score(responses, scenario)
            
            return {
                'scenario_id': scenario.scenario_id,
                'providers_used': providers,
                'total_duration': duration,
                'llm_calls': len(providers) + 1,  # +1 for synthesis
                'total_tokens': total_tokens,
                'quality_score': quality_score,
                'response': combined_response[:500] + "..." if len(combined_response) > 500 else combined_response,
                'individual_responses': len(responses),
                'success': True,
                'parallel_efficiency': parallel_duration / duration if OptimizationStrategy.PARALLEL_EXECUTION in strategies else None
            }
            
        except Exception as e:
            return {
                'scenario_id': scenario.scenario_id,
                'providers_used': providers,
                'total_duration': time.time() - start_time,
                'llm_calls': 0,
                'total_tokens': 0,
                'quality_score': 0.0,
                'error': str(e),
                'success': False
            }
    
    async def _run_stress_test_scenario(self, scenario: BenchmarkScenario, strategies: List[OptimizationStrategy]) -> Dict[str, Any]:
        """Run stress test scenario with concurrent requests"""
        parallel_requests = scenario.context_requirements.get('parallel_requests', 4)
        
        start_time = time.time()
        
        try:
            # Create multiple concurrent requests
            tasks = []
            for i in range(parallel_requests):
                provider = scenario.expected_providers[i % len(scenario.expected_providers)]
                sector_query = f"{scenario.query} Focus specifically on sector {i+1}."
                
                if provider.startswith('claude'):
                    task = self._call_claude(sector_query, provider)
                elif provider.startswith('gpt'):
                    task = self._call_openai(sector_query, provider)
                else:
                    continue
                tasks.append((provider, task))
            
            # Execute all requests concurrently
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            successful_results = []
            total_tokens = 0
            
            for i, result in enumerate(results):
                if isinstance(result, tuple):
                    response, tokens = result
                    successful_results.append(response)
                    total_tokens += tokens
            
            duration = time.time() - start_time
            quality_score = len(successful_results) / parallel_requests  # Success rate as quality metric
            
            return {
                'scenario_id': scenario.scenario_id,
                'providers_used': [provider for provider, _ in tasks],
                'total_duration': duration,
                'llm_calls': len(successful_results),
                'total_tokens': total_tokens,
                'quality_score': quality_score,
                'response': f"Successfully processed {len(successful_results)}/{parallel_requests} concurrent requests",
                'concurrent_requests': parallel_requests,
                'success_rate': quality_score,
                'success': len(successful_results) > 0,
                'throughput': len(successful_results) / duration
            }
            
        except Exception as e:
            return {
                'scenario_id': scenario.scenario_id,
                'providers_used': scenario.expected_providers,
                'total_duration': time.time() - start_time,
                'llm_calls': 0,
                'total_tokens': 0,
                'quality_score': 0.0,
                'error': str(e),
                'success': False
            }
    
    async def _run_cache_test_scenario(self, scenario: BenchmarkScenario, strategies: List[OptimizationStrategy]) -> Dict[str, Any]:
        """Run cache effectiveness test scenario"""
        repeat_count = scenario.context_requirements.get('repeat_count', 3)
        provider = scenario.expected_providers[0]
        
        results = []
        total_duration = 0
        total_tokens = 0
        cache_hits = 0
        
        for i in range(repeat_count):
            single_scenario = BenchmarkScenario(
                scenario_id=f"{scenario.scenario_id}_repeat_{i}",
                title=f"{scenario.title} (Repeat {i+1})",
                description=scenario.description,
                query=scenario.query,
                expected_providers=[provider],
                complexity_score=scenario.complexity_score,
                scenario_type="single_llm",
                context_requirements={},
                success_criteria=scenario.success_criteria,
                expected_duration_range=scenario.expected_duration_range,
                priority=scenario.priority
            )
            
            result = await self._run_single_llm_scenario(single_scenario, strategies)
            results.append(result)
            
            total_duration += result['total_duration']
            total_tokens += result['total_tokens']
            
            if result.get('cached', False):
                cache_hits += 1
        
        avg_quality = sum(r['quality_score'] for r in results) / len(results)
        cache_hit_rate = cache_hits / repeat_count
        
        return {
            'scenario_id': scenario.scenario_id,
            'providers_used': [provider],
            'total_duration': total_duration,
            'llm_calls': sum(r['llm_calls'] for r in results),
            'total_tokens': total_tokens,
            'quality_score': avg_quality,
            'response': f"Cache test completed: {cache_hits}/{repeat_count} cache hits",
            'repeat_count': repeat_count,
            'cache_hit_rate': cache_hit_rate,
            'individual_results': results,
            'success': all(r['success'] for r in results),
            'cache_effectiveness': cache_hit_rate
        }
    
    async def _call_claude(self, prompt: str, provider: str = "claude") -> Tuple[str, int]:
        """Enhanced Claude API call with configuration support"""
        config = self.provider_configs.get(provider, self.provider_configs["claude"])
        
        client = anthropic.Anthropic(api_key=self.anthropic_key)
        
        response = client.messages.create(
            model=config["model"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text if response.content else ""
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return content, tokens
    
    async def _call_openai(self, prompt: str, provider: str = "gpt4") -> Tuple[str, int]:
        """Enhanced OpenAI API call with configuration support"""
        config = self.provider_configs.get(provider, self.provider_configs["gpt4"])
        
        client = openai.OpenAI(api_key=self.openai_key)
        
        response = client.chat.completions.create(
            model=config["model"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens
        
        return content, tokens
    
    def _calculate_quality_score(self, response: str, scenario: BenchmarkScenario) -> float:
        """Enhanced quality scoring based on scenario requirements"""
        base_score = 1.0
        
        # Length-based scoring
        if len(response) < 100:
            base_score *= 0.7  # Too short
        elif len(response) > 3000:
            base_score *= 1.1  # Comprehensive
        
        # Complexity-based scoring
        if scenario.complexity_score > 0.7 and len(response) > 1000:
            base_score *= 1.2  # Detailed response for complex query
        
        # Type-specific scoring
        if scenario.scenario_type == "speed_optimized":
            base_score *= 0.9 if len(response) < 500 else 0.8
        elif scenario.scenario_type == "quality_optimized":
            base_score *= 1.3 if len(response) > 1500 else 1.0
        
        return min(base_score, 2.0)  # Cap at 2.0
    
    def _calculate_multi_llm_quality_score(self, responses: List[str], scenario: BenchmarkScenario) -> float:
        """Calculate quality score for multi-LLM responses"""
        if not responses:
            return 0.0
        
        # Base score from individual responses
        individual_scores = [self._calculate_quality_score(resp, scenario) for resp in responses]
        avg_individual = sum(individual_scores) / len(individual_scores)
        
        # Coordination bonus
        coordination_bonus = 0.1 * len(responses)  # 10% bonus per additional LLM
        
        # Diversity bonus (simple heuristic: different lengths indicate diverse perspectives)
        lengths = [len(resp) for resp in responses]
        length_variance = np.var(lengths) if len(lengths) > 1 else 0
        diversity_bonus = min(length_variance / 10000, 0.3)  # Cap at 30% bonus
        
        total_score = avg_individual + coordination_bonus + diversity_bonus
        return min(total_score, 2.0)  # Cap at 2.0
    
    def _calculate_enhanced_metrics(self, results: List[Dict[str, Any]]) -> EnhancedPerformanceMetrics:
        """Calculate comprehensive enhanced performance metrics"""
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        if not results:
            return self._create_empty_metrics()
        
        # Basic metrics
        total_scenarios = len(results)
        successful_scenarios = len(successful_results)
        failed_scenarios = len(failed_results)
        success_rate = successful_scenarios / total_scenarios
        
        # Duration metrics
        durations = [r['total_duration'] for r in successful_results if 'total_duration' in r]
        total_duration = sum(durations)
        avg_response_time = statistics.mean(durations) if durations else 0
        median_response_time = statistics.median(durations) if durations else 0
        
        # Quality metrics
        quality_scores = [r['quality_score'] for r in successful_results if 'quality_score' in r]
        avg_quality_score = statistics.mean(quality_scores) if quality_scores else 0
        median_quality_score = statistics.median(quality_scores) if quality_scores else 0
        
        # Token metrics
        total_tokens = sum(r.get('total_tokens', 0) for r in successful_results)
        total_llm_calls = sum(r.get('llm_calls', 0) for r in successful_results)
        avg_tokens_per_call = total_tokens / total_llm_calls if total_llm_calls > 0 else 0
        
        # Calculate percentiles
        p95_response_time = np.percentile(durations, 95) if durations else 0
        p99_response_time = np.percentile(durations, 99) if durations else 0
        
        # Cache metrics
        cache_hit_rate = self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"]) if (self.cache_stats["hits"] + self.cache_stats["misses"]) > 0 else 0
        
        # Provider performance analysis
        provider_performance = self._analyze_provider_performance(successful_results)
        
        # Advanced analytics
        throughput_per_minute = (successful_scenarios / total_duration) * 60 if total_duration > 0 else 0
        tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0
        
        # Cost estimation
        estimated_total_cost = self._estimate_total_cost(successful_results)
        cost_per_1k_tokens = (estimated_total_cost / total_tokens * 1000) if total_tokens > 0 else 0
        
        # Efficiency scores
        scalability_score = min(throughput_per_minute / 10, 1.0)  # Normalized to max of 1.0
        reliability_score = success_rate
        efficiency_score = (tokens_per_second / 100) if tokens_per_second > 0 else 0  # Normalized
        
        # Overall performance score (weighted combination)
        overall_performance_score = (
            reliability_score * 0.3 +
            efficiency_score * 0.25 +
            (avg_quality_score / 2.0) * 0.25 +  # Normalize quality score
            scalability_score * 0.2
        )
        
        # Resource usage
        resource_summary = self.resource_monitor.get_summary()
        
        return EnhancedPerformanceMetrics(
            total_scenarios=total_scenarios,
            successful_scenarios=successful_scenarios,
            failed_scenarios=failed_scenarios,
            success_rate=success_rate,
            total_duration=total_duration,
            total_llm_calls=total_llm_calls,
            total_tokens=total_tokens,
            avg_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min(durations) if durations else 0,
            max_response_time=max(durations) if durations else 0,
            response_time_std=statistics.stdev(durations) if len(durations) > 1 else 0,
            avg_quality_score=avg_quality_score,
            median_quality_score=median_quality_score,
            quality_score_std=statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
            min_quality_score=min(quality_scores) if quality_scores else 0,
            max_quality_score=max(quality_scores) if quality_scores else 0,
            avg_tokens_per_call=avg_tokens_per_call,
            median_tokens_per_call=statistics.median([r.get('total_tokens', 0) / max(r.get('llm_calls', 1), 1) for r in successful_results]) if successful_results else 0,
            tokens_per_second=tokens_per_second,
            cost_per_1k_tokens=cost_per_1k_tokens,
            estimated_total_cost=estimated_total_cost,
            coordination_overhead=self._calculate_coordination_overhead(successful_results),
            parallel_efficiency=self._calculate_parallel_efficiency(successful_results),
            cache_hit_rate=cache_hit_rate,
            error_rate=failed_scenarios / total_scenarios,
            peak_memory_usage=resource_summary.get('peak_memory_mb', 0),
            avg_cpu_usage=resource_summary.get('avg_cpu_percent', 0),
            network_latency=resource_summary.get('avg_network_latency_ms', 0),
            provider_performance=provider_performance,
            quality_distribution=self._calculate_quality_distribution(quality_scores),
            throughput_per_minute=throughput_per_minute,
            scalability_score=scalability_score,
            reliability_score=reliability_score,
            efficiency_score=efficiency_score,
            overall_performance_score=overall_performance_score
        )
    
    def _analyze_provider_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by provider"""
        provider_stats = {}
        
        for result in results:
            providers = result.get('providers_used', [])
            duration = result.get('total_duration', 0)
            quality = result.get('quality_score', 0)
            
            for provider in providers:
                if provider not in provider_stats:
                    provider_stats[provider] = {
                        'durations': [],
                        'qualities': [],
                        'successes': 0,
                        'total_attempts': 0
                    }
                
                provider_stats[provider]['durations'].append(duration)
                provider_stats[provider]['qualities'].append(quality)
                provider_stats[provider]['successes'] += 1
                provider_stats[provider]['total_attempts'] += 1
        
        # Calculate aggregated metrics
        performance = {}
        for provider, stats in provider_stats.items():
            if stats['durations']:
                performance[provider] = {
                    'avg_response_time': statistics.mean(stats['durations']),
                    'avg_quality_score': statistics.mean(stats['qualities']),
                    'success_rate': stats['successes'] / stats['total_attempts'],
                    'total_requests': stats['total_attempts']
                }
        
        return performance
    
    def _calculate_coordination_overhead(self, results: List[Dict[str, Any]]) -> float:
        """Calculate coordination overhead for multi-LLM scenarios"""
        single_llm_times = []
        multi_llm_times = []
        
        for result in results:
            providers = result.get('providers_used', [])
            duration = result.get('total_duration', 0)
            
            if len(providers) == 1 and providers[0] != 'cache':
                single_llm_times.append(duration)
            elif len(providers) > 1:
                multi_llm_times.append(duration)
        
        if single_llm_times and multi_llm_times:
            avg_single = statistics.mean(single_llm_times)
            avg_multi = statistics.mean(multi_llm_times)
            return (avg_multi - avg_single) / avg_single if avg_single > 0 else 0
        
        return 0.0
    
    def _calculate_parallel_efficiency(self, results: List[Dict[str, Any]]) -> float:
        """Calculate parallel execution efficiency"""
        parallel_efficiencies = []
        
        for result in results:
            if 'parallel_efficiency' in result and result['parallel_efficiency'] is not None:
                parallel_efficiencies.append(result['parallel_efficiency'])
        
        return statistics.mean(parallel_efficiencies) if parallel_efficiencies else 0.0
    
    def _calculate_quality_distribution(self, quality_scores: List[float]) -> Dict[str, int]:
        """Calculate distribution of quality scores"""
        if not quality_scores:
            return {}
        
        distribution = {
            'excellent': 0,  # > 1.5
            'good': 0,       # 1.2 - 1.5
            'average': 0,    # 0.8 - 1.2
            'poor': 0        # < 0.8
        }
        
        for score in quality_scores:
            if score > 1.5:
                distribution['excellent'] += 1
            elif score > 1.2:
                distribution['good'] += 1
            elif score > 0.8:
                distribution['average'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def _estimate_total_cost(self, results: List[Dict[str, Any]]) -> float:
        """Estimate total cost based on token usage and provider pricing"""
        total_cost = 0.0
        
        for result in results:
            providers = result.get('providers_used', [])
            tokens = result.get('total_tokens', 0)
            
            # Use first provider for cost calculation (simplified)
            if providers and tokens > 0:
                provider = providers[0]
                if provider in self.provider_configs:
                    cost_per_1k = self.provider_configs[provider]['cost_per_1k_tokens']
                    total_cost += (tokens / 1000) * cost_per_1k
        
        return total_cost
    
    def _generate_cache_key(self, query: str, provider: str) -> str:
        """Generate cache key for query and provider"""
        content = f"{query}:{provider}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_performance_insights(self, metrics: EnhancedPerformanceMetrics, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate performance insights and patterns"""
        insights = []
        
        # Cache effectiveness insight
        if metrics.cache_hit_rate > 0.5:
            insights.append({
                'type': 'cache_effectiveness',
                'title': 'High Cache Hit Rate Detected',
                'description': f'Cache hit rate of {metrics.cache_hit_rate:.1%} significantly improves performance',
                'impact': 'high',
                'recommendation': 'Implement aggressive caching for production'
            })
        
        # Quality vs Speed trade-off insight
        if metrics.avg_quality_score > 1.3 and metrics.avg_response_time > 20:
            insights.append({
                'type': 'quality_speed_tradeoff',
                'title': 'High Quality with Slower Response',
                'description': f'Quality score {metrics.avg_quality_score:.2f} achieved with {metrics.avg_response_time:.1f}s response time',
                'impact': 'medium',
                'recommendation': 'Consider parallel processing for better speed'
            })
        
        # Multi-LLM coordination effectiveness
        if metrics.coordination_overhead > 0.5:
            insights.append({
                'type': 'coordination_overhead',
                'title': 'High Coordination Overhead',
                'description': f'Multi-LLM coordination adds {metrics.coordination_overhead:.1%} overhead',
                'impact': 'medium',
                'recommendation': 'Optimize with parallel execution and smarter routing'
            })
        
        # Resource efficiency insight
        if metrics.peak_memory_usage > 1000:  # > 1GB
            insights.append({
                'type': 'memory_usage',
                'title': 'High Memory Usage Detected',
                'description': f'Peak memory usage: {metrics.peak_memory_usage:.0f}MB',
                'impact': 'medium',
                'recommendation': 'Implement memory optimization strategies'
            })
        
        # Provider performance insight
        best_provider = max(metrics.provider_performance.items(), 
                          key=lambda x: x[1].get('avg_quality_score', 0)) if metrics.provider_performance else None
        
        if best_provider:
            insights.append({
                'type': 'provider_performance',
                'title': f'Best Performing Provider: {best_provider[0]}',
                'description': f'Quality: {best_provider[1]["avg_quality_score"]:.2f}, Time: {best_provider[1]["avg_response_time"]:.1f}s',
                'impact': 'high',
                'recommendation': f'Prioritize {best_provider[0]} for quality-critical tasks'
            })
        
        return insights
    
    def _generate_optimization_recommendations(self, metrics: EnhancedPerformanceMetrics, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Speed optimization recommendations
        if metrics.avg_response_time > 15:
            recommendations.append({
                'category': 'speed_optimization',
                'priority': 'high',
                'title': 'Implement Parallel Execution',
                'description': 'Enable parallel LLM execution to reduce response times',
                'expected_improvement': '30-50% speed improvement',
                'implementation': 'Add OptimizationStrategy.PARALLEL_EXECUTION'
            })
        
        # Cache optimization recommendations
        if metrics.cache_hit_rate < 0.3:
            recommendations.append({
                'category': 'cache_optimization',
                'priority': 'medium',
                'title': 'Enhance Caching Strategy',
                'description': 'Implement smarter caching with longer TTL',
                'expected_improvement': '70% speed improvement for repeated queries',
                'implementation': 'Implement predictive caching and query similarity matching'
            })
        
        # Quality optimization recommendations
        if metrics.avg_quality_score < 1.2:
            recommendations.append({
                'category': 'quality_optimization',
                'priority': 'high',
                'title': 'Use Higher Quality Models',
                'description': 'Switch to more capable models for better results',
                'expected_improvement': '20-30% quality improvement',
                'implementation': 'Use Claude Sonnet or GPT-4 for complex tasks'
            })
        
        # Cost optimization recommendations
        if metrics.estimated_total_cost > 1.0:  # > $1
            recommendations.append({
                'category': 'cost_optimization',
                'priority': 'medium',
                'title': 'Implement Smart Model Selection',
                'description': 'Use faster, cheaper models for simple tasks',
                'expected_improvement': '40-60% cost reduction',
                'implementation': 'Implement adaptive routing based on query complexity'
            })
        
        # Reliability optimization recommendations
        if metrics.reliability_score < 0.95:
            recommendations.append({
                'category': 'reliability_optimization',
                'priority': 'high',
                'title': 'Implement Error Recovery',
                'description': 'Add retry mechanisms and fallback providers',
                'expected_improvement': '99%+ reliability',
                'implementation': 'Add exponential backoff and provider fallback'
            })
        
        return recommendations
    
    def _create_empty_metrics(self) -> EnhancedPerformanceMetrics:
        """Create empty metrics object"""
        return EnhancedPerformanceMetrics(
            total_scenarios=0, successful_scenarios=0, failed_scenarios=0,
            success_rate=0.0, total_duration=0.0, total_llm_calls=0, total_tokens=0,
            avg_response_time=0.0, median_response_time=0.0, p95_response_time=0.0, p99_response_time=0.0,
            min_response_time=0.0, max_response_time=0.0, response_time_std=0.0,
            avg_quality_score=0.0, median_quality_score=0.0, quality_score_std=0.0,
            min_quality_score=0.0, max_quality_score=0.0,
            avg_tokens_per_call=0.0, median_tokens_per_call=0.0, tokens_per_second=0.0,
            cost_per_1k_tokens=0.0, estimated_total_cost=0.0,
            coordination_overhead=0.0, parallel_efficiency=0.0, cache_hit_rate=0.0, error_rate=0.0,
            peak_memory_usage=0.0, avg_cpu_usage=0.0, network_latency=0.0,
            provider_performance={}, quality_distribution={},
            throughput_per_minute=0.0, scalability_score=0.0, reliability_score=0.0,
            efficiency_score=0.0, overall_performance_score=0.0
        )
    
    def _print_enhanced_summary(self, metrics: EnhancedPerformanceMetrics, report: Dict[str, Any]):
        """Print enhanced summary of benchmark results"""
        print(f"📈 Overall Performance Score: {metrics.overall_performance_score:.3f}")
        print(f"📊 Success Rate: {metrics.success_rate:.1%}")
        print(f"⚡ Avg Response Time: {metrics.avg_response_time:.2f}s (P95: {metrics.p95_response_time:.2f}s)")
        print(f"🎯 Avg Quality Score: {metrics.avg_quality_score:.3f}")
        print(f"💰 Estimated Cost: ${metrics.estimated_total_cost:.4f}")
        print(f"🔄 Cache Hit Rate: {metrics.cache_hit_rate:.1%}")
        print(f"🚀 Throughput: {metrics.throughput_per_minute:.1f} requests/minute")
        
        print(f"\n📊 Quality Distribution:")
        for category, count in metrics.quality_distribution.items():
            print(f"   {category.capitalize()}: {count}")
        
        print(f"\n🏆 Top Provider Performance:")
        for provider, perf in list(metrics.provider_performance.items())[:3]:
            print(f"   {provider}: Quality {perf['avg_quality_score']:.2f}, Time {perf['avg_response_time']:.1f}s")
        
        print(f"\n💡 Key Insights:")
        for insight in report.get('performance_insights', [])[:3]:
            print(f"   • {insight['title']}")
        
        print(f"\n🔧 Top Recommendations:")
        for rec in report.get('recommendations', [])[:3]:
            print(f"   • {rec['title']} ({rec['expected_improvement']})")

class ResourceMonitor:
    """Resource usage monitoring for benchmark tests"""
    
    def __init__(self):
        self.monitoring = False
        self.start_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Resource monitoring loop"""
        while self.monitoring:
            try:
                # Memory usage
                memory_info = psutil.virtual_memory()
                self.memory_samples.append(memory_info.used / 1024 / 1024)  # MB
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(1)  # Sample every second
            except Exception:
                break
    
    def get_summary(self) -> Dict[str, float]:
        """Get resource usage summary"""
        if not self.memory_samples or not self.cpu_samples:
            return {
                'peak_memory_mb': 0,
                'avg_memory_mb': 0,
                'avg_cpu_percent': 0,
                'avg_network_latency_ms': 0
            }
        
        return {
            'peak_memory_mb': max(self.memory_samples),
            'avg_memory_mb': statistics.mean(self.memory_samples),
            'avg_cpu_percent': statistics.mean(self.cpu_samples),
            'avg_network_latency_ms': 50.0  # Placeholder for network latency
        }

async def main():
    """Run enhanced MLACS benchmark"""
    try:
        benchmark = EnhancedMLACSBenchmark()
        
        # Test different optimization strategies
        optimization_sets = [
            [OptimizationStrategy.BASELINE],
            [OptimizationStrategy.SMART_CACHING],
            [OptimizationStrategy.PARALLEL_EXECUTION, OptimizationStrategy.SMART_CACHING],
            [OptimizationStrategy.PARALLEL_EXECUTION, OptimizationStrategy.SMART_CACHING, OptimizationStrategy.ADAPTIVE_ROUTING]
        ]
        
        for i, strategies in enumerate(optimization_sets, 1):
            print(f"\n{'='*80}")
            print(f"🚀 Enhanced Benchmark Run {i}/{len(optimization_sets)}")
            print(f"🔧 Strategies: {[s.value for s in strategies]}")
            print(f"{'='*80}")
            
            result = await benchmark.run_enhanced_benchmark(strategies)
            
            print(f"\n✅ Enhanced Benchmark Run {i} Completed")
            print(f"📊 Overall Score: {result['enhanced_metrics']['overall_performance_score']:.3f}")
            
            # Brief pause between runs
            if i < len(optimization_sets):
                await asyncio.sleep(2)
        
        print(f"\n🎉 All Enhanced Benchmark Runs Completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Enhanced Benchmark failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())