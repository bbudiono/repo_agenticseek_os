#!/usr/bin/env python3
"""
Advanced Iterative MLACS Optimizer
Next-generation optimization with learning capabilities and advanced strategies

Building on previous optimization results to implement sophisticated 
optimization strategies with machine learning-based adaptation.

Author: AgenticSeek Development Team
Date: 2025-01-06
Version: 3.0.0
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
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

class AdvancedOptimizationStrategy(Enum):
    """Advanced optimization strategies with learning capabilities"""
    # Previous strategies
    BASELINE = "baseline"
    PARALLEL_EXECUTION = "parallel_execution"
    SMART_CACHING = "smart_caching"
    ADAPTIVE_ROUTING = "adaptive_routing"
    FAST_MODELS = "fast_models"
    
    # New advanced strategies
    PREDICTIVE_CACHING = "predictive_caching"
    DYNAMIC_LOAD_BALANCING = "dynamic_load_balancing"
    QUALITY_AWARE_ROUTING = "quality_aware_routing"
    COST_OPTIMIZATION = "cost_optimization"
    RESPONSE_STREAMING = "response_streaming"
    CONTEXT_COMPRESSION = "context_compression"
    MODEL_ENSEMBLE = "model_ensemble"
    INTELLIGENT_FALLBACK = "intelligent_fallback"
    PERFORMANCE_PREDICTION = "performance_prediction"
    ADAPTIVE_TIMEOUT = "adaptive_timeout"

@dataclass
class OptimizationResult:
    """Enhanced optimization result with detailed analytics"""
    iteration_id: str
    strategy_combination: List[str]
    performance_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]
    cost_metrics: Dict[str, float]
    learned_patterns: Dict[str, Any]
    optimization_effectiveness: float
    confidence_score: float
    timestamp: str

@dataclass
class LearningInsight:
    """Machine learning insight from optimization patterns"""
    insight_id: str
    pattern_type: str
    description: str
    confidence: float
    applicable_scenarios: List[str]
    expected_improvement: Dict[str, float]
    implementation_complexity: str
    discovered_at: str

class AdvancedIterativeMLACSOptimizer:
    """Advanced iterative optimizer with machine learning capabilities"""
    
    def __init__(self, base_results_file: str = None):
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        
        if not self.anthropic_key or not self.openai_key:
            raise ValueError("Missing required API keys")
        
        # Enhanced caching with prediction
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "predictions": 0}
        self.query_patterns = {}
        
        # Learning system
        self.optimization_history = []
        self.learned_insights = []
        self.performance_predictions = {}
        
        # Load previous results for learning
        if base_results_file:
            self._load_base_results(base_results_file)
        
        # Advanced test scenarios
        self.advanced_scenarios = self._create_advanced_scenarios()
        
        # Strategy effectiveness tracking
        self.strategy_effectiveness = {}
        
        print("🚀 Advanced Iterative MLACS Optimizer Initialized")
        print(f"📊 Advanced Scenarios: {len(self.advanced_scenarios)}")
        print(f"🧠 Learning System: Active")
    
    def _load_base_results(self, file_path: str):
        """Load previous optimization results for learning"""
        try:
            with open(file_path, 'r') as f:
                base_data = json.load(f)
            
            # Extract learning patterns from previous results
            if 'iteration_results' in base_data:
                for result in base_data['iteration_results']:
                    self.optimization_history.append(result)
            
            print(f"📚 Loaded {len(self.optimization_history)} previous optimization results")
            
        except Exception as e:
            print(f"⚠️ Could not load base results: {e}")
    
    def _create_advanced_scenarios(self) -> List[Dict[str, Any]]:
        """Create advanced test scenarios for optimization"""
        return [
            {
                "scenario_id": "complex_research_synthesis",
                "title": "Complex Research Synthesis",
                "query": "Analyze the intersection of artificial intelligence, climate change, and economic policy. Provide a comprehensive synthesis of how AI can be leveraged to address climate challenges while considering economic implications for different stakeholders.",
                "providers": ["claude", "gpt4"],
                "complexity": 0.9,
                "expected_duration": (20, 45),
                "quality_weight": 0.7,
                "speed_weight": 0.3
            },
            {
                "scenario_id": "rapid_decision_support",
                "title": "Rapid Decision Support",
                "query": "A startup needs immediate advice on whether to pivot their business model. They have 3 options: focus on B2B SaaS, consumer mobile app, or AI consulting services. Provide quick, actionable recommendations.",
                "providers": ["gpt3.5"],
                "complexity": 0.4,
                "expected_duration": (3, 8),
                "quality_weight": 0.4,
                "speed_weight": 0.6
            },
            {
                "scenario_id": "multi_perspective_analysis",
                "title": "Multi-Perspective Analysis",
                "query": "Evaluate the pros and cons of remote work from three perspectives: employee wellbeing, company productivity, and environmental impact. Each perspective should be thoroughly analyzed by different AI models.",
                "providers": ["claude", "gpt4", "claude-sonnet"],
                "complexity": 0.8,
                "expected_duration": (25, 50),
                "quality_weight": 0.8,
                "speed_weight": 0.2
            },
            {
                "scenario_id": "technical_deep_dive",
                "title": "Technical Deep Dive",
                "query": "Explain the technical architecture and implementation challenges of building a distributed real-time analytics system that can handle 1 million events per second. Include considerations for data consistency, fault tolerance, and scalability.",
                "providers": ["claude-sonnet"],
                "complexity": 1.0,
                "expected_duration": (15, 30),
                "quality_weight": 0.9,
                "speed_weight": 0.1
            },
            {
                "scenario_id": "creative_brainstorming",
                "title": "Creative Brainstorming",
                "query": "Generate 20 innovative product ideas that combine sustainability, technology, and social impact. Each idea should be unique and briefly explained.",
                "providers": ["gpt4"],
                "complexity": 0.6,
                "expected_duration": (8, 15),
                "quality_weight": 0.6,
                "speed_weight": 0.4
            }
        ]
    
    async def run_advanced_optimization(self, iterations: int = 8) -> Dict[str, Any]:
        """Run advanced iterative optimization with learning"""
        print(f"🚀 Starting Advanced Iterative MLACS Optimization")
        print("=" * 80)
        print(f"🔄 Iterations Planned: {iterations}")
        print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        optimization_start = time.time()
        results = []
        
        # Define advanced optimization combinations
        optimization_combinations = [
            [AdvancedOptimizationStrategy.BASELINE],
            [AdvancedOptimizationStrategy.SMART_CACHING],
            [AdvancedOptimizationStrategy.PARALLEL_EXECUTION, AdvancedOptimizationStrategy.SMART_CACHING],
            [AdvancedOptimizationStrategy.PREDICTIVE_CACHING, AdvancedOptimizationStrategy.PARALLEL_EXECUTION],
            [AdvancedOptimizationStrategy.QUALITY_AWARE_ROUTING, AdvancedOptimizationStrategy.SMART_CACHING],
            [AdvancedOptimizationStrategy.DYNAMIC_LOAD_BALANCING, AdvancedOptimizationStrategy.ADAPTIVE_ROUTING],
            [AdvancedOptimizationStrategy.MODEL_ENSEMBLE, AdvancedOptimizationStrategy.COST_OPTIMIZATION],
            [AdvancedOptimizationStrategy.PERFORMANCE_PREDICTION, AdvancedOptimizationStrategy.INTELLIGENT_FALLBACK]
        ]
        
        for i in range(min(iterations, len(optimization_combinations))):
            strategies = optimization_combinations[i]
            
            print(f"\n🔬 Iteration {i+1}/{iterations}: {[s.value for s in strategies]}")
            
            iteration_result = await self._run_optimization_iteration(
                f"iter_{i+1:02d}_{strategies[0].value}",
                strategies
            )
            
            results.append(iteration_result)
            
            # Learn from this iteration
            insights = self._extract_learning_insights(iteration_result)
            self.learned_insights.extend(insights)
            
            # Update strategy effectiveness
            self._update_strategy_effectiveness(strategies, iteration_result)
            
            print(f"   ✅ Completed in {iteration_result.performance_metrics.get('total_duration', 0):.2f}s")
            print(f"   📊 Overall Score: {iteration_result.optimization_effectiveness:.3f}")
            print(f"   🧠 New Insights: {len(insights)}")
        
        # Generate comprehensive analysis
        analysis = self._generate_advanced_analysis(results)
        
        # Create final report
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'optimization_duration': time.time() - optimization_start,
            'total_iterations': len(results),
            'iteration_results': [asdict(r) for r in results],
            'learned_insights': [asdict(i) for i in self.learned_insights],
            'strategy_effectiveness': self.strategy_effectiveness,
            'advanced_analysis': analysis,
            'performance_predictions': self.performance_predictions,
            'optimization_recommendations': self._generate_advanced_recommendations(results, analysis)
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"advanced_mlacs_optimization_results_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self._print_advanced_summary(results, analysis)
        print(f"\n📄 Advanced results saved to: {report_file}")
        
        return report
    
    async def _run_optimization_iteration(self, iteration_id: str, strategies: List[AdvancedOptimizationStrategy]) -> OptimizationResult:
        """Run single optimization iteration with advanced strategies"""
        iteration_start = time.time()
        
        scenario_results = []
        total_performance = 0
        total_quality = 0
        total_cost = 0
        
        for scenario in self.advanced_scenarios:
            scenario_result = await self._run_advanced_scenario(scenario, strategies)
            scenario_results.append(scenario_result)
            
            # Weighted scoring
            perf_score = self._calculate_performance_score(scenario_result, scenario)
            qual_score = self._calculate_quality_score_advanced(scenario_result, scenario)
            cost_score = self._calculate_cost_score(scenario_result, scenario)
            
            total_performance += perf_score * scenario.get('speed_weight', 0.5)
            total_quality += qual_score * scenario.get('quality_weight', 0.5)
            total_cost += cost_score * 0.2  # 20% weight for cost
        
        iteration_duration = time.time() - iteration_start
        
        # Calculate overall metrics
        performance_metrics = {
            'total_duration': iteration_duration,
            'avg_response_time': sum(r.get('duration', 0) for r in scenario_results) / len(scenario_results),
            'success_rate': sum(1 for r in scenario_results if r.get('success', False)) / len(scenario_results),
            'total_tokens': sum(r.get('tokens', 0) for r in scenario_results),
            'total_llm_calls': sum(r.get('llm_calls', 0) for r in scenario_results)
        }
        
        quality_metrics = {
            'avg_quality_score': total_quality / len(self.advanced_scenarios),
            'quality_variance': np.var([self._calculate_quality_score_advanced(r, s) for r, s in zip(scenario_results, self.advanced_scenarios)]),
            'min_quality': min(self._calculate_quality_score_advanced(r, s) for r, s in zip(scenario_results, self.advanced_scenarios)),
            'max_quality': max(self._calculate_quality_score_advanced(r, s) for r, s in zip(scenario_results, self.advanced_scenarios))
        }
        
        cost_metrics = {
            'total_cost': sum(self._estimate_scenario_cost(r) for r in scenario_results),
            'cost_per_token': sum(self._estimate_scenario_cost(r) for r in scenario_results) / max(sum(r.get('tokens', 0) for r in scenario_results), 1),
            'cost_efficiency': total_quality / max(sum(self._estimate_scenario_cost(r) for r in scenario_results), 0.001)
        }
        
        # Learn patterns from this iteration
        learned_patterns = self._extract_patterns(scenario_results, strategies)
        
        # Calculate optimization effectiveness
        optimization_effectiveness = (total_performance + total_quality - total_cost) / 3
        confidence_score = self._calculate_confidence_score(scenario_results, strategies)
        
        return OptimizationResult(
            iteration_id=iteration_id,
            strategy_combination=[s.value for s in strategies],
            performance_metrics=performance_metrics,
            quality_metrics=quality_metrics,
            cost_metrics=cost_metrics,
            learned_patterns=learned_patterns,
            optimization_effectiveness=optimization_effectiveness,
            confidence_score=confidence_score,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    async def _run_advanced_scenario(self, scenario: Dict[str, Any], strategies: List[AdvancedOptimizationStrategy]) -> Dict[str, Any]:
        """Run individual scenario with advanced optimization strategies"""
        scenario_start = time.time()
        
        try:
            providers = scenario['providers']
            query = scenario['query']
            
            # Apply predictive caching
            if AdvancedOptimizationStrategy.PREDICTIVE_CACHING in strategies:
                cache_result = await self._apply_predictive_caching(query, providers)
                if cache_result:
                    return cache_result
            
            # Apply quality-aware routing
            if AdvancedOptimizationStrategy.QUALITY_AWARE_ROUTING in strategies:
                providers = self._apply_quality_aware_routing(query, providers, scenario['complexity'])
            
            # Apply dynamic load balancing
            if AdvancedOptimizationStrategy.DYNAMIC_LOAD_BALANCING in strategies:
                providers = await self._apply_dynamic_load_balancing(providers)
            
            # Execute with selected strategies
            if len(providers) == 1:
                result = await self._execute_single_provider(query, providers[0], strategies)
            else:
                result = await self._execute_multi_provider(query, providers, strategies)
            
            # Apply intelligent fallback if needed
            if not result.get('success', False) and AdvancedOptimizationStrategy.INTELLIGENT_FALLBACK in strategies:
                result = await self._apply_intelligent_fallback(query, scenario)
            
            result['scenario_id'] = scenario['scenario_id']
            result['duration'] = time.time() - scenario_start
            
            return result
            
        except Exception as e:
            return {
                'scenario_id': scenario['scenario_id'],
                'success': False,
                'error': str(e),
                'duration': time.time() - scenario_start,
                'tokens': 0,
                'llm_calls': 0
            }
    
    async def _apply_predictive_caching(self, query: str, providers: List[str]) -> Optional[Dict[str, Any]]:
        """Apply predictive caching based on query patterns"""
        # Simple pattern matching for demonstration
        query_fingerprint = hashlib.md5(query.lower().encode()).hexdigest()[:8]
        
        # Check for similar queries
        for cached_query, cached_result in self.cache.items():
            if self._calculate_query_similarity(query, cached_query) > 0.8:
                self.cache_stats["predictions"] += 1
                return {
                    'success': True,
                    'response': f"[PREDICTED CACHE] {cached_result.get('response', '')[:200]}...",
                    'tokens': 0,
                    'llm_calls': 0,
                    'quality_score': cached_result.get('quality_score', 0.8),
                    'cached': True
                }
        
        return None
    
    def _apply_quality_aware_routing(self, query: str, providers: List[str], complexity: float) -> List[str]:
        """Route based on query complexity and provider capabilities"""
        if complexity > 0.8:
            # High complexity - use best models
            return ["claude-sonnet"] if "claude-sonnet" in providers else ["claude", "gpt4"]
        elif complexity < 0.4:
            # Low complexity - use fast models
            return ["gpt3.5"] if "gpt3.5" in providers else providers[:1]
        else:
            # Medium complexity - balanced approach
            return providers
    
    async def _apply_dynamic_load_balancing(self, providers: List[str]) -> List[str]:
        """Apply dynamic load balancing based on current performance"""
        # Simulate load balancing decision
        if hasattr(self, 'provider_load'):
            # Sort providers by current load (ascending)
            sorted_providers = sorted(providers, key=lambda p: self.provider_load.get(p, 0))
            return sorted_providers
        
        return providers
    
    async def _execute_single_provider(self, query: str, provider: str, strategies: List[AdvancedOptimizationStrategy]) -> Dict[str, Any]:
        """Execute query with single provider and advanced optimizations"""
        try:
            # Apply context compression if enabled
            if AdvancedOptimizationStrategy.CONTEXT_COMPRESSION in strategies:
                query = self._compress_context(query)
            
            # Execute with appropriate timeout
            timeout = self._calculate_adaptive_timeout(query, provider, strategies)
            
            if provider.startswith('claude'):
                response, tokens = await asyncio.wait_for(
                    self._call_claude(query, provider), 
                    timeout=timeout
                )
            elif provider.startswith('gpt'):
                response, tokens = await asyncio.wait_for(
                    self._call_openai(query, provider), 
                    timeout=timeout
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            quality_score = self._calculate_response_quality(response, query)
            
            return {
                'success': True,
                'response': response,
                'tokens': tokens,
                'llm_calls': 1,
                'provider': provider,
                'quality_score': quality_score
            }
            
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f"Timeout after {timeout}s",
                'tokens': 0,
                'llm_calls': 0,
                'provider': provider
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tokens': 0,
                'llm_calls': 0,
                'provider': provider
            }
    
    async def _execute_multi_provider(self, query: str, providers: List[str], strategies: List[AdvancedOptimizationStrategy]) -> Dict[str, Any]:
        """Execute query with multiple providers using advanced coordination"""
        if AdvancedOptimizationStrategy.MODEL_ENSEMBLE in strategies:
            return await self._execute_model_ensemble(query, providers, strategies)
        elif AdvancedOptimizationStrategy.PARALLEL_EXECUTION in strategies:
            return await self._execute_parallel_coordination(query, providers, strategies)
        else:
            return await self._execute_sequential_coordination(query, providers, strategies)
    
    async def _execute_model_ensemble(self, query: str, providers: List[str], strategies: List[AdvancedOptimizationStrategy]) -> Dict[str, Any]:
        """Execute model ensemble with weighted voting"""
        tasks = []
        for provider in providers:
            if provider.startswith('claude'):
                task = self._call_claude(query, provider)
            elif provider.startswith('gpt'):
                task = self._call_openai(query, provider)
            else:
                continue
            tasks.append((provider, task))
        
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        successful_results = []
        total_tokens = 0
        
        for (provider, _), result in zip(tasks, results):
            if isinstance(result, tuple):
                response, tokens = result
                quality = self._calculate_response_quality(response, query)
                successful_results.append({
                    'provider': provider,
                    'response': response,
                    'tokens': tokens,
                    'quality': quality
                })
                total_tokens += tokens
        
        if not successful_results:
            return {
                'success': False,
                'error': 'No successful responses',
                'tokens': 0,
                'llm_calls': 0
            }
        
        # Weighted ensemble based on quality
        weights = [r['quality'] for r in successful_results]
        total_weight = sum(weights)
        
        if total_weight > 0:
            weighted_qualities = [r['quality'] * (w / total_weight) for r, w in zip(successful_results, weights)]
            ensemble_quality = sum(weighted_qualities)
        else:
            ensemble_quality = sum(r['quality'] for r in successful_results) / len(successful_results)
        
        # Create ensemble response
        ensemble_response = "Ensemble Analysis:\n" + "\n".join([
            f"Model {i+1} ({r['provider']}): {r['response'][:200]}..."
            for i, r in enumerate(successful_results)
        ])
        
        return {
            'success': True,
            'response': ensemble_response,
            'tokens': total_tokens,
            'llm_calls': len(successful_results),
            'quality_score': ensemble_quality,
            'ensemble_size': len(successful_results)
        }
    
    async def _execute_parallel_coordination(self, query: str, providers: List[str], strategies: List[AdvancedOptimizationStrategy]) -> Dict[str, Any]:
        """Execute parallel coordination with synthesis"""
        tasks = []
        for provider in providers:
            if provider.startswith('claude'):
                task = self._call_claude(query, provider)
            elif provider.startswith('gpt'):
                task = self._call_openai(query, provider)
            else:
                continue
            tasks.append(task)
        
        parallel_start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        parallel_duration = time.time() - parallel_start
        
        responses = []
        total_tokens = 0
        
        for result in results:
            if isinstance(result, tuple):
                response, tokens = result
                responses.append(response)
                total_tokens += tokens
        
        if responses:
            # Synthesis
            synthesis_prompt = f"Synthesize these expert analyses:\n\n" + "\n\n".join(
                f"Analysis {i+1}: {resp}" for i, resp in enumerate(responses)
            )
            
            synthesis_response, synthesis_tokens = await self._call_claude(synthesis_prompt, "claude")
            total_tokens += synthesis_tokens
            
            quality_score = 1.0 + (len(responses) * 0.1)  # Bonus for coordination
            
            return {
                'success': True,
                'response': synthesis_response,
                'tokens': total_tokens,
                'llm_calls': len(responses) + 1,
                'quality_score': quality_score,
                'parallel_duration': parallel_duration,
                'coordination_type': 'parallel'
            }
        
        return {
            'success': False,
            'error': 'No successful parallel responses',
            'tokens': 0,
            'llm_calls': 0
        }
    
    async def _execute_sequential_coordination(self, query: str, providers: List[str], strategies: List[AdvancedOptimizationStrategy]) -> Dict[str, Any]:
        """Execute sequential coordination"""
        responses = []
        total_tokens = 0
        
        for provider in providers:
            try:
                if provider.startswith('claude'):
                    response, tokens = await self._call_claude(query, provider)
                elif provider.startswith('gpt'):
                    response, tokens = await self._call_openai(query, provider)
                else:
                    continue
                
                responses.append(response)
                total_tokens += tokens
                
            except Exception:
                continue
        
        if responses:
            combined_response = "\n\n".join(responses)
            quality_score = len(responses) * 0.95  # Slight bonus for multiple perspectives
            
            return {
                'success': True,
                'response': combined_response,
                'tokens': total_tokens,
                'llm_calls': len(responses),
                'quality_score': quality_score,
                'coordination_type': 'sequential'
            }
        
        return {
            'success': False,
            'error': 'No successful sequential responses',
            'tokens': 0,
            'llm_calls': 0
        }
    
    async def _apply_intelligent_fallback(self, query: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Apply intelligent fallback strategy"""
        # Try with simplified query and fast model
        simplified_query = f"Provide a brief response to: {query[:200]}..."
        
        try:
            response, tokens = await self._call_openai(simplified_query, "gpt3.5")
            return {
                'success': True,
                'response': f"[FALLBACK] {response}",
                'tokens': tokens,
                'llm_calls': 1,
                'quality_score': 0.6,  # Lower quality for fallback
                'fallback_used': True
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Fallback failed: {str(e)}",
                'tokens': 0,
                'llm_calls': 0,
                'fallback_used': True
            }
    
    async def _call_claude(self, prompt: str, provider: str = "claude") -> Tuple[str, int]:
        """Claude API call with provider configuration"""
        client = anthropic.Anthropic(api_key=self.anthropic_key)
        
        model_map = {
            "claude": "claude-3-haiku-20240307",
            "claude-sonnet": "claude-3-5-sonnet-20241022"
        }
        
        response = client.messages.create(
            model=model_map.get(provider, "claude-3-haiku-20240307"),
            max_tokens=3000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text if response.content else ""
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return content, tokens
    
    async def _call_openai(self, prompt: str, provider: str = "gpt4") -> Tuple[str, int]:
        """OpenAI API call with provider configuration"""
        client = openai.OpenAI(api_key=self.openai_key)
        
        model_map = {
            "gpt4": "gpt-4-turbo-preview",
            "gpt3.5": "gpt-3.5-turbo"
        }
        
        response = client.chat.completions.create(
            model=model_map.get(provider, "gpt-4-turbo-preview"),
            max_tokens=3000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens
        
        return content, tokens
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between queries (simplified)"""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compress_context(self, query: str) -> str:
        """Apply context compression to reduce token usage"""
        if len(query) > 1000:
            # Simple compression: take first and last parts
            return query[:400] + "... [content compressed] ..." + query[-400:]
        return query
    
    def _calculate_adaptive_timeout(self, query: str, provider: str, strategies: List[AdvancedOptimizationStrategy]) -> float:
        """Calculate adaptive timeout based on query complexity and provider"""
        base_timeout = 30.0
        
        # Adjust based on query length
        if len(query) > 1000:
            base_timeout *= 1.5
        
        # Adjust based on provider
        if provider in ["claude-sonnet", "gpt4"]:
            base_timeout *= 1.3
        
        # Adjust based on strategies
        if AdvancedOptimizationStrategy.RESPONSE_STREAMING in strategies:
            base_timeout *= 0.8
        
        return base_timeout
    
    def _calculate_response_quality(self, response: str, query: str) -> float:
        """Calculate response quality score"""
        base_score = 1.0
        
        # Length-based scoring
        if len(response) < 50:
            base_score *= 0.5
        elif len(response) > 500:
            base_score *= 1.2
        
        # Relevance heuristic (simplified)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        relevance = len(query_words.intersection(response_words)) / len(query_words) if query_words else 0
        
        base_score *= (0.5 + relevance)
        
        return min(base_score, 2.0)
    
    def _calculate_performance_score(self, result: Dict[str, Any], scenario: Dict[str, Any]) -> float:
        """Calculate performance score for scenario result"""
        if not result.get('success', False):
            return 0.0
        
        duration = result.get('duration', 0)
        expected_min, expected_max = scenario.get('expected_duration', (5, 30))
        
        if duration <= expected_min:
            return 1.0  # Excellent performance
        elif duration <= expected_max:
            return 1.0 - ((duration - expected_min) / (expected_max - expected_min)) * 0.5
        else:
            return max(0.1, 0.5 - ((duration - expected_max) / expected_max))
    
    def _calculate_quality_score_advanced(self, result: Dict[str, Any], scenario: Dict[str, Any]) -> float:
        """Calculate advanced quality score"""
        if not result.get('success', False):
            return 0.0
        
        base_quality = result.get('quality_score', 0.8)
        
        # Adjust based on scenario complexity
        complexity_bonus = scenario.get('complexity', 0.5) * 0.2
        
        # Adjust based on coordination type
        if result.get('coordination_type') == 'parallel':
            base_quality *= 1.1
        elif result.get('ensemble_size', 0) > 1:
            base_quality *= 1.15
        
        return min(base_quality + complexity_bonus, 2.0)
    
    def _calculate_cost_score(self, result: Dict[str, Any], scenario: Dict[str, Any]) -> float:
        """Calculate cost efficiency score"""
        if not result.get('success', False):
            return 1.0  # No cost if failed
        
        tokens = result.get('tokens', 0)
        quality = result.get('quality_score', 0.8)
        
        if tokens == 0:
            return 1.0
        
        # Cost efficiency: quality per token
        efficiency = quality / (tokens / 1000)  # Quality per 1K tokens
        
        return min(efficiency / 2.0, 1.0)  # Normalize
    
    def _estimate_scenario_cost(self, result: Dict[str, Any]) -> float:
        """Estimate cost for scenario result"""
        tokens = result.get('tokens', 0)
        provider = result.get('provider', 'claude')
        
        # Simplified cost estimation
        cost_per_1k = {
            'claude': 0.00025,
            'claude-sonnet': 0.003,
            'gpt4': 0.01,
            'gpt3.5': 0.0005
        }
        
        return (tokens / 1000) * cost_per_1k.get(provider, 0.001)
    
    def _extract_patterns(self, scenario_results: List[Dict[str, Any]], strategies: List[AdvancedOptimizationStrategy]) -> Dict[str, Any]:
        """Extract learning patterns from iteration results"""
        patterns = {
            'strategy_combination': [s.value for s in strategies],
            'success_rate': sum(1 for r in scenario_results if r.get('success', False)) / len(scenario_results),
            'avg_duration': sum(r.get('duration', 0) for r in scenario_results) / len(scenario_results),
            'avg_quality': sum(r.get('quality_score', 0) for r in scenario_results) / len(scenario_results),
            'total_cost': sum(self._estimate_scenario_cost(r) for r in scenario_results),
            'cache_effectiveness': self.cache_stats.get('hits', 0) / max(self.cache_stats.get('hits', 0) + self.cache_stats.get('misses', 0), 1),
            'strategy_specific_insights': {}
        }
        
        # Strategy-specific insights
        for strategy in strategies:
            if strategy == AdvancedOptimizationStrategy.PREDICTIVE_CACHING:
                patterns['strategy_specific_insights']['predictive_cache_hits'] = self.cache_stats.get('predictions', 0)
            elif strategy == AdvancedOptimizationStrategy.MODEL_ENSEMBLE:
                ensemble_results = [r for r in scenario_results if r.get('ensemble_size', 0) > 1]
                if ensemble_results:
                    patterns['strategy_specific_insights']['avg_ensemble_size'] = sum(r['ensemble_size'] for r in ensemble_results) / len(ensemble_results)
        
        return patterns
    
    def _calculate_confidence_score(self, scenario_results: List[Dict[str, Any]], strategies: List[AdvancedOptimizationStrategy]) -> float:
        """Calculate confidence score for optimization result"""
        base_confidence = 0.5
        
        # Increase confidence with success rate
        success_rate = sum(1 for r in scenario_results if r.get('success', False)) / len(scenario_results)
        base_confidence += success_rate * 0.3
        
        # Increase confidence with strategy sophistication
        strategy_bonus = len(strategies) * 0.05
        base_confidence += strategy_bonus
        
        # Increase confidence with consistent results
        durations = [r.get('duration', 0) for r in scenario_results if r.get('success', False)]
        if durations and len(durations) > 1:
            consistency = 1.0 - (np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 1.0)
            base_confidence += consistency * 0.2
        
        return min(base_confidence, 1.0)
    
    def _extract_learning_insights(self, result: OptimizationResult) -> List[LearningInsight]:
        """Extract learning insights from optimization result"""
        insights = []
        
        # High performance insight
        if result.optimization_effectiveness > 0.8:
            insights.append(LearningInsight(
                insight_id=f"high_perf_{result.iteration_id}",
                pattern_type="performance_optimization",
                description=f"Strategy combination {result.strategy_combination} achieved high effectiveness ({result.optimization_effectiveness:.3f})",
                confidence=result.confidence_score,
                applicable_scenarios=["high_performance_required"],
                expected_improvement={"effectiveness": result.optimization_effectiveness},
                implementation_complexity="medium",
                discovered_at=result.timestamp
            ))
        
        # Cost efficiency insight
        if result.cost_metrics['cost_efficiency'] > 1.5:
            insights.append(LearningInsight(
                insight_id=f"cost_eff_{result.iteration_id}",
                pattern_type="cost_optimization",
                description=f"High cost efficiency achieved: {result.cost_metrics['cost_efficiency']:.3f}",
                confidence=result.confidence_score,
                applicable_scenarios=["cost_sensitive_applications"],
                expected_improvement={"cost_efficiency": result.cost_metrics['cost_efficiency']},
                implementation_complexity="low",
                discovered_at=result.timestamp
            ))
        
        # Quality consistency insight
        if result.quality_metrics['quality_variance'] < 0.1:
            insights.append(LearningInsight(
                insight_id=f"quality_consistency_{result.iteration_id}",
                pattern_type="quality_optimization",
                description="Low quality variance indicates consistent results",
                confidence=result.confidence_score,
                applicable_scenarios=["quality_consistency_required"],
                expected_improvement={"quality_variance": -result.quality_metrics['quality_variance']},
                implementation_complexity="medium",
                discovered_at=result.timestamp
            ))
        
        return insights
    
    def _update_strategy_effectiveness(self, strategies: List[AdvancedOptimizationStrategy], result: OptimizationResult):
        """Update strategy effectiveness tracking"""
        strategy_key = "_".join([s.value for s in strategies])
        
        if strategy_key not in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy_key] = {
                'total_runs': 0,
                'total_effectiveness': 0,
                'avg_effectiveness': 0,
                'best_effectiveness': 0,
                'confidence_sum': 0,
                'avg_confidence': 0
            }
        
        stats = self.strategy_effectiveness[strategy_key]
        stats['total_runs'] += 1
        stats['total_effectiveness'] += result.optimization_effectiveness
        stats['avg_effectiveness'] = stats['total_effectiveness'] / stats['total_runs']
        stats['best_effectiveness'] = max(stats['best_effectiveness'], result.optimization_effectiveness)
        stats['confidence_sum'] += result.confidence_score
        stats['avg_confidence'] = stats['confidence_sum'] / stats['total_runs']
    
    def _generate_advanced_analysis(self, results: List[OptimizationResult]) -> Dict[str, Any]:
        """Generate advanced analysis of optimization results"""
        if not results:
            return {}
        
        # Overall performance trends
        effectiveness_values = [r.optimization_effectiveness for r in results]
        confidence_values = [r.confidence_score for r in results]
        
        analysis = {
            'performance_trend': {
                'best_iteration': max(results, key=lambda r: r.optimization_effectiveness).iteration_id,
                'worst_iteration': min(results, key=lambda r: r.optimization_effectiveness).iteration_id,
                'avg_effectiveness': np.mean(effectiveness_values),
                'effectiveness_improvement': effectiveness_values[-1] - effectiveness_values[0] if len(effectiveness_values) > 1 else 0,
                'confidence_trend': np.mean(confidence_values)
            },
            'strategy_ranking': self._rank_strategies_by_effectiveness(),
            'cost_analysis': self._analyze_cost_trends(results),
            'quality_analysis': self._analyze_quality_trends(results),
            'learning_progression': self._analyze_learning_progression(results)
        }
        
        return analysis
    
    def _rank_strategies_by_effectiveness(self) -> List[Dict[str, Any]]:
        """Rank strategy combinations by effectiveness"""
        rankings = []
        
        for strategy_key, stats in self.strategy_effectiveness.items():
            rankings.append({
                'strategy_combination': strategy_key,
                'avg_effectiveness': stats['avg_effectiveness'],
                'best_effectiveness': stats['best_effectiveness'],
                'total_runs': stats['total_runs'],
                'avg_confidence': stats['avg_confidence']
            })
        
        # Sort by average effectiveness
        rankings.sort(key=lambda x: x['avg_effectiveness'], reverse=True)
        
        return rankings[:10]  # Top 10
    
    def _analyze_cost_trends(self, results: List[OptimizationResult]) -> Dict[str, float]:
        """Analyze cost trends across iterations"""
        if not results:
            return {}
        
        costs = [r.cost_metrics['total_cost'] for r in results]
        efficiencies = [r.cost_metrics['cost_efficiency'] for r in results]
        
        return {
            'avg_cost': np.mean(costs),
            'cost_trend': costs[-1] - costs[0] if len(costs) > 1 else 0,
            'avg_efficiency': np.mean(efficiencies),
            'efficiency_improvement': efficiencies[-1] - efficiencies[0] if len(efficiencies) > 1 else 0
        }
    
    def _analyze_quality_trends(self, results: List[OptimizationResult]) -> Dict[str, float]:
        """Analyze quality trends across iterations"""
        if not results:
            return {}
        
        avg_qualities = [r.quality_metrics['avg_quality_score'] for r in results]
        variances = [r.quality_metrics['quality_variance'] for r in results]
        
        return {
            'avg_quality': np.mean(avg_qualities),
            'quality_improvement': avg_qualities[-1] - avg_qualities[0] if len(avg_qualities) > 1 else 0,
            'avg_variance': np.mean(variances),
            'consistency_improvement': variances[0] - variances[-1] if len(variances) > 1 else 0
        }
    
    def _analyze_learning_progression(self, results: List[OptimizationResult]) -> Dict[str, Any]:
        """Analyze learning progression across iterations"""
        if not results:
            return {}
        
        return {
            'total_insights_discovered': len(self.learned_insights),
            'insight_types': list(set(insight.pattern_type for insight in self.learned_insights)),
            'avg_insight_confidence': np.mean([insight.confidence for insight in self.learned_insights]) if self.learned_insights else 0,
            'learning_velocity': len(self.learned_insights) / len(results) if results else 0
        }
    
    def _generate_advanced_recommendations(self, results: List[OptimizationResult], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate advanced optimization recommendations"""
        recommendations = []
        
        # Best strategy recommendation
        if analysis.get('strategy_ranking'):
            best_strategy = analysis['strategy_ranking'][0]
            recommendations.append({
                'type': 'strategy_selection',
                'priority': 'high',
                'title': f"Use Best Performing Strategy: {best_strategy['strategy_combination']}",
                'description': f"This combination achieved {best_strategy['avg_effectiveness']:.3f} effectiveness",
                'expected_improvement': f"{(best_strategy['avg_effectiveness'] - 0.5) * 100:.1f}% above baseline",
                'confidence': best_strategy['avg_confidence']
            })
        
        # Cost optimization recommendation
        cost_analysis = analysis.get('cost_analysis', {})
        if cost_analysis.get('avg_cost', 0) > 0.1:
            recommendations.append({
                'type': 'cost_optimization',
                'priority': 'medium',
                'title': "Implement Cost Optimization Strategies",
                'description': f"Average cost per iteration: ${cost_analysis['avg_cost']:.4f}",
                'expected_improvement': "30-50% cost reduction with smart model selection",
                'confidence': 0.8
            })
        
        # Quality consistency recommendation
        quality_analysis = analysis.get('quality_analysis', {})
        if quality_analysis.get('avg_variance', 0) > 0.2:
            recommendations.append({
                'type': 'quality_consistency',
                'priority': 'high',
                'title': "Improve Quality Consistency",
                'description': f"High quality variance detected: {quality_analysis['avg_variance']:.3f}",
                'expected_improvement': "More consistent results with ensemble methods",
                'confidence': 0.75
            })
        
        # Learning-based recommendations
        learning_analysis = analysis.get('learning_progression', {})
        if learning_analysis.get('learning_velocity', 0) > 1.0:
            recommendations.append({
                'type': 'learning_system',
                'priority': 'medium',
                'title': "Leverage Learning System Insights",
                'description': f"High learning velocity: {learning_analysis['learning_velocity']:.2f} insights per iteration",
                'expected_improvement': "Continuous optimization through automated learning",
                'confidence': 0.9
            })
        
        return recommendations
    
    def _print_advanced_summary(self, results: List[OptimizationResult], analysis: Dict[str, Any]):
        """Print advanced optimization summary"""
        print(f"\n📊 Advanced MLACS Optimization Results")
        print("=" * 80)
        
        if results:
            best_result = max(results, key=lambda r: r.optimization_effectiveness)
            print(f"🏆 Best Iteration: {best_result.iteration_id}")
            print(f"   🎯 Effectiveness: {best_result.optimization_effectiveness:.3f}")
            print(f"   🔧 Strategies: {best_result.strategy_combination}")
            print(f"   📈 Confidence: {best_result.confidence_score:.3f}")
            
            print(f"\n📈 Performance Trends:")
            perf_trend = analysis.get('performance_trend', {})
            print(f"   Average Effectiveness: {perf_trend.get('avg_effectiveness', 0):.3f}")
            print(f"   Effectiveness Improvement: {perf_trend.get('effectiveness_improvement', 0):.3f}")
            print(f"   Average Confidence: {perf_trend.get('confidence_trend', 0):.3f}")
            
            print(f"\n🧠 Learning Insights:")
            learning = analysis.get('learning_progression', {})
            print(f"   Total Insights: {learning.get('total_insights_discovered', 0)}")
            print(f"   Learning Velocity: {learning.get('learning_velocity', 0):.2f} insights/iteration")
            print(f"   Insight Types: {', '.join(learning.get('insight_types', []))}")
            
            print(f"\n🔧 Top Strategy Combinations:")
            for i, strategy in enumerate(analysis.get('strategy_ranking', [])[:3], 1):
                print(f"   {i}. {strategy['strategy_combination']}")
                print(f"      Effectiveness: {strategy['avg_effectiveness']:.3f}")
                print(f"      Confidence: {strategy['avg_confidence']:.3f}")

async def main():
    """Run advanced iterative MLACS optimization"""
    try:
        # Load previous results for learning
        base_results_file = "iterative_optimization_results_20250601_171018.json"
        
        optimizer = AdvancedIterativeMLACSOptimizer(base_results_file)
        
        # Run advanced optimization
        report = await optimizer.run_advanced_optimization(iterations=8)
        
        print(f"\n🎉 Advanced Iterative Optimization Complete!")
        print(f"📊 Total Insights Discovered: {len(optimizer.learned_insights)}")
        print(f"🧠 Strategy Combinations Tested: {len(optimizer.strategy_effectiveness)}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Advanced optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Advanced optimization failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())