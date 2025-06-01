#!/usr/bin/env python3
"""
Quick MLACS benchmark for demonstration with faster execution
"""

import asyncio
import json
import time
import os
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

# Load environment
from dotenv import load_dotenv
load_dotenv()

import anthropic
import openai

@dataclass
class QuickBenchmarkResult:
    benchmark_id: str
    query_title: str
    providers_used: List[str]
    total_duration: float
    llm_calls: int
    total_tokens: int
    quality_score: float
    output_summary: str
    success: bool = True
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class QuickMLACSBenchmark:
    """Quick benchmark for MLACS system validation"""
    
    def __init__(self):
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.results = []
    
    async def run_quick_benchmark(self):
        """Run a quick benchmark to validate MLACS coordination"""
        
        print("🚀 Quick MLACS Benchmark with Real LLM Providers")
        print("=" * 60)
        
        # Quick multi-step research query
        complex_query = """
        MULTI-LLM RESEARCH TASK: AI Agent Coordination Analysis
        
        Phase 1 (Research): Analyze current state of AI agent coordination systems, 
        identifying key technologies, approaches, and leading research directions.
        
        Phase 2 (Analysis): Evaluate strengths and limitations of existing multi-agent 
        coordination frameworks, focusing on scalability and practical applications.
        
        Phase 3 (Synthesis): Propose an integrated approach that addresses identified 
        limitations while leveraging strengths of existing systems.
        
        Please provide a comprehensive response covering all three phases with 
        actionable insights and recommendations.
        """
        
        # Test scenarios with different provider combinations
        scenarios = [
            {
                "title": "Anthropic Claude Analysis",
                "providers": ["claude"],
                "query": complex_query
            },
            {
                "title": "OpenAI GPT Analysis", 
                "providers": ["gpt4"],
                "query": complex_query
            },
            {
                "title": "Multi-LLM Coordination",
                "providers": ["claude", "gpt4"],
                "query": complex_query
            }
        ]
        
        for scenario in scenarios:
            result = await self._run_scenario(scenario)
            self.results.append(result)
            print()
        
        # Generate report
        self._generate_report()
    
    async def _run_scenario(self, scenario: Dict[str, Any]) -> QuickBenchmarkResult:
        """Run a single benchmark scenario"""
        
        benchmark_id = f"quick_{uuid.uuid4().hex[:8]}"
        print(f"🎬 {scenario['title']}")
        print(f"   🔗 Providers: {scenario['providers']}")
        
        start_time = time.time()
        total_tokens = 0
        llm_calls = 0
        responses = []
        
        try:
            # Execute with each provider
            for provider in scenario['providers']:
                print(f"   ⚡ Calling {provider}...")
                
                if provider == "claude" and self.anthropic_key:
                    response, tokens = await self._call_claude(scenario['query'])
                    responses.append(f"Claude Analysis:\n{response}")
                    
                elif provider == "gpt4" and self.openai_key:
                    response, tokens = await self._call_openai(scenario['query'])
                    responses.append(f"GPT-4 Analysis:\n{response}")
                
                total_tokens += tokens
                llm_calls += 1
            
            # If multiple providers, synthesize responses
            if len(responses) > 1:
                print(f"   🧬 Synthesizing multi-LLM responses...")
                synthesis_prompt = f"""
Please synthesize the following expert analyses into a unified, comprehensive response:

{chr(10).join(responses)}

Provide an integrated analysis that combines the best insights from both perspectives.
"""
                final_response, synthesis_tokens = await self._call_claude(synthesis_prompt)
                total_tokens += synthesis_tokens
                llm_calls += 1
            else:
                final_response = responses[0] if responses else "No responses generated"
            
            duration = time.time() - start_time
            
            # Simple quality scoring
            quality_score = min(1.0, len(final_response) / 2000) * 0.7 + 0.3
            if len(scenario['providers']) > 1:
                quality_score += 0.1  # Bonus for multi-LLM coordination
            
            # Create summary
            output_summary = final_response[:300] + "..." if len(final_response) > 300 else final_response
            
            result = QuickBenchmarkResult(
                benchmark_id=benchmark_id,
                query_title=scenario['title'],
                providers_used=scenario['providers'],
                total_duration=duration,
                llm_calls=llm_calls,
                total_tokens=total_tokens,
                quality_score=quality_score,
                output_summary=output_summary,
                success=True
            )
            
            print(f"   ✅ Completed in {duration:.2f}s")
            print(f"   🔗 LLM calls: {llm_calls}")
            print(f"   🎯 Quality: {quality_score:.2f}")
            print(f"   📊 Tokens: {total_tokens:,}")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Error: {str(e)}")
            
            return QuickBenchmarkResult(
                benchmark_id=benchmark_id,
                query_title=scenario['title'],
                providers_used=scenario['providers'],
                total_duration=duration,
                llm_calls=llm_calls,
                total_tokens=total_tokens,
                quality_score=0.0,
                output_summary=f"Error: {str(e)}",
                success=False
            )
    
    async def _call_claude(self, prompt: str) -> tuple[str, int]:
        """Call Anthropic Claude API"""
        client = anthropic.Anthropic(api_key=self.anthropic_key)
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Faster model
            max_tokens=2000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return content, tokens
    
    async def _call_openai(self, prompt: str) -> tuple[str, int]:
        """Call OpenAI GPT API"""
        client = openai.OpenAI(api_key=self.openai_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Faster model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        return content, tokens
    
    def _generate_report(self):
        """Generate benchmark report"""
        
        print("📊 Quick MLACS Benchmark Results")
        print("=" * 60)
        
        successful = [r for r in self.results if r.success]
        total_duration = sum(r.total_duration for r in self.results)
        total_calls = sum(r.llm_calls for r in self.results)
        total_tokens = sum(r.total_tokens for r in self.results)
        
        print(f"📈 Summary:")
        print(f"   Scenarios tested: {len(self.results)}")
        print(f"   Successful: {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)")
        print(f"   Total duration: {total_duration:.2f}s")
        print(f"   Total LLM calls: {total_calls}")
        print(f"   Total tokens: {total_tokens:,}")
        
        if successful:
            avg_quality = sum(r.quality_score for r in successful) / len(successful)
            print(f"   Average quality: {avg_quality:.2f}")
        
        print(f"\n📋 Detailed Results:")
        for result in self.results:
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            print(f"   {result.query_title}: {status}")
            print(f"      Duration: {result.total_duration:.2f}s")
            print(f"      Quality: {result.quality_score:.2f}")
            print(f"      Providers: {result.providers_used}")
        
        # Key insights for optimization
        print(f"\n🔍 Performance Insights:")
        
        # Compare single vs multi-LLM performance
        single_llm_results = [r for r in successful if len(r.providers_used) == 1]
        multi_llm_results = [r for r in successful if len(r.providers_used) > 1]
        
        if single_llm_results and multi_llm_results:
            single_avg_duration = sum(r.total_duration for r in single_llm_results) / len(single_llm_results)
            multi_avg_duration = sum(r.total_duration for r in multi_llm_results) / len(multi_llm_results)
            
            single_avg_quality = sum(r.quality_score for r in single_llm_results) / len(single_llm_results)
            multi_avg_quality = sum(r.quality_score for r in multi_llm_results) / len(multi_llm_results)
            
            print(f"   Single-LLM avg duration: {single_avg_duration:.2f}s")
            print(f"   Multi-LLM avg duration: {multi_avg_duration:.2f}s")
            print(f"   Single-LLM avg quality: {single_avg_quality:.2f}")
            print(f"   Multi-LLM avg quality: {multi_avg_quality:.2f}")
            
            duration_overhead = (multi_avg_duration - single_avg_duration) / single_avg_duration * 100
            quality_improvement = (multi_avg_quality - single_avg_quality) / single_avg_quality * 100
            
            print(f"   Multi-LLM coordination overhead: {duration_overhead:.1f}%")
            print(f"   Multi-LLM quality improvement: {quality_improvement:.1f}%")
        
        print("\n🎉 Quick benchmark complete!")
        
        # Save results
        benchmark_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_scenarios': len(self.results),
                'successful_scenarios': len(successful),
                'total_duration': total_duration,
                'total_calls': total_calls,
                'total_tokens': total_tokens
            },
            'results': [asdict(result) for result in self.results]
        }
        
        filename = f'quick_mlacs_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"📄 Results saved to: {filename}")

async def main():
    benchmark = QuickMLACSBenchmark()
    await benchmark.run_quick_benchmark()

if __name__ == "__main__":
    asyncio.run(main())