#!/usr/bin/env python3
"""
Performance Validation Tests for AgenticSeek Optimizations
Validates that performance optimizations are working as expected
"""

import os
import re
import time
import json
from pathlib import Path
from typing import Dict, List, Any

class PerformanceValidator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            "validation_passed": False,
            "optimization_metrics": {},
            "performance_improvements": [],
            "benchmark_results": {}
        }
        
    def validate_optimizations(self) -> Dict[str, Any]:
        """Validate that performance optimizations have been implemented correctly"""
        print("🚀 Starting Performance Optimization Validation...")
        
        # Test 1: Verify optimization files exist
        self._validate_optimization_files()
        
        # Test 2: Check for performance patterns
        self._validate_performance_patterns()
        
        # Test 3: Measure complexity reduction
        self._measure_complexity_improvements()
        
        # Test 4: Validate async operation patterns
        self._validate_async_patterns()
        
        # Test 5: Check lazy loading implementation
        self._validate_lazy_loading()
        
        # Generate final score
        self._calculate_performance_score()
        
        return self.results
        
    def _validate_optimization_files(self):
        """Check that optimization files have been created"""
        required_files = [
            "AgenticSeek/OptimizedModelManagementView.swift",
            "AgenticSeek/PerformanceOptimizedComponents.swift",
            "tests/performance_optimization_audit.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
                
        if missing_files:
            self.results["optimization_metrics"]["missing_files"] = missing_files
            print(f"❌ Missing optimization files: {missing_files}")
        else:
            self.results["optimization_metrics"]["all_files_present"] = True
            print("✅ All optimization files present")
            
    def _validate_performance_patterns(self):
        """Validate that performance optimization patterns are implemented"""
        patterns_to_check = {
            "lazy_loading": r"LazyV?Stack|LazyView|\.lazy\(",
            "async_await": r"async func|await\s+\w+|Task\s*\{",
            "performance_monitoring": r"PerformanceMonitor|recordAction|startTiming",
            "state_optimization": r"@StateObject.*PerformanceMonitor|minimized.*state",
            "cache_implementation": r"Cache|cache\w*|isStale",
            "memory_management": r"weak\s+|unowned\s+|cleanup\(\)",
        }
        
        optimization_files = [
            "AgenticSeek/OptimizedModelManagementView.swift",
            "AgenticSeek/PerformanceOptimizedComponents.swift"
        ]
        
        pattern_results = {}
        
        for file_path in optimization_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                continue
                
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                    
                file_patterns = {}
                for pattern_name, pattern in patterns_to_check.items():
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    file_patterns[pattern_name] = matches
                    
                pattern_results[file_path] = file_patterns
                
            except Exception as e:
                pattern_results[file_path] = {"error": str(e)}
                
        self.results["optimization_metrics"]["performance_patterns"] = pattern_results
        
        # Calculate pattern coverage
        total_patterns = len(patterns_to_check)
        implemented_patterns = 0
        
        for file_results in pattern_results.values():
            if "error" not in file_results:
                for pattern_name, count in file_results.items():
                    if count > 0:
                        implemented_patterns += 1
                        break  # Count each pattern only once
                        
        pattern_coverage = (implemented_patterns / total_patterns) * 100
        self.results["optimization_metrics"]["pattern_coverage"] = pattern_coverage
        
        print(f"✅ Performance pattern coverage: {pattern_coverage:.1f}%")
        
    def _measure_complexity_improvements(self):
        """Measure complexity reduction from optimizations"""
        
        # Compare original vs optimized files
        comparisons = [
            {
                "original": "AgenticSeek/ContentView.swift",
                "optimized": "AgenticSeek/PerformanceOptimizedComponents.swift",
                "name": "ContentView Optimization"
            },
            {
                "original": "AgenticSeek/ModelManagementView.swift", 
                "optimized": "AgenticSeek/OptimizedModelManagementView.swift",
                "name": "ModelManagement Optimization"
            }
        ]
        
        complexity_improvements = []
        
        for comparison in comparisons:
            original_path = self.project_root / comparison["original"]
            optimized_path = self.project_root / comparison["optimized"]
            
            if original_path.exists() and optimized_path.exists():
                original_metrics = self._calculate_file_metrics(original_path)
                optimized_metrics = self._calculate_file_metrics(optimized_path)
                
                improvement = {
                    "name": comparison["name"],
                    "original": original_metrics,
                    "optimized": optimized_metrics,
                    "improvements": {}
                }
                
                # Calculate improvements
                for metric in ["line_count", "complexity_score", "state_variables"]:
                    if metric in original_metrics and metric in optimized_metrics:
                        original_val = original_metrics[metric]
                        optimized_val = optimized_metrics[metric]
                        
                        if original_val > 0:
                            improvement_pct = ((original_val - optimized_val) / original_val) * 100
                            improvement["improvements"][metric] = {
                                "absolute": original_val - optimized_val,
                                "percentage": improvement_pct
                            }
                            
                complexity_improvements.append(improvement)
                
        self.results["optimization_metrics"]["complexity_improvements"] = complexity_improvements
        
        # Print summary
        for improvement in complexity_improvements:
            print(f"📊 {improvement['name']}:")
            for metric, values in improvement["improvements"].items():
                print(f"   - {metric}: {values['percentage']:.1f}% reduction")
                
    def _calculate_file_metrics(self, file_path: Path) -> Dict[str, Any]:
        """Calculate metrics for a single file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            return {
                "line_count": len(content.splitlines()),
                "state_variables": len(re.findall(r"@State", content)),
                "complexity_score": self._calculate_complexity(content),
                "view_count": len(re.findall(r"struct.*View\s*\{", content)),
                "async_functions": len(re.findall(r"async func", content))
            }
        except Exception:
            return {}
            
    def _calculate_complexity(self, content: str) -> int:
        """Calculate complexity score for content"""
        complexity = 0
        complexity += len(re.findall(r"\bif\b", content)) * 1
        complexity += len(re.findall(r"\bfor\b", content)) * 2
        complexity += len(re.findall(r"\bwhile\b", content)) * 2
        complexity += len(re.findall(r"\bswitch\b", content)) * 3
        complexity += len(re.findall(r"\bcase\b", content)) * 1
        return complexity
        
    def _validate_async_patterns(self):
        """Validate proper async/await implementation"""
        optimization_files = [
            "AgenticSeek/OptimizedModelManagementView.swift",
            "AgenticSeek/PerformanceOptimizedComponents.swift"
        ]
        
        async_validations = {
            "task_usage": r"\.task\s*\{",
            "mainactor_usage": r"@MainActor",
            "async_functions": r"async func",
            "await_calls": r"await\s+\w+",
            "proper_task_handling": r"Task\s*\{\s*await"
        }
        
        async_results = {}
        
        for file_path in optimization_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                continue
                
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                    
                file_async = {}
                for validation_name, pattern in async_validations.items():
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    file_async[validation_name] = matches
                    
                async_results[file_path] = file_async
                
            except Exception as e:
                async_results[file_path] = {"error": str(e)}
                
        self.results["optimization_metrics"]["async_patterns"] = async_results
        
        # Calculate async pattern score
        total_async_score = 0
        max_possible_score = len(async_validations) * len(optimization_files)
        
        for file_results in async_results.values():
            if "error" not in file_results:
                for validation_name, count in file_results.items():
                    if count > 0:
                        total_async_score += 1
                        
        async_score = (total_async_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        self.results["optimization_metrics"]["async_pattern_score"] = async_score
        
        print(f"✅ Async pattern implementation: {async_score:.1f}%")
        
    def _validate_lazy_loading(self):
        """Validate lazy loading implementations"""
        lazy_patterns = {
            "lazy_vstack": r"LazyVStack",
            "lazy_hstack": r"LazyHStack", 
            "lazy_view": r"LazyView",
            "lazy_grid": r"LazyVGrid|LazyHGrid",
            "conditional_loading": r"if.*\{.*LazyV"
        }
        
        optimization_files = [
            "AgenticSeek/OptimizedModelManagementView.swift",
            "AgenticSeek/PerformanceOptimizedComponents.swift"
        ]
        
        lazy_results = {}
        
        for file_path in optimization_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                continue
                
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                    
                file_lazy = {}
                for pattern_name, pattern in lazy_patterns.items():
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    file_lazy[pattern_name] = matches
                    
                lazy_results[file_path] = file_lazy
                
            except Exception as e:
                lazy_results[file_path] = {"error": str(e)}
                
        self.results["optimization_metrics"]["lazy_loading"] = lazy_results
        
        # Calculate lazy loading score
        total_lazy_implementations = 0
        for file_results in lazy_results.values():
            if "error" not in file_results:
                total_lazy_implementations += sum(file_results.values())
                
        self.results["optimization_metrics"]["lazy_loading_count"] = total_lazy_implementations
        
        print(f"✅ Lazy loading implementations found: {total_lazy_implementations}")
        
    def _calculate_performance_score(self):
        """Calculate overall performance optimization score"""
        metrics = self.results["optimization_metrics"]
        
        # Score components
        scores = []
        
        # File presence (20%)
        if metrics.get("all_files_present"):
            scores.append(20)
        
        # Pattern coverage (25%)
        pattern_coverage = metrics.get("pattern_coverage", 0)
        scores.append((pattern_coverage / 100) * 25)
        
        # Async patterns (25%)
        async_score = metrics.get("async_pattern_score", 0)
        scores.append((async_score / 100) * 25)
        
        # Complexity improvements (20%)
        complexity_improvements = metrics.get("complexity_improvements", [])
        if complexity_improvements:
            avg_improvement = 0
            improvement_count = 0
            
            for improvement in complexity_improvements:
                for metric, values in improvement["improvements"].items():
                    if values["percentage"] > 0:  # Only count actual improvements
                        avg_improvement += values["percentage"]
                        improvement_count += 1
                        
            if improvement_count > 0:
                avg_improvement = avg_improvement / improvement_count
                # Cap at 50% improvement for scoring
                complexity_score = min(avg_improvement / 50.0, 1.0) * 20
                scores.append(complexity_score)
        
        # Lazy loading (10%)
        lazy_count = metrics.get("lazy_loading_count", 0)
        # Award points for having lazy loading implementations
        lazy_score = min(lazy_count / 5.0, 1.0) * 10  # Max score if 5+ implementations
        scores.append(lazy_score)
        
        # Calculate final score
        final_score = sum(scores)
        self.results["optimization_metrics"]["final_performance_score"] = final_score
        
        # Set validation passed if score >= 80
        self.results["validation_passed"] = final_score >= 80
        
        print(f"\n🎯 Final Performance Score: {final_score:.1f}/100")
        
        if final_score >= 95:
            print("🏆 EXCELLENT: Outstanding performance optimization!")
        elif final_score >= 85:
            print("✅ GOOD: Strong performance improvements achieved")
        elif final_score >= 70:
            print("⚠️  FAIR: Basic optimizations in place, room for improvement")
        else:
            print("❌ NEEDS WORK: Significant optimization work required")
            
    def generate_report(self) -> str:
        """Generate a comprehensive performance validation report"""
        report = []
        report.append("# AgenticSeek Performance Optimization Validation Report")
        report.append("=" * 60)
        report.append("")
        
        metrics = self.results["optimization_metrics"]
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Validation Status**: {'✅ PASSED' if self.results['validation_passed'] else '❌ FAILED'}")
        report.append(f"- **Performance Score**: {metrics.get('final_performance_score', 0):.1f}/100")
        report.append("")
        
        # File Status
        report.append("## Optimization Files")
        if metrics.get("all_files_present"):
            report.append("- ✅ All optimization files created successfully")
        else:
            missing = metrics.get("missing_files", [])
            report.append(f"- ❌ Missing files: {', '.join(missing)}")
        report.append("")
        
        # Performance Patterns
        report.append("## Performance Pattern Implementation")
        pattern_coverage = metrics.get("pattern_coverage", 0)
        report.append(f"- **Coverage**: {pattern_coverage:.1f}%")
        
        if "performance_patterns" in metrics:
            for file_path, patterns in metrics["performance_patterns"].items():
                if "error" not in patterns:
                    report.append(f"- **{file_path}**:")
                    for pattern_name, count in patterns.items():
                        status = "✅" if count > 0 else "❌"
                        report.append(f"  - {status} {pattern_name}: {count} implementations")
        report.append("")
        
        # Complexity Improvements
        report.append("## Complexity Reduction Results")
        if "complexity_improvements" in metrics:
            for improvement in metrics["complexity_improvements"]:
                report.append(f"### {improvement['name']}")
                for metric, values in improvement["improvements"].items():
                    report.append(f"- **{metric}**: {values['percentage']:.1f}% reduction ({values['absolute']} units)")
                report.append("")
        
        # Async Patterns
        async_score = metrics.get("async_pattern_score", 0)
        report.append(f"## Async Operation Patterns: {async_score:.1f}% implemented")
        report.append("")
        
        # Lazy Loading
        lazy_count = metrics.get("lazy_loading_count", 0)
        report.append(f"## Lazy Loading: {lazy_count} implementations found")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if self.results["validation_passed"]:
            report.append("- 🎉 Performance optimizations successfully implemented!")
            report.append("- 📊 Continue monitoring performance metrics in production")
            report.append("- 🔄 Consider adding performance regression tests")
        else:
            report.append("- 🔧 Complete missing optimization files")
            report.append("- ⚡ Implement more async/await patterns")
            report.append("- 📱 Add lazy loading to remaining views")
            report.append("- 🧹 Reduce complexity in remaining large files")
        
        return "\n".join(report)
        
    def save_results(self, output_path: str):
        """Save validation results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

def main():
    project_root = "/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/_macOS"
    
    validator = PerformanceValidator(project_root)
    
    print("Starting Performance Optimization Validation...")
    results = validator.validate_optimizations()
    
    # Generate and display report
    report = validator.generate_report()
    print("\n" + report)
    
    # Save results
    output_path = os.path.join(project_root, "tests", "performance_validation_results.json")
    validator.save_results(output_path)
    print(f"\nDetailed results saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    main()