#!/usr/bin/env python3
"""
Performance Optimization Audit for AgenticSeek macOS App
Analyzes Swift files for performance issues and optimization opportunities
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any

class PerformanceAuditor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            "performance_issues": [],
            "optimization_opportunities": [],
            "code_metrics": {},
            "recommendations": []
        }
        
        # Performance anti-patterns to detect
        self.performance_patterns = {
            "heavy_operations_in_body": {
                "pattern": r"var body.*?\{.*?(\w+\.load\w*\(|\w+\.fetch\w*\(|\w+\.process\w*\()",
                "severity": "high",
                "description": "Heavy operations detected in View body (should be in .onAppear or async)"
            },
            "excessive_state_variables": {
                "pattern": r"@State.*?private.*?var",
                "severity": "medium", 
                "description": "Many @State variables detected - consider using ObservableObject"
            },
            "missing_lazy_loading": {
                "pattern": r"List.*?\{.*?ForEach.*?\{.*?\}.*?\}",
                "severity": "medium",
                "description": "List without lazy loading detected"
            },
            "synchronous_network_calls": {
                "pattern": r"URLSession\.shared\.data\(from:",
                "severity": "high",
                "description": "Synchronous network calls detected - use async/await"
            },
            "missing_memory_management": {
                "pattern": r"@StateObject.*?=.*?(?!.*weak)",
                "severity": "medium",
                "description": "StateObject without weak references may cause memory leaks"
            },
            "inefficient_recomposition": {
                "pattern": r"\.onChange\(of:.*?\{.*?@State.*?\}",
                "severity": "medium",
                "description": "onChange triggering state updates may cause excessive recomposition"
            }
        }
        
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Swift file for performance issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            file_results = {
                "file": str(file_path.relative_to(self.project_root)),
                "line_count": len(content.splitlines()),
                "issues": [],
                "metrics": {}
            }
            
            # Count various performance metrics
            file_results["metrics"] = {
                "state_variables": len(re.findall(r"@State", content)),
                "observed_objects": len(re.findall(r"@ObservedObject", content)),
                "state_objects": len(re.findall(r"@StateObject", content)),
                "view_builders": len(re.findall(r"@ViewBuilder", content)),
                "async_functions": len(re.findall(r"async func", content)),
                "task_calls": len(re.findall(r"Task\s*\{", content)),
                "onappear_calls": len(re.findall(r"\.onAppear", content)),
                "onchange_calls": len(re.findall(r"\.onChange", content)),
                "complexity_score": self._calculate_complexity_score(content)
            }
            
            # Check for performance anti-patterns
            for pattern_name, pattern_info in self.performance_patterns.items():
                matches = re.findall(pattern_info["pattern"], content, re.DOTALL | re.IGNORECASE)
                if matches:
                    file_results["issues"].append({
                        "type": pattern_name,
                        "severity": pattern_info["severity"],
                        "description": pattern_info["description"],
                        "count": len(matches),
                        "examples": matches[:3]  # First 3 examples
                    })
                    
            return file_results
            
        except Exception as e:
            return {
                "file": str(file_path.relative_to(self.project_root)),
                "error": str(e)
            }
    
    def _calculate_complexity_score(self, content: str) -> int:
        """Calculate a rough complexity score for the file"""
        complexity = 0
        
        # Add complexity for various constructs
        complexity += len(re.findall(r"\bif\b", content)) * 1
        complexity += len(re.findall(r"\bfor\b", content)) * 2
        complexity += len(re.findall(r"\bwhile\b", content)) * 2
        complexity += len(re.findall(r"\bswitch\b", content)) * 3
        complexity += len(re.findall(r"\bcase\b", content)) * 1
        complexity += len(re.findall(r"\.onChange\b", content)) * 2
        complexity += len(re.findall(r"Task\s*\{", content)) * 2
        
        return complexity
        
    def audit_project(self) -> None:
        """Audit the entire project for performance issues"""
        swift_files = list(self.project_root.rglob("*.swift"))
        
        # Filter out test files and generated code
        swift_files = [f for f in swift_files if not any(
            excluded in str(f) for excluded in [
                "Tests", "test", "generated", "Generated", 
                "DerivedData", ".build", "Preview"
            ]
        )]
        
        print(f"Analyzing {len(swift_files)} Swift files...")
        
        total_metrics = {
            "total_files": len(swift_files),
            "total_lines": 0,
            "total_state_variables": 0,
            "total_observed_objects": 0,
            "total_issues": 0,
            "average_complexity": 0
        }
        
        for file_path in swift_files:
            file_result = self.analyze_file(file_path)
            
            if "error" not in file_result:
                self.results["performance_issues"].append(file_result)
                
                # Aggregate metrics
                metrics = file_result.get("metrics", {})
                total_metrics["total_lines"] += file_result.get("line_count", 0)
                total_metrics["total_state_variables"] += metrics.get("state_variables", 0)
                total_metrics["total_observed_objects"] += metrics.get("observed_objects", 0)
                total_metrics["total_issues"] += len(file_result.get("issues", []))
                
        if total_metrics["total_files"] > 0:
            total_metrics["average_complexity"] = sum(
                result.get("metrics", {}).get("complexity_score", 0) 
                for result in self.results["performance_issues"]
            ) / total_metrics["total_files"]
            
        self.results["code_metrics"] = total_metrics
        
    def generate_recommendations(self) -> None:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Analyze aggregated results
        high_complexity_files = [
            result for result in self.results["performance_issues"]
            if result.get("metrics", {}).get("complexity_score", 0) > 20
        ]
        
        if high_complexity_files:
            recommendations.append({
                "priority": "high",
                "category": "Code Complexity",
                "description": f"Refactor {len(high_complexity_files)} high-complexity files",
                "files": [f["file"] for f in high_complexity_files],
                "action": "Break down large view bodies, extract reusable components"
            })
            
        # Check for excessive state management
        state_heavy_files = [
            result for result in self.results["performance_issues"]
            if result.get("metrics", {}).get("state_variables", 0) > 5
        ]
        
        if state_heavy_files:
            recommendations.append({
                "priority": "medium",
                "category": "State Management",
                "description": f"Optimize state management in {len(state_heavy_files)} files",
                "files": [f["file"] for f in state_heavy_files],
                "action": "Consider using ObservableObject pattern or state consolidation"
            })
            
        # Check for missing async patterns
        async_opportunities = [
            result for result in self.results["performance_issues"]
            if any(issue["type"] in ["heavy_operations_in_body", "synchronous_network_calls"] 
                   for issue in result.get("issues", []))
        ]
        
        if async_opportunities:
            recommendations.append({
                "priority": "high",
                "category": "Async Operations", 
                "description": f"Implement async patterns in {len(async_opportunities)} files",
                "files": [f["file"] for f in async_opportunities],
                "action": "Move heavy operations to .onAppear or background tasks"
            })
            
        self.results["recommendations"] = recommendations
        
    def generate_report(self) -> str:
        """Generate a comprehensive performance audit report"""
        report = []
        report.append("# AgenticSeek Performance Optimization Audit")
        report.append("=" * 50)
        report.append("")
        
        # Summary metrics
        metrics = self.results["code_metrics"]
        report.append("## Summary Metrics")
        report.append(f"- Total files analyzed: {metrics['total_files']}")
        report.append(f"- Total lines of code: {metrics['total_lines']}")
        report.append(f"- Total performance issues: {metrics['total_issues']}")
        report.append(f"- Average complexity score: {metrics['average_complexity']:.1f}")
        report.append("")
        
        # Top issues by severity
        all_issues = []
        for file_result in self.results["performance_issues"]:
            for issue in file_result.get("issues", []):
                all_issues.append({
                    "file": file_result["file"],
                    "severity": issue["severity"],
                    "type": issue["type"],
                    "description": issue["description"],
                    "count": issue["count"]
                })
                
        high_issues = [i for i in all_issues if i["severity"] == "high"]
        medium_issues = [i for i in all_issues if i["severity"] == "medium"]
        
        report.append("## Critical Issues (High Priority)")
        if high_issues:
            for issue in high_issues[:10]:  # Top 10
                report.append(f"- **{issue['file']}**: {issue['description']} ({issue['count']} occurrences)")
        else:
            report.append("- No high-priority issues found")
        report.append("")
        
        # Recommendations
        report.append("## Optimization Recommendations")
        for rec in self.results["recommendations"]:
            report.append(f"### {rec['category']} ({rec['priority']} priority)")
            report.append(f"**Description:** {rec['description']}")
            report.append(f"**Action:** {rec['action']}")
            if rec["files"]:
                report.append(f"**Files:** {', '.join(rec['files'][:5])}")
            report.append("")
            
        return "\n".join(report)
        
    def save_results(self, output_path: str) -> None:
        """Save audit results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
def main():
    project_root = "/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/_macOS"
    
    auditor = PerformanceAuditor(project_root)
    
    print("Starting performance audit...")
    auditor.audit_project()
    auditor.generate_recommendations()
    
    # Generate and save report
    report = auditor.generate_report()
    print(report)
    
    # Save detailed results
    output_path = os.path.join(project_root, "tests", "performance_audit_results.json")
    auditor.save_results(output_path)
    print(f"\nDetailed results saved to: {output_path}")
    
    return auditor.results

if __name__ == "__main__":
    main()