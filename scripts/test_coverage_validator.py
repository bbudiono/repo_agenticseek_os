#!/usr/bin/env python3
"""
Test Coverage Threshold Validator for AgenticSeek
================================================

* Purpose: Validates that test coverage meets minimum thresholds before commits
* Usage: Called automatically by pre-commit hooks
* Exit Codes: 0 = success, 1 = coverage below threshold
"""

import sys
import json
import subprocess
from pathlib import Path

def run_test_suite() -> dict:
    """Run comprehensive test suite and get results"""
    try:
        result = subprocess.run(
            [sys.executable, "comprehensive_test_suite.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Try to load test_report.json
        report_file = Path("test_report.json")
        if report_file.exists():
            with open(report_file, 'r') as f:
                return json.load(f)
        else:
            return {"summary": {"success_rate": 0}}
    
    except Exception as e:
        print(f"❌ Failed to run test suite: {e}")
        return {"summary": {"success_rate": 0}}

def check_coverage_threshold(test_results: dict, min_threshold: float = 90.0) -> bool:
    """Check if test coverage meets minimum threshold"""
    success_rate = test_results.get("summary", {}).get("success_rate", 0)
    return success_rate >= min_threshold

def main():
    """Main validation function"""
    print("🔍 Running Test Coverage Validation...")
    
    # Configuration
    MIN_COVERAGE_THRESHOLD = 90.0  # 90% minimum success rate
    
    # Run tests
    test_results = run_test_suite()
    success_rate = test_results.get("summary", {}).get("success_rate", 0)
    
    print(f"📊 Current test success rate: {success_rate:.1f}%")
    print(f"🎯 Required threshold: {MIN_COVERAGE_THRESHOLD}%")
    
    if check_coverage_threshold(test_results, MIN_COVERAGE_THRESHOLD):
        print("✅ Test Coverage Validation PASSED")
        
        # Print quick summary
        summary = test_results.get("summary", {})
        print(f"   📈 Tests passed: {summary.get('passed', 0)}")
        print(f"   ❌ Tests failed: {summary.get('failed', 0)}")
        print(f"   💥 Tests errored: {summary.get('errors', 0)}")
        
        sys.exit(0)
    else:
        print("❌ Test Coverage Validation FAILED")
        print(f"   Current coverage ({success_rate:.1f}%) below threshold ({MIN_COVERAGE_THRESHOLD}%)")
        
        # Show failed tests
        if "detailed_results" in test_results:
            failed_tests = [
                test for test in test_results["detailed_results"]
                if test.get("status") in ["FAILED", "ERROR"]
            ]
            
            if failed_tests:
                print(f"\n🔍 Failed/Error Tests ({len(failed_tests)}):")
                for test in failed_tests[:5]:  # Show first 5
                    category = test.get("category", "Unknown")
                    name = test.get("test", "Unknown")
                    status = test.get("status", "Unknown")
                    error = test.get("error", "")
                    
                    print(f"   ❌ {category}/{name}: {status}")
                    if error:
                        print(f"      Error: {error[:100]}...")
                
                if len(failed_tests) > 5:
                    print(f"   ... and {len(failed_tests) - 5} more")
        
        print(f"\n💡 To fix:")
        print(f"   1. Run: python comprehensive_test_suite.py")
        print(f"   2. Fix failing tests")
        print(f"   3. Ensure success rate >= {MIN_COVERAGE_THRESHOLD}%")
        print(f"   4. Retry commit")
        
        sys.exit(1)

if __name__ == "__main__":
    main()