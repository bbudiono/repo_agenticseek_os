#!/usr/bin/env python3
"""
TDD Test Runner - Automated test validation for AgenticSeek
Ensures tests pass before allowing code changes
"""

import subprocess
import sys
import json
import time
from pathlib import Path

class TDDTestRunner:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.test_results = {}
        
    def run_comprehensive_tests(self):
        """Run the comprehensive test suite"""
        print("🧪 Running Comprehensive Test Suite...")
        
        try:
            result = subprocess.run([
                sys.executable, 
                str(self.project_root / "comprehensive_test_suite.py")
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                # Parse test report
                report_file = self.project_root / "test_report.json"
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        self.test_results = json.load(f)
                    
                    success_rate = self.test_results.get('summary', {}).get('success_rate', 0)
                    print(f"✅ Tests completed - Success rate: {success_rate:.1f}%")
                    return success_rate >= 70  # Require 70% success rate
                else:
                    print("❌ Test report not found")
                    return False
            else:
                print(f"❌ Test execution failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"💥 Test runner error: {str(e)}")
            return False
    
    def run_production_build(self):
        """Verify production build works"""
        print("🏗️ Testing Production Build...")
        
        try:
            xcode_path = self.project_root / "_macOS"
            if not xcode_path.exists():
                print("⚠️ No macOS build directory found")
                return True  # Skip if not applicable
            
            result = subprocess.run([
                "xcodebuild", "-workspace", "AgenticSeek.xcworkspace",
                "-scheme", "AgenticSeek", "-configuration", "Release", 
                "build", "-quiet"
            ], capture_output=True, text=True, cwd=xcode_path)
            
            if result.returncode == 0:
                print("✅ Production build successful")
                return True
            else:
                print(f"❌ Production build failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"💥 Build test error: {str(e)}")
            return False
    
    def validate_sandbox_compliance(self):
        """Check sandbox environment compliance"""
        print("🏖️ Validating Sandbox Compliance...")
        
        sandbox_dir = self.project_root / "_macOS" / "AgenticSeek-Sandbox"
        if not sandbox_dir.exists():
            print("⚠️ Sandbox directory not found")
            return False
        
        # Check for mandatory sandbox comments
        swift_files = list(sandbox_dir.rglob("*.swift"))
        compliant_files = 0
        
        for file_path in swift_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if "// SANDBOX FILE:" in content:
                        compliant_files += 1
            except Exception:
                continue
        
        compliance_rate = (compliant_files / len(swift_files)) * 100 if swift_files else 0
        print(f"📊 Sandbox file compliance: {compliance_rate:.1f}%")
        
        return compliance_rate >= 90  # Require 90% compliance
    
    def generate_tdd_report(self):
        """Generate TDD compliance report"""
        print("\n" + "="*60)
        print("📋 TDD COMPLIANCE REPORT")
        print("="*60)
        
        if self.test_results:
            summary = self.test_results.get('summary', {})
            print(f"🎯 Test Success Rate: {summary.get('success_rate', 0):.1f}%")
            print(f"✅ Tests Passed: {summary.get('passed', 0)}")
            print(f"❌ Tests Failed: {summary.get('failed', 0)}")
            print(f"💥 Tests Errored: {summary.get('errors', 0)}")
            
            # Show category breakdown
            metrics = self.test_results.get('performance_metrics', {})
            print("\n📊 Category Performance:")
            for category, data in metrics.items():
                success_rate = data.get('success_rate', 0)
                print(f"  {category}: {success_rate:.1f}%")
        
        print("="*60)
    
    def run_full_tdd_validation(self):
        """Run complete TDD validation suite"""
        print("🚀 Starting Full TDD Validation...")
        print("="*60)
        
        start_time = time.time()
        
        # Run all validation steps
        tests_passed = self.run_comprehensive_tests()
        build_passed = self.run_production_build()
        sandbox_compliant = self.validate_sandbox_compliance()
        
        duration = time.time() - start_time
        
        # Generate report
        self.generate_tdd_report()
        
        # Final assessment
        all_passed = tests_passed and build_passed and sandbox_compliant
        
        print(f"\n⏱️ Total validation time: {duration:.2f}s")
        
        if all_passed:
            print("🎉 TDD VALIDATION PASSED - Ready for development!")
            return True
        else:
            print("🚫 TDD VALIDATION FAILED - Fix issues before proceeding")
            print("\nIssues found:")
            if not tests_passed:
                print("  ❌ Test suite needs improvement")
            if not build_passed:
                print("  ❌ Production build is broken")
            if not sandbox_compliant:
                print("  ❌ Sandbox compliance issues")
            return False

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = Path(__file__).parent.parent
    
    runner = TDDTestRunner(project_root)
    success = runner.run_full_tdd_validation()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()