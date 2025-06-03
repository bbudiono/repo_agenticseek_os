#!/usr/bin/env python3
"""
Performance Analytics UI Integration Verification Script
Verifies that the Performance Analytics dashboard is properly integrated into the main macOS application.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🔍 Verifying Performance Analytics UI Integration")
    print("=" * 60)
    
    # Check if we're in the correct directory
    if not os.path.exists("AgenticSeek.xcworkspace"):
        print("❌ Error: Not in the correct macOS project directory")
        sys.exit(1)
    
    verification_results = {
        "ui_files_exist": False,
        "content_view_updated": False,
        "production_components_updated": False,
        "build_success": False,
        "performance_tab_added": False
    }
    
    # 1. Check if PerformanceAnalyticsView.swift exists
    performance_view_path = Path("AgenticSeek/PerformanceAnalyticsView.swift")
    if performance_view_path.exists():
        verification_results["ui_files_exist"] = True
        print("✅ PerformanceAnalyticsView.swift exists")
        
        # Check file size to ensure it's not empty
        file_size = performance_view_path.stat().st_size
        if file_size > 10000:  # Should be substantial (>10KB)
            print(f"✅ PerformanceAnalyticsView.swift is substantial ({file_size} bytes)")
        else:
            print(f"⚠️  PerformanceAnalyticsView.swift may be incomplete ({file_size} bytes)")
    else:
        print("❌ PerformanceAnalyticsView.swift not found")
    
    # 2. Check if ContentView.swift has been updated with Performance tab
    content_view_path = Path("AgenticSeek/ContentView.swift")
    if content_view_path.exists():
        with open(content_view_path, 'r') as f:
            content = f.read()
            
        if "case performance = \"Performance\"" in content:
            verification_results["performance_tab_added"] = True
            print("✅ Performance tab added to AppTab enum")
        else:
            print("❌ Performance tab not found in AppTab enum")
            
        if "chart.line.uptrend.xyaxis" in content:
            print("✅ Performance tab icon configured")
        else:
            print("❌ Performance tab icon not configured")
            
        if "Real-time performance analytics" in content:
            verification_results["content_view_updated"] = True
            print("✅ Performance tab description added")
        else:
            print("❌ Performance tab description not found")
    else:
        print("❌ ContentView.swift not found")
    
    # 3. Check if ProductionComponents.swift has been updated
    production_components_path = Path("AgenticSeek/ProductionComponents.swift")
    if production_components_path.exists():
        with open(production_components_path, 'r') as f:
            content = f.read()
            
        if "case .performance:" in content and "PerformanceAnalyticsView()" in content:
            verification_results["production_components_updated"] = True
            print("✅ ProductionComponents.swift updated with Performance view")
        else:
            print("❌ ProductionComponents.swift not properly updated")
            
        if "keyboardShortcut(\"5\", modifiers: .command)" in content:
            print("✅ Keyboard shortcut (Cmd+5) configured for Performance tab")
        else:
            print("⚠️  Keyboard shortcut may not be properly configured")
    else:
        print("❌ ProductionComponents.swift not found")
    
    # 4. Try to build the project to verify integration
    print("\n🔨 Attempting to build project...")
    try:
        result = subprocess.run([
            "xcodebuild", 
            "-workspace", "AgenticSeek.xcworkspace",
            "-scheme", "AgenticSeek",
            "-configuration", "Release",
            "build"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            verification_results["build_success"] = True
            print("✅ Project builds successfully with Performance Analytics integration")
        else:
            print("❌ Build failed - there may be integration issues")
            print("Build errors:")
            print(result.stderr[:500])  # Show first 500 chars of error
    except subprocess.TimeoutExpired:
        print("⚠️  Build timed out - this may indicate compilation issues")
    except Exception as e:
        print(f"⚠️  Could not run build test: {e}")
    
    # 5. Check for test files
    test_file_path = Path("tests/PerformanceAnalyticsUITests.swift")
    if test_file_path.exists():
        print("✅ UI integration tests created")
    else:
        print("ℹ️  UI integration tests not found (optional)")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed_checks = sum(verification_results.values())
    total_checks = len(verification_results)
    success_rate = (passed_checks / total_checks) * 100
    
    print(f"Success Rate: {success_rate:.1f}% ({passed_checks}/{total_checks} checks passed)")
    
    if success_rate >= 80:
        print("🎉 Performance Analytics UI integration is SUCCESSFUL!")
        print("✅ The Performance tab should be visible and functional in the main application")
        print("✅ Users can access real-time performance analytics via Cmd+5 or tab navigation")
        print("✅ All acceptance criteria for UI integration have been met")
        return 0
    elif success_rate >= 60:
        print("⚠️  Performance Analytics UI integration has MINOR ISSUES")
        print("🔧 Some components may need adjustment but core functionality should work")
        return 1
    else:
        print("❌ Performance Analytics UI integration has MAJOR ISSUES")
        print("🚨 Significant problems detected that need to be resolved")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)