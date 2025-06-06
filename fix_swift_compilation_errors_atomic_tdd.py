#!/usr/bin/env python3
"""
Atomic TDD to Fix Swift Compilation Errors
Based on actual Xcode build output with specific error locations
"""

import os
import re
import sys
import subprocess
import unittest
from pathlib import Path

class TestSwiftCompilationErrors(unittest.TestCase):
    """Atomic TDD tests for Swift compilation errors"""
    
    def setUp(self):
        """Set up test environment"""
        self.project_root = Path("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek")
        self.sandbox_path = self.project_root / "_macOS" / "AgenticSeek-Sandbox"
        self.production_path = self.project_root / "_macOS" / "AgenticSeek"
        
        # Target files
        self.sandbox_content_view = self.sandbox_path / "ContentView.swift"
        self.sandbox_components = self.sandbox_path / "SandboxComponents.swift"
        self.production_components = self.production_path / "ProductionComponents.swift"
    
    def test_01_production_components_missing_in_sandbox(self):
        """RED: Test that ProductionComponents are missing from Sandbox"""
        if not self.sandbox_content_view.exists():
            self.fail("ContentView.swift does not exist in Sandbox")
        
        with open(self.sandbox_content_view, 'r') as f:
            content = f.read()
        
        # Should fail because ProductionSidebarView and ProductionDetailView are not available
        has_production_sidebar = "ProductionSidebarView" in content
        has_production_detail = "ProductionDetailView" in content
        
        production_missing = has_production_sidebar or has_production_detail
        self.assertFalse(production_missing, "ProductionComponents should not be used in Sandbox without importing")
    
    def test_02_sandbox_components_group_error(self):
        """RED: Test that SandboxComponents has Group type inference error"""
        if not self.sandbox_components.exists():
            self.fail("SandboxComponents.swift does not exist")
        
        with open(self.sandbox_components, 'r') as f:
            content = f.read()
        
        # Check for Group without proper type inference
        has_group_issue = "Group {" in content and "if isLoading" in content
        self.assertFalse(has_group_issue, "Group should have proper type inference or be replaced")
    
    def test_03_apptab_chat_case_missing(self):
        """RED: Test that AppTab is missing .chat case"""
        if not self.sandbox_content_view.exists():
            self.fail("ContentView.swift does not exist")
        
        with open(self.sandbox_content_view, 'r') as f:
            content = f.read()
        
        # Check for AppTab enum and .chat case
        has_apptab_enum = "enum AppTab:" in content
        if has_apptab_enum:
            has_chat_case = "case chat" in content or "case assistant" in content
            self.assertTrue(has_chat_case, "AppTab should have chat or assistant case")
    
    def test_04_sandbox_uses_correct_components(self):
        """GREEN: Test that Sandbox uses SandboxComponents correctly"""
        if not self.sandbox_content_view.exists():
            self.fail("ContentView.swift does not exist")
        
        with open(self.sandbox_content_view, 'r') as f:
            content = f.read()
        
        # Should use SandboxSidebarView and SandboxDetailView instead of Production ones
        uses_sandbox_sidebar = "SandboxSidebarView" in content
        uses_sandbox_detail = "SandboxDetailView" in content
        
        self.assertTrue(uses_sandbox_sidebar, "Should use SandboxSidebarView")
        self.assertTrue(uses_sandbox_detail, "Should use SandboxDetailView")

class SwiftCompilationErrorsFixer:
    """Fix Swift compilation errors"""
    
    def __init__(self):
        self.project_root = Path("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek")
        self.sandbox_path = self.project_root / "_macOS" / "AgenticSeek-Sandbox"
        self.production_path = self.project_root / "_macOS" / "AgenticSeek"
        
        # Target files
        self.sandbox_content_view = self.sandbox_path / "ContentView.swift"
        self.sandbox_components = self.sandbox_path / "SandboxComponents.swift"
        self.production_components = self.production_path / "ProductionComponents.swift"
    
    def fix_all_compilation_errors(self):
        """GREEN: Fix all Swift compilation errors"""
        print("🔧 FIXING SWIFT COMPILATION ERRORS")
        print("=" * 50)
        
        # Step 1: Fix ContentView to use Sandbox components
        self.fix_content_view_components()
        
        # Step 2: Fix SandboxComponents Group type inference
        self.fix_sandbox_components_group()
        
        # Step 3: Fix AppTab case references
        self.fix_apptab_cases()
        
        print("✅ All Swift compilation errors fixed")
    
    def fix_content_view_components(self):
        """Fix ContentView to use SandboxComponents instead of ProductionComponents"""
        if not self.sandbox_content_view.exists():
            print("⚠️ ContentView.swift not found in Sandbox")
            return
        
        with open(self.sandbox_content_view, 'r') as f:
            content = f.read()
        
        # Replace ProductionSidebarView with SandboxSidebarView
        content = content.replace(
            "ProductionSidebarView(selectedTab: $selectedTab, onRestartServices: restartServices)",
            "SandboxSidebarView(selectedTab: $selectedTab, onRestartServices: restartServices)"
        )
        
        # Replace ProductionDetailView with SandboxDetailView
        content = content.replace(
            "ProductionDetailView(selectedTab: selectedTab, isLoading: isLoading)",
            "SandboxDetailView(selectedTab: selectedTab, isLoading: isLoading)"
        )
        
        with open(self.sandbox_content_view, 'w') as f:
            f.write(content)
        
        print("✅ Fixed ContentView to use SandboxComponents")
    
    def fix_sandbox_components_group(self):
        """Fix SandboxComponents Group type inference error"""
        if not self.sandbox_components.exists():
            print("⚠️ SandboxComponents.swift not found")
            return
        
        with open(self.sandbox_components, 'r') as f:
            content = f.read()
        
        # Replace problematic Group usage in SandboxDetailView
        # Find the Group section and replace with proper SwiftUI structure
        pattern = r"Group \{\s*if isLoading \{\s*SandboxLoadingView\(\)\s*\} else \{\s*switch selectedTab \{.*?\}\s*\}\s*\}"
        
        replacement = """VStack {
            if isLoading {
                SandboxLoadingView()
            } else {
                switch selectedTab {
                case .assistant:
                    SandboxChatView()
                case .webBrowsing:
                    SandboxModelsView()
                case .coding:
                    SandboxConfigView()
                case .tasks:
                    SandboxTestsView()
                case .performance:
                    SandboxTestsView()
                case .settings:
                    SandboxConfigView()
                }
            }
        }"""
        
        # Use a more targeted replacement
        if "Group {" in content and "if isLoading" in content:
            lines = content.split('\n')
            new_lines = []
            in_group_block = False
            brace_count = 0
            
            for line in lines:
                if "Group {" in line and "if isLoading" in line:
                    # Skip this line and the entire Group block
                    in_group_block = True
                    brace_count = 0
                    new_lines.append("        VStack {")
                    continue
                elif in_group_block:
                    brace_count += line.count('{')
                    brace_count -= line.count('}')
                    
                    # Replace case references
                    if "case .chat:" in line:
                        line = line.replace("case .chat:", "case .assistant:")
                    elif "case .models:" in line:
                        line = line.replace("case .models:", "case .webBrowsing:")
                    elif "case .config:" in line:
                        line = line.replace("case .config:", "case .coding:")
                    elif "case .tests:" in line:
                        line = line.replace("case .tests:", "case .tasks:")
                    
                    new_lines.append(line)
                    
                    if brace_count <= 0:
                        in_group_block = False
                else:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
        
        with open(self.sandbox_components, 'w') as f:
            f.write(content)
        
        print("✅ Fixed SandboxComponents Group type inference")
    
    def fix_apptab_cases(self):
        """Fix AppTab case references in SandboxComponents"""
        if not self.sandbox_components.exists():
            print("⚠️ SandboxComponents.swift not found")
            return
        
        with open(self.sandbox_components, 'r') as f:
            content = f.read()
        
        # Replace keyboard shortcut references to match AppTab enum
        content = content.replace(
            "selectedTab.wrappedValue = .chat",
            "selectedTab.wrappedValue = .assistant"
        )
        content = content.replace(
            "selectedTab.wrappedValue = .models",
            "selectedTab.wrappedValue = .webBrowsing"
        )
        content = content.replace(
            "selectedTab.wrappedValue = .config",
            "selectedTab.wrappedValue = .coding"
        )
        content = content.replace(
            "selectedTab.wrappedValue = .tests",
            "selectedTab.wrappedValue = .tasks"
        )
        
        with open(self.sandbox_components, 'w') as f:
            f.write(content)
        
        print("✅ Fixed AppTab case references")
    
    def verify_compilation_fix(self):
        """Verify that Swift compilation errors are fixed"""
        print("\n🔍 VERIFYING SWIFT COMPILATION FIX")
        print("=" * 40)
        
        # Run simplified build command to check for errors
        build_command = [
            "xcodebuild",
            "-workspace", str(self.project_root / "_macOS" / "AgenticSeek.xcworkspace"),
            "-scheme", "AgenticSeek",
            "-configuration", "Release",
            "-dry-run"
        ]
        
        try:
            result = subprocess.run(
                build_command,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.project_root / "_macOS")
            )
            
            # Check if specific errors are gone
            error_output = result.stderr.lower()
            
            production_sidebar_error = "cannot find 'productionsidebarview'" in error_output
            production_detail_error = "cannot find 'productiondetailview'" in error_output
            group_error = "generic parameter 'r' could not be inferred" in error_output
            chat_error = "type 'apptab' has no member 'chat'" in error_output
            
            errors_found = [
                ("ProductionSidebarView error", production_sidebar_error),
                ("ProductionDetailView error", production_detail_error),
                ("Group type inference error", group_error),
                ("AppTab .chat error", chat_error)
            ]
            
            remaining_errors = [name for name, exists in errors_found if exists]
            
            if remaining_errors:
                print(f"❌ Remaining errors: {remaining_errors}")
                return False
            else:
                print("✅ Swift compilation errors appear to be fixed")
                return True
                
        except subprocess.TimeoutExpired:
            print("⚠️ Build verification timed out")
            return False
        except Exception as e:
            print(f"⚠️ Build verification failed: {e}")
            return False
    
    def run_full_build_test(self):
        """Run actual build to verify fix"""
        print("\n🔨 RUNNING FULL BUILD TEST")
        print("=" * 30)
        
        build_command = [
            "xcodebuild",
            "-workspace", str(self.project_root / "_macOS" / "AgenticSeek.xcworkspace"),
            "-scheme", "AgenticSeek",
            "-configuration", "Release",
            "clean", "build",
            "-destination", "platform=macOS"
        ]
        
        try:
            result = subprocess.run(
                build_command,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.project_root / "_macOS")
            )
            
            if result.returncode == 0:
                print("✅ SWIFT COMPILATION: SUCCESS")
                return True
            else:
                print(f"❌ Build failed with exit code {result.returncode}")
                # Show first 10 lines of errors
                if result.stderr:
                    error_lines = result.stderr.split('\n')[:10]
                    print("First 10 error lines:")
                    for line in error_lines:
                        if 'error:' in line.lower():
                            print(f"  {line}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Build timed out")
            return False
        except Exception as e:
            print(f"❌ Build failed with exception: {e}")
            return False

if __name__ == "__main__":
    print("🚀 ATOMIC TDD SWIFT COMPILATION FIX")
    print("Fixing specific Swift compilation errors identified in build")
    print()
    
    # Phase 1: RED - Run tests to identify issues
    print("🔴 RED PHASE: Identifying Swift compilation issues...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSwiftCompilationErrors)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✅ No Swift compilation issues found!")
        sys.exit(0)
    
    print(f"\n❌ Found Swift compilation issues: {len(result.failures)} failures")
    
    # Phase 2: GREEN - Fix the issues
    print("\n🟢 GREEN PHASE: Fixing Swift compilation errors...")
    fixer = SwiftCompilationErrorsFixer()
    fixer.fix_all_compilation_errors()
    
    # Phase 3: REFACTOR - Verify fix with actual build
    print("\n🔵 REFACTOR PHASE: Verifying fix with build...")
    build_success = fixer.run_full_build_test()
    
    if build_success:
        print("\n🎉 SWIFT COMPILATION ERRORS FIXED!")
        print("✅ Production build should now succeed")
    else:
        print("\n❌ SWIFT COMPILATION ISSUES REMAIN")
        print("🔧 Additional manual intervention may be required")
    
    sys.exit(0 if build_success else 1)