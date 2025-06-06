#!/usr/bin/env python3
"""
Atomic TDD to Fix AppTab Naming Conflict
Swift compilation error: 'AppTab' is ambiguous for type lookup
"""

import os
import re
import sys
import unittest
from pathlib import Path

class TestAppTabConflict(unittest.TestCase):
    """Atomic TDD tests for AppTab naming conflict"""
    
    def setUp(self):
        """Set up test environment"""
        self.project_root = Path("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek")
        self.sandbox_path = self.project_root / "_macOS" / "AgenticSeek-Sandbox"
        self.content_view = self.sandbox_path / "ContentView.swift"
        self.sandbox_components = self.sandbox_path / "SandboxComponents.swift"
    
    def test_01_apptab_conflict_exists(self):
        """RED: Test that AppTab conflict exists"""
        # Check if AppTab is defined in both files
        content_view_has_apptab = False
        sandbox_components_has_apptab = False
        
        if self.content_view.exists():
            with open(self.content_view, 'r') as f:
                content = f.read()
                content_view_has_apptab = "enum AppTab:" in content
        
        if self.sandbox_components.exists():
            with open(self.sandbox_components, 'r') as f:
                content = f.read()
                sandbox_components_has_apptab = "enum AppTab:" in content
        
        print(f"ContentView has AppTab: {content_view_has_apptab}")
        print(f"SandboxComponents has AppTab: {sandbox_components_has_apptab}")
        
        # Test should fail if both have AppTab (RED phase)
        conflict_exists = content_view_has_apptab and sandbox_components_has_apptab
        self.assertFalse(conflict_exists, "AppTab should not be defined in both files")
    
    def test_02_content_view_has_apptab_definition(self):
        """GREEN: Test that ContentView has the complete AppTab definition"""
        if not self.content_view.exists():
            self.fail("ContentView.swift does not exist")
        
        with open(self.content_view, 'r') as f:
            content = f.read()
        
        # Check for complete AppTab enum definition with all cases
        apptab_pattern = r"enum AppTab: String, CaseIterable \{.*?case assistant.*?case settings.*?\}"
        has_complete_apptab = bool(re.search(apptab_pattern, content, re.DOTALL))
        
        self.assertTrue(has_complete_apptab, "ContentView should have complete AppTab definition")
    
    def test_03_sandbox_components_no_apptab(self):
        """GREEN: Test that SandboxComponents does not have AppTab definition"""
        if not self.sandbox_components.exists():
            self.fail("SandboxComponents.swift does not exist")
        
        with open(self.sandbox_components, 'r') as f:
            content = f.read()
        
        has_apptab = "enum AppTab:" in content
        self.assertFalse(has_apptab, "SandboxComponents should not have AppTab definition")

class AppTabConflictFixer:
    """Fix AppTab naming conflict"""
    
    def __init__(self):
        self.project_root = Path("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek")
        self.sandbox_path = self.project_root / "_macOS" / "AgenticSeek-Sandbox"
        self.content_view = self.sandbox_path / "ContentView.swift"
        self.sandbox_components = self.sandbox_path / "SandboxComponents.swift"
    
    def fix_conflict(self):
        """GREEN: Fix the AppTab naming conflict"""
        print("🔧 FIXING APPTAB NAMING CONFLICT")
        print("=" * 40)
        
        # Step 1: Remove AppTab from SandboxComponents
        self.remove_apptab_from_sandbox_components()
        
        # Step 2: Ensure ContentView has the complete definition
        self.ensure_content_view_has_apptab()
        
        print("✅ AppTab conflict fixed")
    
    def remove_apptab_from_sandbox_components(self):
        """Remove AppTab enum from SandboxComponents.swift"""
        if not self.sandbox_components.exists():
            print("⚠️ SandboxComponents.swift not found")
            return
        
        with open(self.sandbox_components, 'r') as f:
            content = f.read()
        
        # Remove AppTab enum definition
        # Find the enum definition and remove it
        lines = content.split('\n')
        new_lines = []
        skip_enum = False
        brace_count = 0
        
        for line in lines:
            if "enum AppTab:" in line:
                print(f"🗑️ Removing AppTab enum from SandboxComponents")
                skip_enum = True
                brace_count = 0
                continue
            
            if skip_enum:
                # Count braces to find end of enum
                brace_count += line.count('{')
                brace_count -= line.count('}')
                
                if brace_count <= 0:
                    skip_enum = False
                continue
            
            new_lines.append(line)
        
        # Write the cleaned content
        with open(self.sandbox_components, 'w') as f:
            f.write('\n'.join(new_lines))
        
        print("✅ Removed AppTab from SandboxComponents.swift")
    
    def ensure_content_view_has_apptab(self):
        """Ensure ContentView has the complete AppTab definition"""
        if not self.content_view.exists():
            print("⚠️ ContentView.swift not found")
            return
        
        with open(self.content_view, 'r') as f:
            content = f.read()
        
        # Check if AppTab is already defined
        if "enum AppTab:" in content:
            print("✅ ContentView already has AppTab definition")
            return
        
        # If not, add the AppTab definition
        apptab_definition = '''
// MARK: - Enhanced App Tabs for AgenticSeek

enum AppTab: String, CaseIterable {
    case assistant = "Assistant"
    case webBrowsing = "Web Browsing"
    case coding = "Coding"
    case tasks = "Tasks"
    case performance = "Performance"
    case settings = "Settings"
    
    var icon: String {
        switch self {
        case .assistant: return "brain.head.profile"
        case .webBrowsing: return "globe"
        case .coding: return "chevron.left.forwardslash.chevron.right"
        case .tasks: return "list.bullet.clipboard"
        case .performance: return "chart.line.uptrend.xyaxis"
        case .settings: return "gearshape"
        }
    }
    
    var description: String {
        switch self {
        case .assistant: return "Voice-enabled AI assistant"
        case .webBrowsing: return "Autonomous web browsing"
        case .coding: return "Multi-language code assistant"
        case .tasks: return "Task planning and execution"
        case .performance: return "Real-time performance analytics"
        case .settings: return "Application settings"
        }
    }
}
'''
        
        # Add AppTab definition before the last closing brace
        lines = content.split('\n')
        insert_index = -2  # Before the last closing brace
        
        lines.insert(insert_index, apptab_definition)
        
        with open(self.content_view, 'w') as f:
            f.write('\n'.join(lines))
        
        print("✅ Added AppTab definition to ContentView.swift")
    
    def verify_fix(self):
        """Verify that the fix was successful"""
        print("\n🔍 VERIFYING FIX")
        print("=" * 20)
        
        # Run the tests again to verify
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestAppTabConflict)
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("✅ APPTAB CONFLICT: FIXED")
            return True
        else:
            print("❌ APPTAB CONFLICT: STILL EXISTS")
            return False

if __name__ == "__main__":
    print("🚀 ATOMIC TDD APPTAB CONFLICT FIX")
    print("Fixing Swift compilation error: AppTab ambiguous type")
    print()
    
    # Phase 1: RED - Run tests to identify conflict
    print("🔴 RED PHASE: Identifying AppTab conflict...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAppTabConflict)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✅ No AppTab conflict found!")
        sys.exit(0)
    
    print(f"\n❌ Found AppTab conflict: {len(result.failures)} failures")
    
    # Phase 2: GREEN - Fix the conflict
    print("\n🟢 GREEN PHASE: Fixing AppTab conflict...")
    fixer = AppTabConflictFixer()
    fixer.fix_conflict()
    
    # Phase 3: REFACTOR - Verify fix
    print("\n🔵 REFACTOR PHASE: Verifying fix...")
    success = fixer.verify_fix()
    
    if success:
        print("\n🎉 APPTAB CONFLICT FIXED!")
        print("✅ Swift compilation should now succeed")
    else:
        print("\n❌ APPTAB CONFLICT REMAINS")
        print("🔧 Manual intervention may be required")
    
    sys.exit(0 if success else 1)