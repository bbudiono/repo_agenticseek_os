#!/usr/bin/env python3
"""
ATOMIC TDD TO FIX 32-FILE MISALIGNMENT - NO FUCKING LIES
Real codebase alignment with actual file comparison and fixes
"""

import os
import sys
import subprocess
import unittest
import hashlib
import shutil
import re
from pathlib import Path

class TestReal32FileMisalignment(unittest.TestCase):
    """REAL tests for 32-file misalignment - NO SIMULATION"""
    
    def setUp(self):
        """Set up REAL test environment"""
        self.project_root = Path("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek")
        self.macos_path = self.project_root / "_macOS"
        self.sandbox_path = self.macos_path / "AgenticSeek-Sandbox"
        self.production_path = self.macos_path / "AgenticSeek"
    
    def test_01_get_real_misaligned_files(self):
        """RED: Get ACTUAL list of misaligned files"""
        misaligned_files = self.get_actual_misaligned_files()
        
        # This test will FAIL until we fix alignment
        self.assertEqual(len(misaligned_files), 0, f"Found {len(misaligned_files)} misaligned files: {misaligned_files[:10]}")
    
    def test_02_production_build_succeeds_real(self):
        """RED: Test that Production project builds successfully"""
        build_cmd = [
            "xcodebuild",
            "-project", str(self.production_path.parent / "AgenticSeek.xcodeproj"),
            "-scheme", "AgenticSeek",
            "-configuration", "Debug",
            "-destination", "platform=macOS,arch=arm64",
            "build"
        ]
        
        try:
            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(self.macos_path)
            )
            
            self.assertEqual(result.returncode, 0, f"Production build must succeed. Error: {result.stderr[:300]}")
            
        except subprocess.TimeoutExpired:
            self.fail("Production build timed out")
        except Exception as e:
            self.fail(f"Production build failed: {e}")
    
    def get_actual_misaligned_files(self):
        """Get REAL list of misaligned files using file content comparison"""
        misaligned = []
        
        if not self.sandbox_path.exists() or not self.production_path.exists():
            return ["MISSING_DIRECTORIES"]
        
        # Get all Swift files in both directories
        sandbox_files = set()
        production_files = set()
        
        for swift_file in self.sandbox_path.rglob("*.swift"):
            rel_path = swift_file.relative_to(self.sandbox_path)
            sandbox_files.add(rel_path)
        
        for swift_file in self.production_path.rglob("*.swift"):
            rel_path = swift_file.relative_to(self.production_path)
            production_files.add(rel_path)
        
        # Check for missing files
        missing_in_production = sandbox_files - production_files
        missing_in_sandbox = production_files - sandbox_files
        
        for missing_file in missing_in_production:
            misaligned.append(f"MISSING_IN_PROD: {missing_file}")
        
        for missing_file in missing_in_sandbox:
            misaligned.append(f"MISSING_IN_SANDBOX: {missing_file}")
        
        # Check content differences for common files
        common_files = sandbox_files & production_files
        
        for common_file in common_files:
            sandbox_file_path = self.sandbox_path / common_file
            production_file_path = self.production_path / common_file
            
            if self.files_content_different(sandbox_file_path, production_file_path):
                misaligned.append(f"CONTENT_DIFF: {common_file}")
        
        return misaligned
    
    def files_content_different(self, file1, file2):
        """Check if two files have different content (ignoring SANDBOX markers)"""
        try:
            with open(file1, 'r') as f1:
                content1 = f1.read()
            with open(file2, 'r') as f2:
                content2 = f2.read()
            
            # Remove SANDBOX markers for comparison
            content1_clean = re.sub(r'// SANDBOX FILE:.*?\n', '', content1, flags=re.MULTILINE)
            content2_clean = re.sub(r'// SANDBOX FILE:.*?\n', '', content2, flags=re.MULTILINE)
            
            # Also normalize component references for comparison
            content1_normalized = self.normalize_component_references(content1_clean, "Sandbox")
            content2_normalized = self.normalize_component_references(content2_clean, "Production")
            
            return content1_normalized != content2_normalized
            
        except Exception as e:
            print(f"Error comparing {file1} vs {file2}: {e}")
            return True
    
    def normalize_component_references(self, content, target_type):
        """Normalize component references for comparison"""
        if target_type == "Sandbox":
            # Convert Production references to Sandbox
            content = content.replace("ProductionSidebarView", "SandboxSidebarView")
            content = content.replace("ProductionDetailView", "SandboxDetailView")
            content = content.replace("ProductionLoadingView", "SandboxLoadingView")
            content = content.replace("ProductionChatView", "SandboxChatView")
            content = content.replace("ProductionModelsView", "SandboxModelsView")
            content = content.replace("ProductionConfigView", "SandboxConfigView")
            content = content.replace("ProductionTestsView", "SandboxTestsView")
        else:
            # Convert Sandbox references to Production
            content = content.replace("SandboxSidebarView", "ProductionSidebarView")
            content = content.replace("SandboxDetailView", "ProductionDetailView")
            content = content.replace("SandboxLoadingView", "ProductionLoadingView")
            content = content.replace("SandboxChatView", "ProductionChatView")
            content = content.replace("SandboxModelsView", "ProductionModelsView")
            content = content.replace("SandboxConfigView", "ProductionConfigView")
            content = content.replace("SandboxTestsView", "ProductionTestsView")
        
        return content

class Real32FileMisalignmentFixer:
    """Fix REAL 32-file misalignment with atomic operations"""
    
    def __init__(self):
        self.project_root = Path("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek")
        self.macos_path = self.project_root / "_macOS"
        self.sandbox_path = self.macos_path / "AgenticSeek-Sandbox"
        self.production_path = self.macos_path / "AgenticSeek"
    
    def fix_all_32_misaligned_files(self):
        """GREEN: Fix ALL 32 misaligned files"""
        print("🔥 FIXING 32-FILE MISALIGNMENT - NO LIES")
        print("=" * 50)
        
        # Step 1: Get actual misaligned files
        misaligned_files = self.get_actual_misaligned_files()
        print(f"📊 ACTUAL misaligned files: {len(misaligned_files)}")
        
        if len(misaligned_files) == 0:
            print("✅ No misaligned files found!")
            return True
        
        # Step 2: Show first 10 misaligned files
        print("🔍 First 10 misaligned files:")
        for i, file_info in enumerate(misaligned_files[:10]):
            print(f"  {i+1}. {file_info}")
        
        # Step 3: Fix missing files
        self.fix_missing_files(misaligned_files)
        
        # Step 4: Fix content differences
        self.fix_content_differences(misaligned_files)
        
        # Step 5: Fix component references
        self.fix_component_references()
        
        # Step 6: Verify fix
        remaining_misaligned = self.get_actual_misaligned_files()
        print(f"📊 REMAINING misaligned files: {len(remaining_misaligned)}")
        
        if len(remaining_misaligned) > 0:
            print("❌ Still have misaligned files:")
            for file_info in remaining_misaligned[:5]:
                print(f"  - {file_info}")
        
        return len(remaining_misaligned) == 0
    
    def get_actual_misaligned_files(self):
        """Get REAL list of misaligned files using file content comparison"""
        misaligned = []
        
        if not self.sandbox_path.exists() or not self.production_path.exists():
            return ["MISSING_DIRECTORIES"]
        
        # Get all Swift files in both directories
        sandbox_files = set()
        production_files = set()
        
        for swift_file in self.sandbox_path.rglob("*.swift"):
            rel_path = swift_file.relative_to(self.sandbox_path)
            sandbox_files.add(rel_path)
        
        for swift_file in self.production_path.rglob("*.swift"):
            rel_path = swift_file.relative_to(self.production_path)
            production_files.add(rel_path)
        
        # Check for missing files
        missing_in_production = sandbox_files - production_files
        missing_in_sandbox = production_files - sandbox_files
        
        for missing_file in missing_in_production:
            misaligned.append(f"MISSING_IN_PROD: {missing_file}")
        
        for missing_file in missing_in_sandbox:
            misaligned.append(f"MISSING_IN_SANDBOX: {missing_file}")
        
        # Check content differences for common files
        common_files = sandbox_files & production_files
        
        for common_file in common_files:
            sandbox_file_path = self.sandbox_path / common_file
            production_file_path = self.production_path / common_file
            
            if self.files_content_different(sandbox_file_path, production_file_path):
                misaligned.append(f"CONTENT_DIFF: {common_file}")
        
        return misaligned
    
    def files_content_different(self, file1, file2):
        """Check if two files have different content (ignoring SANDBOX markers)"""
        try:
            with open(file1, 'r') as f1:
                content1 = f1.read()
            with open(file2, 'r') as f2:
                content2 = f2.read()
            
            # Remove SANDBOX markers for comparison
            content1_clean = re.sub(r'// SANDBOX FILE:.*?\n', '', content1, flags=re.MULTILINE)
            content2_clean = re.sub(r'// SANDBOX FILE:.*?\n', '', content2, flags=re.MULTILINE)
            
            # Also normalize component references for comparison
            content1_normalized = self.normalize_component_references(content1_clean, "Sandbox")
            content2_normalized = self.normalize_component_references(content2_clean, "Production")
            
            return content1_normalized != content2_normalized
            
        except Exception as e:
            print(f"Error comparing {file1} vs {file2}: {e}")
            return True
    
    def normalize_component_references(self, content, target_type):
        """Normalize component references for comparison"""
        if target_type == "Sandbox":
            # Convert Production references to Sandbox
            content = content.replace("ProductionSidebarView", "SandboxSidebarView")
            content = content.replace("ProductionDetailView", "SandboxDetailView")
            content = content.replace("ProductionLoadingView", "SandboxLoadingView")
            content = content.replace("ProductionChatView", "SandboxChatView")
            content = content.replace("ProductionModelsView", "SandboxModelsView")
            content = content.replace("ProductionConfigView", "SandboxConfigView")
            content = content.replace("ProductionTestsView", "SandboxTestsView")
        else:
            # Convert Sandbox references to Production
            content = content.replace("SandboxSidebarView", "ProductionSidebarView")
            content = content.replace("SandboxDetailView", "ProductionDetailView")
            content = content.replace("SandboxLoadingView", "ProductionLoadingView")
            content = content.replace("SandboxChatView", "ProductionChatView")
            content = content.replace("SandboxModelsView", "ProductionModelsView")
            content = content.replace("SandboxConfigView", "ProductionConfigView")
            content = content.replace("SandboxTestsView", "ProductionTestsView")
        
        return content
    
    def fix_missing_files(self, misaligned_files):
        """Fix missing files by copying from source"""
        print("📁 Fixing missing files...")
        
        for file_info in misaligned_files:
            if file_info.startswith("MISSING_IN_PROD:"):
                # Copy from Sandbox to Production
                rel_path = file_info.replace("MISSING_IN_PROD: ", "")
                source_file = self.sandbox_path / rel_path
                dest_file = self.production_path / rel_path
                
                if source_file.exists():
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, dest_file)
                    
                    # Remove SANDBOX markers from Production
                    self.remove_sandbox_markers_from_file(dest_file)
                    
                    print(f"  ✅ Copied {rel_path} to Production")
                
            elif file_info.startswith("MISSING_IN_SANDBOX:"):
                # Copy from Production to Sandbox
                rel_path = file_info.replace("MISSING_IN_SANDBOX: ", "")
                source_file = self.production_path / rel_path
                dest_file = self.sandbox_path / rel_path
                
                if source_file.exists():
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, dest_file)
                    
                    # Add SANDBOX markers to Sandbox
                    self.add_sandbox_markers_to_file(dest_file)
                    
                    print(f"  ✅ Copied {rel_path} to Sandbox")
    
    def fix_content_differences(self, misaligned_files):
        """Fix content differences between Sandbox and Production files"""
        print("📝 Fixing content differences...")
        
        for file_info in misaligned_files:
            if file_info.startswith("CONTENT_DIFF:"):
                rel_path = file_info.replace("CONTENT_DIFF: ", "")
                sandbox_file = self.sandbox_path / rel_path
                production_file = self.production_path / rel_path
                
                if sandbox_file.exists() and production_file.exists():
                    # Copy Sandbox content to Production and fix component references
                    with open(sandbox_file, 'r') as f:
                        content = f.read()
                    
                    # Remove SANDBOX markers
                    content = re.sub(r'// SANDBOX FILE:.*?\n', '', content, flags=re.MULTILINE)
                    
                    # Convert Sandbox component references to Production
                    content = content.replace("SandboxSidebarView", "ProductionSidebarView")
                    content = content.replace("SandboxDetailView", "ProductionDetailView")
                    content = content.replace("SandboxLoadingView", "ProductionLoadingView")
                    content = content.replace("SandboxChatView", "ProductionChatView")
                    content = content.replace("SandboxModelsView", "ProductionModelsView")
                    content = content.replace("SandboxConfigView", "ProductionConfigView")
                    content = content.replace("SandboxTestsView", "ProductionTestsView")
                    
                    with open(production_file, 'w') as f:
                        f.write(content)
                    
                    print(f"  ✅ Fixed content diff in {rel_path}")
    
    def fix_component_references(self):
        """Fix component references in both Sandbox and Production"""
        print("🔗 Fixing component references...")
        
        # Fix Sandbox ContentView to use Sandbox components
        sandbox_content_view = self.sandbox_path / "ContentView.swift"
        if sandbox_content_view.exists():
            with open(sandbox_content_view, 'r') as f:
                content = f.read()
            
            content = content.replace("ProductionSidebarView", "SandboxSidebarView")
            content = content.replace("ProductionDetailView", "SandboxDetailView")
            
            if "// SANDBOX FILE:" not in content:
                content = "// SANDBOX FILE: For testing/development. See .cursorrules.\n" + content
            
            with open(sandbox_content_view, 'w') as f:
                f.write(content)
            
            print("  ✅ Fixed Sandbox ContentView references")
        
        # Fix Production ContentView to use Production components
        production_content_view = self.production_path / "ContentView.swift"
        if production_content_view.exists():
            with open(production_content_view, 'r') as f:
                content = f.read()
            
            content = content.replace("SandboxSidebarView", "ProductionSidebarView")
            content = content.replace("SandboxDetailView", "ProductionDetailView")
            
            # Remove SANDBOX markers
            content = re.sub(r'// SANDBOX FILE:.*?\n', '', content, flags=re.MULTILINE)
            
            with open(production_content_view, 'w') as f:
                f.write(content)
            
            print("  ✅ Fixed Production ContentView references")
    
    def remove_sandbox_markers_from_file(self, file_path):
        """Remove SANDBOX markers from a file"""
        if not file_path.exists():
            return
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        content = re.sub(r'// SANDBOX FILE:.*?\n', '', content, flags=re.MULTILINE)
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    def add_sandbox_markers_to_file(self, file_path):
        """Add SANDBOX markers to a file"""
        if not file_path.exists():
            return
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        if "// SANDBOX FILE:" not in content:
            content = "// SANDBOX FILE: For testing/development. See .cursorrules.\n" + content
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    def verify_builds_after_fix(self):
        """Verify that both builds work after alignment fix"""
        print("\n🔨 VERIFYING BUILDS AFTER ALIGNMENT FIX")
        print("=" * 45)
        
        # Test workspace build
        workspace_build_cmd = [
            "xcodebuild",
            "-workspace", str(self.macos_path / "AgenticSeek.xcworkspace"),
            "-scheme", "AgenticSeek",
            "-configuration", "Debug",
            "-destination", "platform=macOS,arch=arm64",
            "build"
        ]
        
        try:
            result = subprocess.run(
                workspace_build_cmd,
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(self.macos_path)
            )
            
            if result.returncode == 0:
                print("✅ WORKSPACE BUILD: SUCCESS")
                workspace_success = True
            else:
                print(f"❌ WORKSPACE BUILD: FAILED (exit {result.returncode})")
                workspace_success = False
                
        except Exception as e:
            print(f"❌ WORKSPACE BUILD: EXCEPTION {e}")
            workspace_success = False
        
        # Test Production project build
        production_build_cmd = [
            "xcodebuild",
            "-project", str(self.macos_path / "AgenticSeek.xcodeproj"),
            "-scheme", "AgenticSeek",
            "-configuration", "Debug",
            "-destination", "platform=macOS,arch=arm64",
            "build"
        ]
        
        try:
            result = subprocess.run(
                production_build_cmd,
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(self.macos_path)
            )
            
            if result.returncode == 0:
                print("✅ PRODUCTION BUILD: SUCCESS")
                production_success = True
            else:
                print(f"❌ PRODUCTION BUILD: FAILED (exit {result.returncode})")
                production_success = False
                
        except Exception as e:
            print(f"❌ PRODUCTION BUILD: EXCEPTION {e}")
            production_success = False
        
        return workspace_success and production_success

if __name__ == "__main__":
    print("🚀 ATOMIC TDD: FIX 32-FILE MISALIGNMENT - NO FUCKING LIES")
    print("Real file comparison and alignment fixes")
    print()
    
    # Phase 1: RED - Run tests to identify REAL issues
    print("🔴 RED PHASE: Identifying REAL 32-file misalignment...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestReal32FileMisalignment)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✅ No misalignment found!")
        sys.exit(0)
    
    print(f"\n❌ Found REAL alignment issues: {len(result.failures)} failures")
    
    # Phase 2: GREEN - Fix ALL alignment issues
    print("\n🟢 GREEN PHASE: Fixing ALL 32-file misalignment...")
    fixer = Real32FileMisalignmentFixer()
    alignment_success = fixer.fix_all_32_misaligned_files()
    
    # Phase 3: REFACTOR - Verify fix with builds
    print("\n🔵 REFACTOR PHASE: Verifying builds after alignment...")
    build_success = fixer.verify_builds_after_fix()
    
    if alignment_success and build_success:
        print("\n🎉 32-FILE MISALIGNMENT FIXED!")
        print("✅ 100% CODEBASE ALIGNMENT ACHIEVED")
        print("✅ ALL BUILDS SUCCEED")
    else:
        print("\n❌ MISALIGNMENT ISSUES REMAIN")
        print("🔧 Additional fixes required")
    
    sys.exit(0 if (alignment_success and build_success) else 1)