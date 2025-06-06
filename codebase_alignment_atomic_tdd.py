#!/usr/bin/env python3
"""
Atomic TDD for 100% Codebase Alignment
Fix the 26-file misalignment between Sandbox and Production
"""

import os
import sys
import shutil
import unittest
import sqlite3
from datetime import datetime
from pathlib import Path
import difflib

class TestCodebaseAlignment(unittest.TestCase):
    """Atomic TDD tests for codebase alignment"""
    
    def setUp(self):
        """Set up test environment"""
        self.project_root = Path("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek")
        self.macos_path = self.project_root / "_macOS"
        self.production_path = self.macos_path / "AgenticSeek"
        self.sandbox_path = self.macos_path / "AgenticSeek-Sandbox"
        
        # Test database for alignment tracking
        self.db_path = self.project_root / "alignment_verification.db"
        self.init_database()
    
    def init_database(self):
        """Initialize alignment tracking database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alignment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                file_path TEXT NOT NULL,
                alignment_status TEXT NOT NULL,
                details TEXT,
                diff_output TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def test_01_identify_misaligned_files(self):
        """RED: Test to identify all misaligned files"""
        production_files = list(self.production_path.glob("**/*.swift"))
        misaligned_files = []
        
        for prod_file in production_files:
            rel_path = prod_file.relative_to(self.production_path)
            sandbox_file = self.sandbox_path / rel_path
            
            if not sandbox_file.exists():
                misaligned_files.append((str(rel_path), "missing_in_sandbox"))
                continue
            
            # Read and compare content
            with open(prod_file, 'r') as f:
                prod_content = f.read()
            with open(sandbox_file, 'r') as f:
                sandbox_content = f.read()
            
            # Remove sandbox watermark for comparison
            sandbox_clean = sandbox_content.replace(
                "// SANDBOX FILE: For testing/development. See .cursorrules.", ""
            ).strip()
            
            if prod_content.strip() != sandbox_clean:
                misaligned_files.append((str(rel_path), "content_mismatch"))
        
        # Log all misaligned files
        for file_path, issue_type in misaligned_files:
            self.log_alignment(file_path, "MISALIGNED", issue_type)
        
        print(f"🔍 Found {len(misaligned_files)} misaligned files")
        for file_path, issue_type in misaligned_files[:10]:  # Show first 10
            print(f"  ❌ {file_path}: {issue_type}")
        
        # Test should fail initially (RED phase)
        self.assertTrue(len(misaligned_files) == 0, f"Found {len(misaligned_files)} misaligned files")
    
    def test_02_production_has_all_required_files(self):
        """RED: Test that Production has all required files"""
        required_files = [
            "ChatbotModels.swift",
            "ProductionComponents.swift", 
            "ContentView.swift",
            "AuthenticationManager.swift",
            "DesignSystem.swift"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = self.production_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            self.log_alignment("production_files", "MISSING", f"Missing: {', '.join(missing_files)}")
        else:
            self.log_alignment("production_files", "COMPLETE", "All required files present")
        
        self.assertEqual(len(missing_files), 0, f"Missing files in Production: {missing_files}")
    
    def test_03_sandbox_has_watermarks(self):
        """RED: Test that Sandbox files have proper watermarks"""
        sandbox_files = list(self.sandbox_path.glob("**/*.swift"))
        files_without_watermark = []
        
        for sandbox_file in sandbox_files:
            with open(sandbox_file, 'r') as f:
                content = f.read()
            
            if "// SANDBOX FILE: For testing/development. See .cursorrules." not in content:
                rel_path = sandbox_file.relative_to(self.sandbox_path)
                files_without_watermark.append(str(rel_path))
        
        if files_without_watermark:
            self.log_alignment("sandbox_watermarks", "MISSING", f"Files without watermark: {len(files_without_watermark)}")
        else:
            self.log_alignment("sandbox_watermarks", "COMPLETE", "All files have watermarks")
        
        self.assertEqual(len(files_without_watermark), 0, f"Files without watermarks: {files_without_watermark[:5]}")
    
    def log_alignment(self, file_path, status, details, diff_output=None):
        """Log alignment result to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO alignment_results (timestamp, file_path, alignment_status, details, diff_output)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            file_path,
            status,
            details,
            diff_output
        ))
        conn.commit()
        conn.close()

class CodebaseAligner:
    """Fix codebase alignment between Sandbox and Production"""
    
    def __init__(self):
        self.project_root = Path("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek")
        self.macos_path = self.project_root / "_macOS"
        self.production_path = self.macos_path / "AgenticSeek"
        self.sandbox_path = self.macos_path / "AgenticSeek-Sandbox"
        
    def fix_alignment(self):
        """GREEN: Fix the alignment issues"""
        print("🔧 FIXING CODEBASE ALIGNMENT")
        print("=" * 50)
        
        # Step 1: Sync missing files from Production to Sandbox
        self.sync_missing_files()
        
        # Step 2: Sync content differences
        self.sync_content_differences()
        
        # Step 3: Add watermarks to Sandbox files
        self.add_sandbox_watermarks()
        
        print("✅ Codebase alignment fixes completed")
    
    def sync_missing_files(self):
        """Sync files missing from Sandbox"""
        production_files = list(self.production_path.glob("**/*.swift"))
        synced_count = 0
        
        for prod_file in production_files:
            rel_path = prod_file.relative_to(self.production_path)
            sandbox_file = self.sandbox_path / rel_path
            
            if not sandbox_file.exists():
                print(f"📁 Copying missing file: {rel_path}")
                
                # Create directory if it doesn't exist
                sandbox_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file from Production to Sandbox
                shutil.copy2(prod_file, sandbox_file)
                synced_count += 1
        
        print(f"📄 Synced {synced_count} missing files")
    
    def sync_content_differences(self):
        """Sync content differences between Production and Sandbox"""
        production_files = list(self.production_path.glob("**/*.swift"))
        synced_count = 0
        
        for prod_file in production_files:
            rel_path = prod_file.relative_to(self.production_path)
            sandbox_file = self.sandbox_path / rel_path
            
            if not sandbox_file.exists():
                continue  # Already handled in sync_missing_files
            
            # Read both files
            with open(prod_file, 'r') as f:
                prod_content = f.read()
            with open(sandbox_file, 'r') as f:
                sandbox_content = f.read()
            
            # Remove watermark for comparison
            sandbox_clean = sandbox_content.replace(
                "// SANDBOX FILE: For testing/development. See .cursorrules.", ""
            ).strip()
            
            if prod_content.strip() != sandbox_clean:
                print(f"🔄 Syncing content: {rel_path}")
                
                # Copy Production content to Sandbox
                with open(sandbox_file, 'w') as f:
                    f.write(prod_content)
                
                synced_count += 1
        
        print(f"🔄 Synced content for {synced_count} files")
    
    def add_sandbox_watermarks(self):
        """Add watermarks to Sandbox files"""
        sandbox_files = list(self.sandbox_path.glob("**/*.swift"))
        watermarked_count = 0
        
        watermark = "// SANDBOX FILE: For testing/development. See .cursorrules.\n"
        
        for sandbox_file in sandbox_files:
            with open(sandbox_file, 'r') as f:
                content = f.read()
            
            if watermark not in content:
                print(f"🏷️ Adding watermark: {sandbox_file.relative_to(self.sandbox_path)}")
                
                # Add watermark at the top
                new_content = watermark + content
                
                with open(sandbox_file, 'w') as f:
                    f.write(new_content)
                
                watermarked_count += 1
        
        print(f"🏷️ Added watermarks to {watermarked_count} files")
    
    def verify_alignment(self):
        """Verify that alignment was successful"""
        print("\n🔍 VERIFYING ALIGNMENT")
        print("=" * 30)
        
        # Run the tests again to verify
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestCodebaseAlignment)
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("✅ CODEBASE ALIGNMENT: 100% SUCCESS")
            return True
        else:
            print("❌ CODEBASE ALIGNMENT: STILL ISSUES")
            return False

if __name__ == "__main__":
    print("🚀 ATOMIC TDD CODEBASE ALIGNMENT")
    print("This will fix the 26-file misalignment")
    print()
    
    # Phase 1: RED - Run tests to identify issues
    print("🔴 RED PHASE: Identifying alignment issues...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCodebaseAlignment)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✅ No alignment issues found!")
        sys.exit(0)
    
    print(f"\n❌ Found alignment issues: {len(result.failures)} failures")
    
    # Phase 2: GREEN - Fix the issues
    print("\n🟢 GREEN PHASE: Fixing alignment issues...")
    aligner = CodebaseAligner()
    aligner.fix_alignment()
    
    # Phase 3: REFACTOR - Verify fixes
    print("\n🔵 REFACTOR PHASE: Verifying fixes...")
    success = aligner.verify_alignment()
    
    if success:
        print("\n🎉 CODEBASE ALIGNMENT COMPLETE!")
        print("✅ 100% alignment between Sandbox and Production")
    else:
        print("\n❌ ALIGNMENT ISSUES REMAIN")
        print("🔧 Manual intervention may be required")
    
    sys.exit(0 if success else 1)