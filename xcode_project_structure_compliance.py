#!/usr/bin/env python3
"""
XCODE PROJECT STRUCTURE COMPLIANCE VERIFICATION
Ensures proper .xcodeproj files for Sandbox and Production with shared .xcworkspace
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

class XcodeStructureVerifier:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.macos_dir = self.project_root / "_macOS"
        self.verification_results = {
            'timestamp': datetime.now().isoformat(),
            'structure_compliance': {},
            'build_verification': {},
            'summary': {
                'compliant_structure': False,
                'production_builds': False,
                'sandbox_builds': False,
                'workspace_configured': False,
                'critical_issues': []
            }
        }
    
    def verify_directory_structure(self):
        """Verify the required Xcode directory structure"""
        print("📂 VERIFYING XCODE PROJECT STRUCTURE")
        print("=" * 60)
        
        required_structure = {
            'production_project': {
                'path': self.macos_dir / "AgenticSeek.xcodeproj",
                'description': 'Production Xcode project',
                'critical': True
            },
            'sandbox_project': {
                'path': self.macos_dir / "Sandbox-AgenticSeek.xcodeproj", 
                'description': 'Sandbox Xcode project',
                'critical': True
            },
            'shared_workspace': {
                'path': self.macos_dir / "AgenticSeek.xcworkspace",
                'description': 'Shared workspace for both projects',
                'critical': True
            },
            'production_sources': {
                'path': self.macos_dir / "AgenticSeek",
                'description': 'Production source code directory',
                'critical': True
            },
            'sandbox_sources': {
                'path': self.macos_dir / "AgenticSeek-Sandbox",
                'description': 'Sandbox source code directory', 
                'critical': True
            },
            'workspace_contents': {
                'path': self.macos_dir / "AgenticSeek.xcworkspace/contents.xcworkspacedata",
                'description': 'Workspace configuration file',
                'critical': True
            }
        }
        
        all_compliant = True
        
        for component_name, component_info in required_structure.items():
            path = component_info['path']
            description = component_info['description']
            critical = component_info['critical']
            
            exists = path.exists()
            
            status = "✅" if exists else "❌"
            critical_marker = " (CRITICAL)" if critical else ""
            
            print(f"  {status} {description}: {path.name}{critical_marker}")
            
            self.verification_results['structure_compliance'][component_name] = {
                'description': description,
                'path': str(path),
                'exists': exists,
                'critical': critical
            }
            
            if critical and not exists:
                all_compliant = False
                self.verification_results['summary']['critical_issues'].append(
                    f"Missing {description}: {path}"
                )
        
        return all_compliant
    
    def verify_workspace_configuration(self):
        """Verify workspace properly references both projects"""
        print("\n🔧 VERIFYING WORKSPACE CONFIGURATION")
        print("=" * 60)
        
        workspace_file = self.macos_dir / "AgenticSeek.xcworkspace/contents.xcworkspacedata"
        
        if not workspace_file.exists():
            self.log_workspace_result("Workspace File", False, "contents.xcworkspacedata missing")
            return False
        
        try:
            with open(workspace_file, 'r') as f:
                content = f.read()
            
            # Check for required project references
            required_refs = [
                "AgenticSeek.xcodeproj",
                "Sandbox-AgenticSeek.xcodeproj"
            ]
            
            missing_refs = []
            for ref in required_refs:
                if ref not in content:
                    missing_refs.append(ref)
            
            if missing_refs:
                self.log_workspace_result("Project References", False, f"Missing references: {missing_refs}")
                return False
            else:
                self.log_workspace_result("Project References", True, "Both projects properly referenced")
            
            # Check XML structure
            if "<Workspace" in content and "version = \"1.0\"" in content:
                self.log_workspace_result("XML Structure", True, "Valid workspace XML structure")
            else:
                self.log_workspace_result("XML Structure", False, "Invalid workspace XML structure")
                return False
            
            self.verification_results['summary']['workspace_configured'] = True
            return True
            
        except Exception as e:
            self.log_workspace_result("Workspace Reading", False, f"Failed to read workspace: {str(e)}")
            return False
    
    def verify_sandbox_compliance(self):
        """Verify Sandbox project has required SANDBOX markers"""
        print("\n🧪 VERIFYING SANDBOX COMPLIANCE")
        print("=" * 60)
        
        sandbox_dir = self.macos_dir / "AgenticSeek-Sandbox"
        
        if not sandbox_dir.exists():
            self.log_sandbox_result("Sandbox Directory", False, "Sandbox source directory missing")
            return False
        
        # Check for SANDBOX file comments
        swift_files = list(sandbox_dir.glob("*.swift"))
        
        if not swift_files:
            self.log_sandbox_result("Swift Files", False, "No Swift files found in sandbox")
            return False
        
        files_with_sandbox_comment = 0
        total_files = len(swift_files)
        
        for swift_file in swift_files:
            try:
                with open(swift_file, 'r') as f:
                    content = f.read()
                    
                if "// SANDBOX FILE: For testing/development. See .cursorrules." in content:
                    files_with_sandbox_comment += 1
                    
            except Exception as e:
                print(f"    ⚠️ Could not read {swift_file.name}: {str(e)}")
        
        compliance_rate = (files_with_sandbox_comment / total_files) * 100 if total_files > 0 else 0
        
        if compliance_rate >= 80:  # At least 80% of files should have the comment
            self.log_sandbox_result("File Comments", True, f"{files_with_sandbox_comment}/{total_files} files have SANDBOX comments ({compliance_rate:.1f}%)")
        else:
            self.log_sandbox_result("File Comments", False, f"Only {files_with_sandbox_comment}/{total_files} files have SANDBOX comments ({compliance_rate:.1f}%)")
        
        # Check for visible SANDBOX watermarks in UI
        sandbox_components = sandbox_dir / "SandboxComponents.swift"
        if sandbox_components.exists():
            try:
                with open(sandbox_components, 'r') as f:
                    content = f.read()
                
                ui_watermarks = [
                    "🧪 AgenticSeek - SANDBOX",
                    "SANDBOX",
                    "🧪"
                ]
                
                found_watermarks = [w for w in ui_watermarks if w in content]
                
                if len(found_watermarks) >= 2:
                    self.log_sandbox_result("UI Watermarks", True, f"Found {len(found_watermarks)} UI watermarks")
                else:
                    self.log_sandbox_result("UI Watermarks", False, f"Only {len(found_watermarks)} UI watermarks found")
                    
            except Exception as e:
                self.log_sandbox_result("UI Watermarks", False, f"Could not check watermarks: {str(e)}")
        else:
            self.log_sandbox_result("UI Watermarks", False, "SandboxComponents.swift not found")
        
        return compliance_rate >= 80
    
    def verify_production_clean(self):
        """Verify Production project has no SANDBOX markers"""
        print("\n🏭 VERIFYING PRODUCTION CLEANLINESS")
        print("=" * 60)
        
        production_dir = self.macos_dir / "AgenticSeek"
        
        if not production_dir.exists():
            self.log_production_result("Production Directory", False, "Production source directory missing")
            return False
        
        # Check that production files don't have SANDBOX markers
        swift_files = list(production_dir.glob("**/*.swift"))
        
        if not swift_files:
            self.log_production_result("Swift Files", False, "No Swift files found in production")
            return False
        
        files_with_sandbox_markers = 0
        total_files = len(swift_files)
        
        for swift_file in swift_files:
            try:
                with open(swift_file, 'r') as f:
                    content = f.read()
                    
                sandbox_markers = ["// SANDBOX FILE:", "🧪", "SANDBOX"]
                if any(marker in content for marker in sandbox_markers):
                    files_with_sandbox_markers += 1
                    print(f"    ⚠️ Found SANDBOX marker in {swift_file.relative_to(production_dir)}")
                    
            except Exception as e:
                print(f"    ⚠️ Could not read {swift_file.name}: {str(e)}")
        
        if files_with_sandbox_markers == 0:
            self.log_production_result("Clean Production", True, f"No SANDBOX markers found in {total_files} files")
            return True
        else:
            self.log_production_result("Clean Production", False, f"{files_with_sandbox_markers}/{total_files} files contain SANDBOX markers")
            return False
    
    def verify_build_capability(self):
        """Verify both projects can build successfully"""
        print("\n🏗️ VERIFYING BUILD CAPABILITY")
        print("=" * 60)
        
        # Test Production build
        print("  Testing Production build...")
        production_success = self.test_xcode_build("AgenticSeek.xcodeproj", "AgenticSeek")
        
        # Test Sandbox build  
        print("  Testing Sandbox build...")
        sandbox_success = self.test_xcode_build("Sandbox-AgenticSeek.xcodeproj", "AgenticSeek")
        
        self.verification_results['summary']['production_builds'] = production_success
        self.verification_results['summary']['sandbox_builds'] = sandbox_success
        
        return production_success and sandbox_success
    
    def test_xcode_build(self, project_name, scheme_name):
        """Test if an Xcode project can build"""
        try:
            result = subprocess.run([
                'xcodebuild',
                '-project', str(self.macos_dir / project_name),
                '-scheme', scheme_name,
                '-configuration', 'Debug',
                'build'
            ], 
            cwd=self.macos_dir,
            capture_output=True, 
            text=True, 
            timeout=120
            )
            
            if result.returncode == 0:
                self.log_build_result(f"{scheme_name} Build", True, "Build completed successfully")
                return True
            else:
                error_lines = result.stderr.split('\n')[-5:]  # Last 5 lines
                self.log_build_result(f"{scheme_name} Build", False, f"Build failed: {' '.join(error_lines)[:200]}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_build_result(f"{scheme_name} Build", False, "Build timed out after 2 minutes")
            return False
        except Exception as e:
            self.log_build_result(f"{scheme_name} Build", False, f"Build test failed: {str(e)}")
            return False
    
    def log_workspace_result(self, check_name, passed, details):
        """Log workspace verification result"""
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {details}")
        
        if 'workspace' not in self.verification_results['structure_compliance']:
            self.verification_results['structure_compliance']['workspace'] = {}
        
        self.verification_results['structure_compliance']['workspace'][check_name] = {
            'passed': passed,
            'details': details
        }
        
        if not passed:
            self.verification_results['summary']['critical_issues'].append(f"Workspace - {check_name}: {details}")
    
    def log_sandbox_result(self, check_name, passed, details):
        """Log sandbox verification result"""
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {details}")
        
        if 'sandbox' not in self.verification_results['structure_compliance']:
            self.verification_results['structure_compliance']['sandbox'] = {}
        
        self.verification_results['structure_compliance']['sandbox'][check_name] = {
            'passed': passed,
            'details': details
        }
        
        if not passed:
            self.verification_results['summary']['critical_issues'].append(f"Sandbox - {check_name}: {details}")
    
    def log_production_result(self, check_name, passed, details):
        """Log production verification result"""
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {details}")
        
        if 'production' not in self.verification_results['structure_compliance']:
            self.verification_results['structure_compliance']['production'] = {}
        
        self.verification_results['structure_compliance']['production'][check_name] = {
            'passed': passed,
            'details': details
        }
        
        if not passed:
            self.verification_results['summary']['critical_issues'].append(f"Production - {check_name}: {details}")
    
    def log_build_result(self, check_name, passed, details):
        """Log build verification result"""
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {details}")
        
        self.verification_results['build_verification'][check_name] = {
            'passed': passed,
            'details': details
        }
        
        if not passed:
            self.verification_results['summary']['critical_issues'].append(f"Build - {check_name}: {details}")
    
    def generate_compliance_summary(self):
        """Generate comprehensive compliance summary"""
        print("\n" + "=" * 80)
        print("📋 XCODE PROJECT STRUCTURE COMPLIANCE SUMMARY")
        print("=" * 80)
        
        structure_compliant = self.verification_results['summary']['compliant_structure']
        workspace_configured = self.verification_results['summary']['workspace_configured']
        production_builds = self.verification_results['summary']['production_builds']
        sandbox_builds = self.verification_results['summary']['sandbox_builds']
        critical_issues = self.verification_results['summary']['critical_issues']
        
        print(f"📂 Directory Structure: {'✅ COMPLIANT' if structure_compliant else '❌ NON-COMPLIANT'}")
        print(f"🔧 Workspace Configuration: {'✅ CONFIGURED' if workspace_configured else '❌ NOT CONFIGURED'}")
        print(f"🏭 Production Build: {'✅ BUILDS' if production_builds else '❌ FAILS'}")
        print(f"🧪 Sandbox Build: {'✅ BUILDS' if sandbox_builds else '❌ FAILS'}")
        
        if critical_issues:
            print(f"\n🚨 Critical Issues ({len(critical_issues)}):")
            for i, issue in enumerate(critical_issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\n✅ No critical issues found")
        
        # Overall compliance assessment
        overall_compliant = (
            structure_compliant and 
            workspace_configured and 
            production_builds and 
            sandbox_builds and
            len(critical_issues) == 0
        )
        
        print(f"\n🎯 OVERALL COMPLIANCE: {'✅ FULLY COMPLIANT' if overall_compliant else '❌ NEEDS FIXES'}")
        
        if overall_compliant:
            print("\n🎉 XCODE PROJECT STRUCTURE VERIFICATION PASSED")
            print("✅ Both projects properly configured with shared workspace")
            print("✅ Ready for development and TestFlight deployment")
        else:
            print("\n🛠️ XCODE PROJECT STRUCTURE NEEDS ATTENTION")
            print("❌ Fix critical issues before proceeding with development")
        
        # Save detailed report
        report_file = self.project_root / f"xcode_structure_compliance_report_{int(datetime.now().timestamp())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.verification_results, f, indent=2)
        
        print(f"\n📄 Detailed report: {report_file}")
        
        return overall_compliant
    
    def run_comprehensive_verification(self):
        """Run complete Xcode structure compliance verification"""
        print("🔍 COMPREHENSIVE XCODE PROJECT STRUCTURE COMPLIANCE")
        print("=" * 80)
        print("Verifying .cursorrules compliance: 1 Prod .xcodeproj + 1 Sandbox .xcodeproj + 1 shared .xcworkspace")
        print("=" * 80)
        
        # Run all verifications
        structure_ok = self.verify_directory_structure()
        workspace_ok = self.verify_workspace_configuration()
        sandbox_ok = self.verify_sandbox_compliance()
        production_ok = self.verify_production_clean()
        builds_ok = self.verify_build_capability()
        
        # Update summary
        self.verification_results['summary']['compliant_structure'] = structure_ok
        self.verification_results['summary']['workspace_configured'] = workspace_ok
        
        # Generate summary
        overall_compliant = self.generate_compliance_summary()
        
        return overall_compliant

def main():
    """Main Xcode structure verification function"""
    verifier = XcodeStructureVerifier()
    
    try:
        success = verifier.run_comprehensive_verification()
        
        if success:
            print("\n🎉 XCODE STRUCTURE COMPLIANCE VERIFIED")
            print("✅ Project structure follows .cursorrules requirements")
            return 0
        else:
            print("\n🛠️ XCODE STRUCTURE COMPLIANCE FAILED")
            print("❌ Project structure needs fixes to meet .cursorrules requirements")
            return 1
            
    except Exception as e:
        print(f"\n💥 VERIFICATION CRASHED: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())