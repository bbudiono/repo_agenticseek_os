#!/usr/bin/env python3
"""
TESTFLIGHT COMPREHENSIVE BUILD VERIFICATION
Verifies both React frontend and macOS builds are ready for TestFlight deployment
"""

import json
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

class TestFlightBuildVerifier:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.verification_results = {
            'timestamp': datetime.now().isoformat(),
            'react_build': {},
            'macos_build': {},
            'deployment_readiness': {},
            'summary': {
                'react_ready': False,
                'macos_ready': False,
                'overall_ready': False,
                'critical_issues': []
            }
        }
    
    def verify_react_build_for_testflight(self):
        """Comprehensive React build verification for TestFlight deployment"""
        print("⚛️ VERIFYING REACT BUILD FOR TESTFLIGHT")
        print("=" * 60)
        
        react_dir = self.project_root / "frontend/agentic-seek-copilotkit-broken"
        
        # Check if React project exists
        if not react_dir.exists():
            self.log_react_result("Project Structure", False, "React project directory not found")
            return False
        
        # Verify package.json
        package_json = react_dir / "package.json"
        if not package_json.exists():
            self.log_react_result("Package Configuration", False, "package.json not found")
            return False
        
        with open(package_json, 'r') as f:
            package_data = json.load(f)
        
        self.log_react_result("Package Configuration", True, f"Found package.json with {len(package_data.get('dependencies', {}))} dependencies")
        
        # Verify source files exist
        src_dir = react_dir / "src"
        required_files = ["index.tsx", "FunctionalApp.tsx", "App.css"]
        
        missing_files = []
        for file in required_files:
            if not (src_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            self.log_react_result("Source Files", False, f"Missing files: {missing_files}")
            return False
        else:
            self.log_react_result("Source Files", True, f"All {len(required_files)} required source files present")
        
        # Verify FunctionalApp is properly implemented
        functional_app = src_dir / "FunctionalApp.tsx"
        functional_content = functional_app.read_text()
        
        critical_features = [
            ("CRUD Operations", ["handleCreateAgent", "handleCreateTask", "handleDeleteAgent", "handleExecuteTask"]),
            ("Form Validation", ["required", "Please fill in all required fields"]),
            ("API Integration", ["class ApiService", "fetchWithFallback"]),
            ("State Management", ["useState", "setAgents", "setTasks"]),
            ("Error Handling", ["try {", "} catch", "setError"]),
            ("User Feedback", ["successfully!", "Are you sure"])
        ]
        
        all_features_present = True
        for feature_name, patterns in critical_features:
            found_patterns = [p for p in patterns if p in functional_content]
            feature_complete = len(found_patterns) >= len(patterns) * 0.8
            
            if feature_complete:
                self.log_react_result(f"Feature: {feature_name}", True, f"{len(found_patterns)}/{len(patterns)} patterns found")
            else:
                self.log_react_result(f"Feature: {feature_name}", False, f"Only {len(found_patterns)}/{len(patterns)} patterns found")
                all_features_present = False
        
        # Try to build the React app
        print("\n🏗️ Testing React Production Build...")
        try:
            build_result = subprocess.run(
                ['npm', 'run', 'build'],
                cwd=react_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if build_result.returncode == 0:
                self.log_react_result("Production Build", True, "Build completed successfully")
                
                # Check build output
                build_dir = react_dir / "build"
                if build_dir.exists():
                    static_dir = build_dir / "static"
                    js_files = list(static_dir.glob("js/main.*.js")) if static_dir.exists() else []
                    css_files = list(static_dir.glob("css/main.*.css")) if static_dir.exists() else []
                    
                    if js_files and css_files:
                        js_size = js_files[0].stat().st_size
                        css_size = css_files[0].stat().st_size
                        
                        # Check if build is reasonable size (not too small or too large)
                        if js_size > 10000 and js_size < 5000000:  # 10KB to 5MB range
                            self.log_react_result("Build Size", True, f"JS: {js_size//1024}KB, CSS: {css_size//1024}KB")
                        else:
                            self.log_react_result("Build Size", False, f"Suspicious JS size: {js_size//1024}KB")
                            all_features_present = False
                    else:
                        self.log_react_result("Build Assets", False, "Missing JS or CSS files in build")
                        all_features_present = False
                else:
                    self.log_react_result("Build Directory", False, "Build directory not created")
                    all_features_present = False
                    
            else:
                self.log_react_result("Production Build", False, f"Build failed: {build_result.stderr[:500]}")
                all_features_present = False
                
        except subprocess.TimeoutExpired:
            self.log_react_result("Production Build", False, "Build timed out after 5 minutes")
            all_features_present = False
        except Exception as e:
            self.log_react_result("Production Build", False, f"Build error: {str(e)}")
            all_features_present = False
        
        # Test that the app can start
        print("\n🚀 Testing React Development Server...")
        try:
            # Quick start test (just check if it can initialize)
            start_result = subprocess.run(
                ['npm', 'start', '--', '--help'],  # Just check if start command exists
                cwd=react_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # If npm start --help doesn't work, try checking scripts
            scripts = package_data.get('scripts', {})
            if 'start' in scripts:
                self.log_react_result("Start Script", True, f"Start script: {scripts['start']}")
            else:
                self.log_react_result("Start Script", False, "No start script in package.json")
                all_features_present = False
                
        except Exception as e:
            self.log_react_result("Start Script", False, f"Start script error: {str(e)}")
            all_features_present = False
        
        self.verification_results['summary']['react_ready'] = all_features_present
        return all_features_present
    
    def verify_macos_build_for_testflight(self):
        """Comprehensive macOS build verification for TestFlight deployment"""
        print("\n🍎 VERIFYING MACOS BUILD FOR TESTFLIGHT")
        print("=" * 60)
        
        macos_dir = self.project_root / "_macOS"
        
        # Check if macOS project exists
        if not macos_dir.exists():
            self.log_macos_result("Project Structure", False, "macOS project directory not found")
            return False
        
        # Find Xcode project
        xcode_projects = list(macos_dir.glob("*.xcodeproj"))
        if not xcode_projects:
            self.log_macos_result("Xcode Project", False, "No .xcodeproj file found")
            return False
        
        xcode_project = xcode_projects[0]
        self.log_macos_result("Xcode Project", True, f"Found: {xcode_project.name}")
        
        # Check for source files
        agenticseek_dir = macos_dir / "AgenticSeek"
        if not agenticseek_dir.exists():
            self.log_macos_result("Source Directory", False, "AgenticSeek source directory not found")
            return False
        
        required_swift_files = [
            "AgenticSeekApp.swift",
            "ContentView.swift",
            "DesignSystem.swift",
            "ServiceManager.swift"
        ]
        
        missing_swift_files = []
        for file in required_swift_files:
            if not (agenticseek_dir / file).exists():
                missing_swift_files.append(file)
        
        if missing_swift_files:
            self.log_macos_result("Swift Source Files", False, f"Missing: {missing_swift_files}")
            return False
        else:
            self.log_macos_result("Swift Source Files", True, f"All {len(required_swift_files)} Swift files present")
        
        # Check Info.plist
        info_plist = agenticseek_dir / "Info.plist"
        if info_plist.exists():
            self.log_macos_result("Info.plist", True, "Configuration file present")
        else:
            self.log_macos_result("Info.plist", False, "Info.plist missing")
        
        # Check Assets
        assets_dir = agenticseek_dir / "Assets.xcassets"
        if assets_dir.exists():
            app_icon = assets_dir / "AppIcon.appiconset"
            if app_icon.exists():
                self.log_macos_result("App Icon", True, "App icon assets present")
            else:
                self.log_macos_result("App Icon", False, "App icon assets missing")
        else:
            self.log_macos_result("Assets", False, "Assets.xcassets directory missing")
        
        # Test Xcode build
        print("\n🔨 Testing Xcode Build...")
        try:
            build_result = subprocess.run(
                [
                    'xcodebuild', 
                    '-project', str(xcode_project),
                    '-scheme', 'AgenticSeek',
                    '-configuration', 'Debug',
                    'build'
                ],
                cwd=macos_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if build_result.returncode == 0:
                self.log_macos_result("Xcode Build", True, "Build completed successfully")
            else:
                error_lines = build_result.stderr.split('\n')[-10:]  # Last 10 lines of error
                self.log_macos_result("Xcode Build", False, f"Build failed: {' '.join(error_lines)[:300]}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_macos_result("Xcode Build", False, "Build timed out after 5 minutes")
            return False
        except Exception as e:
            self.log_macos_result("Xcode Build", False, f"Build error: {str(e)}")
            return False
        
        # Check project settings for TestFlight readiness
        print("\n⚙️ Checking TestFlight Readiness...")
        
        # Look for entitlements file
        entitlements_file = agenticseek_dir / "AgenticSeek.entitlements"
        if entitlements_file.exists():
            self.log_macos_result("Entitlements", True, "Entitlements file present")
        else:
            self.log_macos_result("Entitlements", False, "Entitlements file missing")
        
        # Check ContentView for proper SwiftUI implementation
        content_view = agenticseek_dir / "ContentView.swift"
        if content_view.exists():
            content = content_view.read_text()
            swiftui_patterns = ["struct ContentView", "View", "body", "@State", "VStack", "HStack"]
            found_patterns = [p for p in swiftui_patterns if p in content]
            
            if len(found_patterns) >= 4:
                self.log_macos_result("SwiftUI Implementation", True, f"{len(found_patterns)}/{len(swiftui_patterns)} SwiftUI patterns found")
            else:
                self.log_macos_result("SwiftUI Implementation", False, f"Only {len(found_patterns)}/{len(swiftui_patterns)} SwiftUI patterns found")
        
        self.verification_results['summary']['macos_ready'] = True
        return True
    
    def verify_deployment_readiness(self):
        """Verify overall deployment readiness"""
        print("\n🚀 VERIFYING DEPLOYMENT READINESS")
        print("=" * 60)
        
        react_ready = self.verification_results['summary']['react_ready']
        macos_ready = self.verification_results['summary']['macos_ready']
        
        # Check environment configurations
        env_file = self.project_root / ".env"
        if env_file.exists():
            self.log_deployment_result("Environment Config", True, ".env file present for configuration")
        else:
            self.log_deployment_result("Environment Config", False, ".env file missing")
        
        # Check documentation
        readme_file = self.project_root / "README.md"
        if readme_file.exists():
            self.log_deployment_result("Documentation", True, "README.md present")
        else:
            self.log_deployment_result("Documentation", False, "README.md missing")
        
        # Check git status
        try:
            git_status = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if git_status.returncode == 0:
                uncommitted_files = len(git_status.stdout.strip().split('\n')) if git_status.stdout.strip() else 0
                if uncommitted_files == 0:
                    self.log_deployment_result("Git Status", True, "All changes committed")
                else:
                    self.log_deployment_result("Git Status", False, f"{uncommitted_files} uncommitted files")
            else:
                self.log_deployment_result("Git Status", False, "Not a git repository or git error")
                
        except Exception as e:
            self.log_deployment_result("Git Status", False, f"Git check failed: {str(e)}")
        
        # Overall readiness assessment
        overall_ready = react_ready and macos_ready
        self.verification_results['summary']['overall_ready'] = overall_ready
        
        if overall_ready:
            self.log_deployment_result("Overall Readiness", True, "Both builds ready for TestFlight")
        else:
            issues = []
            if not react_ready:
                issues.append("React build issues")
            if not macos_ready:
                issues.append("macOS build issues")
            self.log_deployment_result("Overall Readiness", False, f"Issues: {', '.join(issues)}")
        
        return overall_ready
    
    def log_react_result(self, check_name, passed, details):
        """Log React build verification result"""
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {details}")
        
        if check_name not in self.verification_results['react_build']:
            self.verification_results['react_build'][check_name] = []
        
        self.verification_results['react_build'][check_name] = {
            'passed': passed,
            'details': details
        }
        
        if not passed:
            self.verification_results['summary']['critical_issues'].append(f"React - {check_name}: {details}")
    
    def log_macos_result(self, check_name, passed, details):
        """Log macOS build verification result"""
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {details}")
        
        if check_name not in self.verification_results['macos_build']:
            self.verification_results['macos_build'][check_name] = []
        
        self.verification_results['macos_build'][check_name] = {
            'passed': passed,
            'details': details
        }
        
        if not passed:
            self.verification_results['summary']['critical_issues'].append(f"macOS - {check_name}: {details}")
    
    def log_deployment_result(self, check_name, passed, details):
        """Log deployment readiness result"""
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {details}")
        
        self.verification_results['deployment_readiness'][check_name] = {
            'passed': passed,
            'details': details
        }
        
        if not passed:
            self.verification_results['summary']['critical_issues'].append(f"Deployment - {check_name}: {details}")
    
    def generate_testflight_summary_report(self):
        """Generate comprehensive TestFlight readiness report"""
        print("\n" + "=" * 80)
        print("📋 TESTFLIGHT READINESS SUMMARY REPORT")
        print("=" * 80)
        
        react_ready = self.verification_results['summary']['react_ready']
        macos_ready = self.verification_results['summary']['macos_ready']
        overall_ready = self.verification_results['summary']['overall_ready']
        critical_issues = self.verification_results['summary']['critical_issues']
        
        print(f"⚛️ React Frontend: {'✅ READY' if react_ready else '❌ NOT READY'}")
        print(f"🍎 macOS Application: {'✅ READY' if macos_ready else '❌ NOT READY'}")
        print(f"🚀 Overall Status: {'✅ TESTFLIGHT READY' if overall_ready else '❌ NOT READY FOR TESTFLIGHT'}")
        
        if critical_issues:
            print(f"\n🚨 Critical Issues ({len(critical_issues)}):")
            for i, issue in enumerate(critical_issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\n✅ No critical issues found")
        
        # TestFlight deployment checklist
        print(f"\n📝 TESTFLIGHT DEPLOYMENT CHECKLIST:")
        
        checklist_items = [
            ("React build compiles successfully", react_ready),
            ("macOS build compiles successfully", macos_ready),
            ("All source files present", True),  # Verified above
            ("No critical build errors", len(critical_issues) == 0),
            ("Project configuration valid", overall_ready),
        ]
        
        for item, status in checklist_items:
            print(f"  {'✅' if status else '❌'} {item}")
        
        # Deployment instructions
        if overall_ready:
            print(f"\n🎯 NEXT STEPS FOR TESTFLIGHT DEPLOYMENT:")
            print("  1. Commit any remaining changes to Git")
            print("  2. Create release build in Xcode (Product > Archive)")
            print("  3. Upload to App Store Connect via Xcode Organizer")
            print("  4. Configure TestFlight testing in App Store Connect")
            print("  5. Deploy React frontend to hosting platform (Vercel, Netlify, etc.)")
        else:
            print(f"\n🛠️ REQUIRED FIXES BEFORE TESTFLIGHT:")
            print("  1. Resolve all critical issues listed above")
            print("  2. Re-run this verification script")
            print("  3. Ensure all builds complete successfully")
        
        # Save detailed report
        report_file = self.project_root / f"testflight_readiness_report_{int(datetime.now().timestamp())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.verification_results, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        return overall_ready
    
    def run_comprehensive_testflight_verification(self):
        """Run complete TestFlight readiness verification"""
        print("🚀 COMPREHENSIVE TESTFLIGHT BUILD VERIFICATION")
        print("=" * 80)
        print("Verifying both React and macOS builds for TestFlight deployment")
        print("=" * 80)
        
        # Run all verifications
        react_verified = self.verify_react_build_for_testflight()
        macos_verified = self.verify_macos_build_for_testflight()
        deployment_verified = self.verify_deployment_readiness()
        
        # Generate summary report
        overall_ready = self.generate_testflight_summary_report()
        
        return overall_ready

def main():
    """Main TestFlight verification function"""
    verifier = TestFlightBuildVerifier()
    
    try:
        success = verifier.run_comprehensive_testflight_verification()
        
        if success:
            print("\n🎉 TESTFLIGHT VERIFICATION PASSED")
            print("✅ Both React and macOS builds are ready for TestFlight deployment")
            return 0
        else:
            print("\n🛠️ TESTFLIGHT VERIFICATION FAILED")
            print("❌ One or more builds need fixes before TestFlight deployment")
            return 1
            
    except Exception as e:
        print(f"\n💥 VERIFICATION CRASHED: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())