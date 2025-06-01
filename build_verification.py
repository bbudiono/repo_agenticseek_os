#!/usr/bin/env python3
"""
Build Verification Script for AgenticSeek
Validates the build integrity after implementing new features
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

def pretty_print(text, color="info"):
    """Simple colored output"""
    colors = {
        "info": "\033[94m",
        "success": "\033[92m", 
        "warning": "\033[93m",
        "failure": "\033[91m",
        "status": "\033[96m"
    }
    reset = "\033[0m"
    print(f"{colors.get(color, '')}{text}{reset}")

class BuildVerification:
    """Comprehensive build verification for AgenticSeek"""
    
    def __init__(self):
        self.results = {
            "core_imports": [],
            "enhanced_features": [],
            "syntax_checks": [],
            "dependency_checks": [],
            "integration_tests": []
        }
        self.errors = []
        
    def verify_syntax(self, file_paths):
        """Verify Python syntax for given files"""
        pretty_print("🔍 Verifying Python syntax...", "status")
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    compile(f.read(), file_path, 'exec')
                self.results["syntax_checks"].append({"file": file_path, "status": "✅ PASS"})
                pretty_print(f"  ✅ {Path(file_path).name}: Syntax OK", "success")
            except SyntaxError as e:
                error_msg = f"Syntax error in {file_path}: {e}"
                self.results["syntax_checks"].append({"file": file_path, "status": f"❌ FAIL: {error_msg}"})
                self.errors.append(error_msg)
                pretty_print(f"  ❌ {Path(file_path).name}: Syntax Error - {e}", "failure")
            except Exception as e:
                error_msg = f"Error checking {file_path}: {e}"
                self.results["syntax_checks"].append({"file": file_path, "status": f"❌ ERROR: {error_msg}"})
                self.errors.append(error_msg)
                pretty_print(f"  ❌ {Path(file_path).name}: Error - {e}", "failure")
    
    def verify_imports(self, modules):
        """Verify module imports with graceful dependency handling"""
        pretty_print("📦 Verifying module imports...", "status")
        
        for module_info in modules:
            module_name = module_info["name"]
            required = module_info.get("required", True)
            
            try:
                # Add current directory to path if not already there
                if "." not in sys.path:
                    sys.path.insert(0, ".")
                
                importlib.import_module(module_name)
                self.results["core_imports"].append({"module": module_name, "status": "✅ PASS"})
                pretty_print(f"  ✅ {module_name}: Import successful", "success")
                
            except ImportError as e:
                status = "❌ FAIL" if required else "⚠️ SKIP"
                color = "failure" if required else "warning"
                
                self.results["core_imports"].append({"module": module_name, "status": f"{status}: {e}"})
                
                if required:
                    self.errors.append(f"Required module {module_name} failed to import: {e}")
                    pretty_print(f"  ❌ {module_name}: Import failed - {e}", "failure")
                else:
                    pretty_print(f"  ⚠️  {module_name}: Optional import skipped - {e}", "warning")
                    
            except Exception as e:
                error_msg = f"Unexpected error importing {module_name}: {e}"
                self.results["core_imports"].append({"module": module_name, "status": f"❌ ERROR: {error_msg}"})
                
                if required:
                    self.errors.append(error_msg)
                    pretty_print(f"  ❌ {module_name}: Unexpected error - {e}", "failure")
                else:
                    pretty_print(f"  ⚠️  {module_name}: Optional error - {e}", "warning")
    
    def verify_enhanced_features(self):
        """Verify new enhanced features are accessible"""
        pretty_print("🚀 Verifying enhanced features...", "status")
        
        # Test enhanced interpreter system
        try:
            from sources.enhanced_interpreter_system import EnhancedInterpreterSystem, LanguageType
            system = EnhancedInterpreterSystem()
            languages = system.get_available_languages()
            self.results["enhanced_features"].append({
                "feature": "Enhanced Interpreter System",
                "status": f"✅ PASS: {len(languages)} languages available"
            })
            pretty_print(f"  ✅ Enhanced Interpreter System: {len(languages)} languages available", "success")
            system.cleanup()
        except Exception as e:
            error_msg = f"Enhanced Interpreter System error: {e}"
            self.results["enhanced_features"].append({
                "feature": "Enhanced Interpreter System",
                "status": f"❌ FAIL: {error_msg}"
            })
            pretty_print(f"  ❌ Enhanced Interpreter System: {e}", "failure")
        
        # Test MCP integration system
        try:
            from sources.mcp_integration_system import MCPIntegrationSystem
            mcp_system = MCPIntegrationSystem()
            status = mcp_system.get_system_status()
            self.results["enhanced_features"].append({
                "feature": "MCP Integration System",
                "status": f"✅ PASS: {status['total_servers_configured']} servers configured"
            })
            pretty_print(f"  ✅ MCP Integration System: {status['total_servers_configured']} servers configured", "success")
            mcp_system.cleanup()
        except Exception as e:
            error_msg = f"MCP Integration System error: {e}"
            self.results["enhanced_features"].append({
                "feature": "MCP Integration System", 
                "status": f"❌ FAIL: {error_msg}"
            })
            pretty_print(f"  ❌ MCP Integration System: {e}", "failure")
        
        # Test unified tool ecosystem
        try:
            from sources.tool_ecosystem_integration import ToolEcosystemIntegration
            ecosystem = ToolEcosystemIntegration()
            tools = ecosystem.get_available_tools()
            self.results["enhanced_features"].append({
                "feature": "Tool Ecosystem Integration",
                "status": f"✅ PASS: {len(tools)} tools available"
            })
            pretty_print(f"  ✅ Tool Ecosystem Integration: {len(tools)} tools available", "success")
            ecosystem.cleanup()
        except Exception as e:
            error_msg = f"Tool Ecosystem Integration error: {e}"
            self.results["enhanced_features"].append({
                "feature": "Tool Ecosystem Integration",
                "status": f"❌ FAIL: {error_msg}"
            })
            pretty_print(f"  ❌ Tool Ecosystem Integration: {e}", "failure")
    
    def verify_dependencies(self):
        """Check critical dependencies"""
        pretty_print("🔗 Checking critical dependencies...", "status")
        
        dependencies = [
            {"name": "asyncio", "required": True},
            {"name": "json", "required": True},
            {"name": "pathlib", "required": True},
            {"name": "dataclasses", "required": True},
            {"name": "enum", "required": True},
            {"name": "typing", "required": True},
            {"name": "torch", "required": False},  # Optional for memory module
            {"name": "selenium", "required": False},  # Optional for browser automation
            {"name": "docker", "required": False},  # Optional for containerization
            {"name": "psutil", "required": False}  # Optional for resource monitoring
        ]
        
        for dep in dependencies:
            try:
                importlib.import_module(dep["name"])
                self.results["dependency_checks"].append({
                    "dependency": dep["name"],
                    "status": "✅ AVAILABLE"
                })
                pretty_print(f"  ✅ {dep['name']}: Available", "success")
            except ImportError:
                status = "❌ MISSING" if dep["required"] else "⚠️ OPTIONAL"
                color = "failure" if dep["required"] else "warning"
                
                self.results["dependency_checks"].append({
                    "dependency": dep["name"],
                    "status": status
                })
                
                if dep["required"]:
                    self.errors.append(f"Required dependency missing: {dep['name']}")
                    pretty_print(f"  ❌ {dep['name']}: Missing (Required)", "failure")
                else:
                    pretty_print(f"  ⚠️  {dep['name']}: Missing (Optional)", "warning")
    
    def run_integration_tests(self):
        """Run basic integration tests"""
        pretty_print("🧪 Running integration tests...", "status")
        
        try:
            # Test that we can run our mock test suite
            import subprocess
            result = subprocess.run([
                sys.executable, "test_tool_ecosystem_integration.py"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.results["integration_tests"].append({
                    "test": "Tool Ecosystem Mock Test Suite",
                    "status": "✅ PASS"
                })
                pretty_print(f"  ✅ Tool Ecosystem Mock Test Suite: PASSED", "success")
            else:
                error_msg = f"Test suite failed with return code {result.returncode}"
                self.results["integration_tests"].append({
                    "test": "Tool Ecosystem Mock Test Suite",
                    "status": f"❌ FAIL: {error_msg}"
                })
                self.errors.append(error_msg)
                pretty_print(f"  ❌ Tool Ecosystem Mock Test Suite: FAILED", "failure")
                
        except subprocess.TimeoutExpired:
            error_msg = "Integration tests timed out"
            self.results["integration_tests"].append({
                "test": "Tool Ecosystem Mock Test Suite",
                "status": f"❌ TIMEOUT: {error_msg}"
            })
            self.errors.append(error_msg)
            pretty_print(f"  ❌ Integration tests: TIMEOUT", "failure")
        except Exception as e:
            error_msg = f"Integration test error: {e}"
            self.results["integration_tests"].append({
                "test": "Tool Ecosystem Mock Test Suite", 
                "status": f"❌ ERROR: {error_msg}"
            })
            self.errors.append(error_msg)
            pretty_print(f"  ❌ Integration tests: ERROR - {e}", "failure")
    
    def generate_report(self):
        """Generate comprehensive build verification report"""
        total_checks = (len(self.results["syntax_checks"]) + 
                       len(self.results["core_imports"]) + 
                       len(self.results["enhanced_features"]) +
                       len(self.results["dependency_checks"]) +
                       len(self.results["integration_tests"]))
        
        total_errors = len(self.errors)
        
        pretty_print("\n" + "="*60, "status")
        pretty_print("📋 BUILD VERIFICATION REPORT", "info")
        pretty_print("="*60, "status")
        
        # Summary
        if total_errors == 0:
            pretty_print("🎉 BUILD STATUS: ✅ PASSING", "success")
        else:
            pretty_print(f"⚠️  BUILD STATUS: ❌ {total_errors} ERRORS FOUND", "failure")
        
        pretty_print(f"Total Checks: {total_checks}", "info")
        pretty_print(f"Total Errors: {total_errors}", "info" if total_errors == 0 else "failure")
        
        # Detailed results
        sections = [
            ("Syntax Checks", self.results["syntax_checks"]),
            ("Module Imports", self.results["core_imports"]),
            ("Enhanced Features", self.results["enhanced_features"]),
            ("Dependencies", self.results["dependency_checks"]),
            ("Integration Tests", self.results["integration_tests"])
        ]
        
        for section_name, section_results in sections:
            if section_results:
                pretty_print(f"\n{section_name}:", "info")
                for result in section_results:
                    key = list(result.keys())[0]
                    status = result["status"]
                    pretty_print(f"  {result[key]}: {status}", "info")
        
        # Errors summary
        if self.errors:
            pretty_print("\n❌ ERRORS FOUND:", "failure")
            for error in self.errors:
                pretty_print(f"  • {error}", "failure")
            
            pretty_print("\n🔧 RECOMMENDED ACTIONS:", "warning")
            pretty_print("  • Fix syntax errors before proceeding", "warning")
            pretty_print("  • Install missing required dependencies", "warning")
            pretty_print("  • Review error messages for integration issues", "warning")
        else:
            pretty_print("\n✅ All critical checks passed!", "success")
            pretty_print("Build is ready for production use.", "success")
        
        return total_errors == 0

def main():
    """Main build verification function"""
    verifier = BuildVerification()
    
    pretty_print("🔍 AgenticSeek Build Verification", "info")
    pretty_print("Checking build integrity after feature implementation...\n", "status")
    
    # Define files to check
    new_files = [
        "sources/enhanced_interpreter_system.py",
        "sources/mcp_integration_system.py", 
        "sources/tool_ecosystem_integration.py",
        "sources/browser_automation_integration.py"
    ]
    
    # Define modules to test
    modules = [
        {"name": "sources.utility", "required": True},
        {"name": "sources.logger", "required": True},
        {"name": "sources.enhanced_interpreter_system", "required": True},
        {"name": "sources.mcp_integration_system", "required": True},
        {"name": "sources.tool_ecosystem_integration", "required": True},
        {"name": "sources.browser_automation_integration", "required": True},
        {"name": "sources.agents.browser_agent", "required": False},  # May have dependency issues
        {"name": "sources.memory", "required": False},  # Requires torch
    ]
    
    # Run verification steps
    verifier.verify_syntax(new_files)
    verifier.verify_imports(modules)
    verifier.verify_dependencies()
    verifier.verify_enhanced_features()
    verifier.run_integration_tests()
    
    # Generate final report
    build_passed = verifier.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if build_passed else 1)

if __name__ == "__main__":
    main()