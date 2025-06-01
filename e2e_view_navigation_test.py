#!/usr/bin/env python3
"""
Advanced E2E Test for AgenticSeek View Navigation
Tests all available views with comprehensive validation
"""

import asyncio
import subprocess
import time
import json
import httpx
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedViewNavigationTest:
    """Advanced E2E test for navigating through all AgenticSeek views"""
    
    def __init__(self):
        self.app_path = "/Users/bernhardbudiono/Library/Developer/Xcode/DerivedData/AgenticSeek-bdpcvbrzemrwhfcxcrtpdmjthvtb/Build/Products/Debug/AgenticSeek.app"
        self.base_url = "http://localhost:8000"
        self.app_process = None
        self.test_results = []
        self.screenshots_dir = Path("e2e_screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Define all views and their expected elements
        self.views_config = {
            "main": {
                "name": "Main View",
                "navigation_method": "click_main_area",
                "expected_elements": ["Models", "Config", "Settings"],
                "validation_checks": ["webview_loaded", "navigation_buttons_visible"]
            },
            "models": {
                "name": "Models View", 
                "navigation_method": "click_models_button",
                "expected_elements": ["Installed", "Available", "Refresh"],
                "validation_checks": ["model_list_loaded", "tabs_visible", "storage_info"]
            },
            "config": {
                "name": "Configuration View",
                "navigation_method": "click_config_button", 
                "expected_elements": ["Providers", "API Keys", "Settings"],
                "validation_checks": ["provider_list_loaded", "api_key_fields", "configuration_tabs"]
            },
            "settings": {
                "name": "Settings View",
                "navigation_method": "click_settings_button",
                "expected_elements": ["Services", "Configuration", "Actions"],
                "validation_checks": ["service_status", "restart_button", "url_display"]
            }
        }
    
    def run_applescript(self, script: str) -> Dict[str, Any]:
        """Execute AppleScript and return results"""
        try:
            cmd = ["osascript", "-e", script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return {
                "success": result.returncode == 0,
                "output": result.stdout.strip(),
                "error": result.stderr.strip() if result.stderr else None
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "AppleScript timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def launch_app(self) -> Dict[str, Any]:
        """Launch the AgenticSeek app"""
        try:
            if not os.path.exists(self.app_path):
                return {"success": False, "error": f"App not found at {self.app_path}"}
            
            # Kill any existing instances
            self.quit_app()
            time.sleep(2)
            
            # Launch app
            cmd = ["open", self.app_path]
            self.app_process = subprocess.Popen(cmd)
            
            # Wait for app to fully load
            time.sleep(5)
            
            # Verify app is running and responsive
            verify_script = '''
            tell application "System Events"
                if exists (process "AgenticSeek") then
                    tell process "AgenticSeek"
                        set frontmost to true
                        return "running"
                    end tell
                else
                    return "not_running"
                end if
            end tell
            '''
            
            result = self.run_applescript(verify_script)
            is_running = result.get("output") == "running"
            
            # Take initial screenshot
            self.take_screenshot("app_initial_launch")
            
            return {
                "success": is_running,
                "pid": self.app_process.pid if self.app_process else None,
                "status": "launched" if is_running else "failed_to_launch"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def quit_app(self) -> Dict[str, Any]:
        """Quit the AgenticSeek app"""
        try:
            # First try AppleScript
            quit_script = '''
            tell application "AgenticSeek"
                try
                    quit
                end try
            end tell
            
            tell application "System Events"
                try
                    tell process "AgenticSeek"
                        click button 1 of window 1
                    end try
                end try
            end tell
            '''
            
            result = self.run_applescript(quit_script)
            
            # Also terminate process if it exists
            if self.app_process:
                try:
                    self.app_process.terminate()
                    self.app_process.wait(timeout=5)
                except:
                    try:
                        self.app_process.kill()
                    except:
                        pass
            
            # Force kill any remaining processes
            try:
                subprocess.run(["pkill", "-f", "AgenticSeek"], capture_output=True)
            except:
                pass
            
            return {"success": True, "method": "comprehensive_quit"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def take_screenshot(self, name: str) -> Dict[str, Any]:
        """Take a screenshot with organized naming"""
        try:
            timestamp = int(time.time())
            filename = f"{name}_{timestamp}.png"
            filepath = self.screenshots_dir / filename
            
            cmd = ["screencapture", "-w", str(filepath)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            return {
                "success": result.returncode == 0,
                "filename": filename,
                "path": str(filepath.absolute())
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def click_navigation_button(self, button_name: str) -> Dict[str, Any]:
        """Click a navigation button in the app"""
        script = f'''
        tell application "System Events"
            tell process "AgenticSeek"
                try
                    set frontmost to true
                    delay 0.5
                    
                    -- Try to find and click the button by name
                    set buttonFound to false
                    repeat with b in buttons of window 1
                        if name of b is "{button_name}" then
                            click b
                            set buttonFound to true
                            exit repeat
                        end if
                    end repeat
                    
                    -- If not found by name, try by description or accessible description
                    if not buttonFound then
                        repeat with b in buttons of window 1
                            try
                                if (description of b contains "{button_name}") or (help of b contains "{button_name}") then
                                    click b
                                    set buttonFound to true
                                    exit repeat
                                end if
                            end try
                        end repeat
                    end if
                    
                    if buttonFound then
                        return "clicked"
                    else
                        return "button_not_found"
                    end if
                    
                on error errMsg
                    return "error: " & errMsg
                end try
            end tell
        end tell
        '''
        
        result = self.run_applescript(script)
        
        # Add a delay for UI to update
        if result.get("output") == "clicked":
            time.sleep(1.5)
        
        return result
    
    def get_ui_elements(self) -> Dict[str, Any]:
        """Get detailed information about UI elements"""
        script = '''
        tell application "System Events"
            tell process "AgenticSeek"
                try
                    set frontmost to true
                    
                    set windowInfo to {}
                    set buttonInfo to {}
                    set textInfo to {}
                    
                    -- Get window information
                    repeat with w in windows
                        set windowData to {name of w, position of w, size of w}
                        set end of windowInfo to windowData
                    end repeat
                    
                    -- Get button information
                    repeat with b in buttons of window 1
                        try
                            set buttonData to {name of b, position of b, enabled of b}
                            set end of buttonInfo to buttonData
                        end try
                    end repeat
                    
                    -- Get text field information
                    repeat with t in text fields of window 1
                        try
                            set textData to {value of t, position of t}
                            set end of textInfo to textData
                        end try
                    end repeat
                    
                    return "Windows: " & (windowInfo as string) & " | Buttons: " & (buttonInfo as string) & " | TextFields: " & (textInfo as string)
                    
                on error errMsg
                    return "error: " & errMsg
                end try
            end tell
        end tell
        '''
        
        return self.run_applescript(script)
    
    def validate_view_elements(self, view_name: str, expected_elements: List[str]) -> Dict[str, Any]:
        """Validate that expected elements are present in the current view"""
        ui_info = self.get_ui_elements()
        
        if not ui_info["success"]:
            return {"success": False, "error": "Failed to get UI elements"}
        
        ui_output = ui_info.get("output", "")
        found_elements = []
        missing_elements = []
        
        for element in expected_elements:
            if element.lower() in ui_output.lower():
                found_elements.append(element)
            else:
                missing_elements.append(element)
        
        return {
            "success": len(missing_elements) == 0,
            "found_elements": found_elements,
            "missing_elements": missing_elements,
            "ui_output": ui_output,
            "validation_score": len(found_elements) / len(expected_elements) if expected_elements else 1.0
        }
    
    async def validate_backend_integration(self, view_name: str) -> Dict[str, Any]:
        """Validate backend integration for specific views"""
        try:
            if view_name == "models":
                async with httpx.AsyncClient() as client:
                    catalog_response = await client.get(f"{self.base_url}/models/catalog", timeout=10)
                    installed_response = await client.get(f"{self.base_url}/models/installed", timeout=10)
                    storage_response = await client.get(f"{self.base_url}/models/storage", timeout=10)
                    
                    return {
                        "success": all(r.status_code == 200 for r in [catalog_response, installed_response, storage_response]),
                        "catalog_status": catalog_response.status_code,
                        "installed_status": installed_response.status_code,
                        "storage_status": storage_response.status_code,
                        "catalog_data_valid": "catalog" in catalog_response.text if catalog_response.status_code == 200 else False
                    }
                    
            elif view_name == "config":
                async with httpx.AsyncClient() as client:
                    providers_response = await client.get(f"{self.base_url}/config/providers", timeout=10)
                    api_keys_response = await client.get(f"{self.base_url}/config/api-keys", timeout=10)
                    
                    return {
                        "success": all(r.status_code == 200 for r in [providers_response, api_keys_response]),
                        "providers_status": providers_response.status_code,
                        "api_keys_status": api_keys_response.status_code,
                        "providers_data_valid": "providers" in providers_response.text if providers_response.status_code == 200 else False
                    }
                    
            elif view_name == "main":
                async with httpx.AsyncClient() as client:
                    health_response = await client.get(f"{self.base_url}/health", timeout=10)
                    
                    return {
                        "success": health_response.status_code == 200,
                        "health_status": health_response.status_code,
                        "backend_running": "backend" in health_response.text if health_response.status_code == 200 else False
                    }
            
            else:
                return {"success": True, "message": "No specific backend validation for this view"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_view_navigation(self, view_name: str, view_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test navigation to a specific view"""
        start_time = time.time()
        logger.info(f"🧪 Testing navigation to {view_config['name']}")
        
        try:
            # Take screenshot before navigation
            pre_nav_screenshot = self.take_screenshot(f"pre_nav_{view_name}")
            
            # Navigate to the view
            if view_name == "main":
                # For main view, click in the center area or navigate back
                nav_script = '''
                tell application "System Events"
                    tell process "AgenticSeek"
                        try
                            set frontmost to true
                            -- Click in the center of the window to focus main content
                            set windowSize to size of window 1
                            set windowPos to position of window 1
                            set centerX to (item 1 of windowPos) + (item 1 of windowSize) / 2
                            set centerY to (item 2 of windowPos) + (item 2 of windowSize) / 2
                            click at {centerX, centerY}
                            return "main_focused"
                        on error errMsg
                            return "error: " & errMsg
                        end try
                    end tell
                end tell
                '''
                nav_result = self.run_applescript(nav_script)
            else:
                # For other views, click the navigation button
                nav_result = self.click_navigation_button(view_name.title())
            
            if not nav_result["success"]:
                return {
                    "view": view_name,
                    "status": "FAILED",
                    "duration": time.time() - start_time,
                    "error": f"Navigation failed: {nav_result.get('error', 'Unknown error')}",
                    "phase": "navigation"
                }
            
            # Wait for view to load
            await asyncio.sleep(2)
            
            # Take screenshot after navigation
            post_nav_screenshot = self.take_screenshot(f"post_nav_{view_name}")
            
            # Validate UI elements
            ui_validation = self.validate_view_elements(view_name, view_config["expected_elements"])
            
            # Validate backend integration
            backend_validation = await self.validate_backend_integration(view_name)
            
            # Calculate overall success
            navigation_success = nav_result.get("output") in ["clicked", "main_focused"]
            ui_success = ui_validation["success"]
            backend_success = backend_validation["success"]
            
            overall_success = navigation_success and (ui_success or ui_validation["validation_score"] > 0.5)
            
            duration = time.time() - start_time
            
            result = {
                "view": view_name,
                "view_name": view_config["name"],
                "status": "PASSED" if overall_success else "FAILED",
                "duration": duration,
                "navigation_result": nav_result,
                "ui_validation": ui_validation,
                "backend_validation": backend_validation,
                "screenshots": {
                    "pre_navigation": pre_nav_screenshot.get("filename"),
                    "post_navigation": post_nav_screenshot.get("filename")
                },
                "overall_score": {
                    "navigation": navigation_success,
                    "ui_elements": ui_validation["validation_score"],
                    "backend_integration": backend_success
                }
            }
            
            logger.info(f"✅ {view_config['name']}: {'PASSED' if overall_success else 'FAILED'} ({duration:.2f}s)")
            return result
            
        except Exception as e:
            return {
                "view": view_name,
                "view_name": view_config["name"],
                "status": "ERROR",
                "duration": time.time() - start_time,
                "error": str(e),
                "phase": "test_execution"
            }
    
    async def run_complete_navigation_test(self) -> List[Dict[str, Any]]:
        """Run complete navigation test for all views"""
        logger.info("🚀 Starting Complete View Navigation E2E Test")
        results = []
        
        # Launch app
        logger.info("📱 Launching AgenticSeek app...")
        launch_result = self.launch_app()
        
        results.append({
            "view": "app_launch",
            "view_name": "App Launch",
            "status": "PASSED" if launch_result["success"] else "FAILED",
            "duration": 0,
            "launch_result": launch_result
        })
        
        if not launch_result["success"]:
            logger.error("❌ App launch failed, aborting navigation tests")
            return results
        
        # Wait for app to fully initialize
        await asyncio.sleep(3)
        
        # Test each view
        for view_name, view_config in self.views_config.items():
            view_result = await self.test_view_navigation(view_name, view_config)
            results.append(view_result)
            
            # Small delay between view tests
            await asyncio.sleep(1)
        
        # Test view transitions (rapid navigation)
        logger.info("🔄 Testing rapid view transitions...")
        transition_start = time.time()
        
        transition_sequence = ["models", "config", "main", "models", "config"]
        transition_success = True
        
        for view in transition_sequence:
            if view == "main":
                nav_result = self.run_applescript('''
                tell application "System Events"
                    tell process "AgenticSeek"
                        set frontmost to true
                        delay 0.2
                        return "main_focused"
                    end tell
                end tell
                ''')
            else:
                nav_result = self.click_navigation_button(view.title())
            
            if not nav_result["success"]:
                transition_success = False
                break
            
            await asyncio.sleep(0.5)
        
        transition_duration = time.time() - transition_start
        results.append({
            "view": "rapid_transitions",
            "view_name": "Rapid View Transitions",
            "status": "PASSED" if transition_success else "FAILED",
            "duration": transition_duration,
            "transition_sequence": transition_sequence,
            "transitions_per_second": len(transition_sequence) / transition_duration
        })
        
        # Take final screenshot
        final_screenshot = self.take_screenshot("final_state")
        
        # Quit app
        logger.info("🔚 Closing app...")
        quit_result = self.quit_app()
        results.append({
            "view": "app_cleanup",
            "view_name": "App Cleanup",
            "status": "PASSED" if quit_result["success"] else "FAILED",
            "duration": 0,
            "quit_result": quit_result,
            "final_screenshot": final_screenshot.get("filename")
        })
        
        return results
    
    def generate_navigation_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive navigation test report"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.get("status") == "PASSED")
        failed_tests = sum(1 for r in results if r.get("status") == "FAILED")
        error_tests = sum(1 for r in results if r.get("status") == "ERROR")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # View-specific analysis
        view_results = {}
        for result in results:
            view = result.get("view", "unknown")
            if view not in ["app_launch", "app_cleanup", "rapid_transitions"]:
                view_results[view] = {
                    "status": result.get("status"),
                    "duration": result.get("duration", 0),
                    "ui_score": result.get("overall_score", {}).get("ui_elements", 0),
                    "backend_score": result.get("overall_score", {}).get("backend_integration", False),
                    "screenshots": result.get("screenshots", {})
                }
        
        report = {
            "timestamp": time.time(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "success_rate": success_rate
            },
            "view_analysis": view_results,
            "screenshots_directory": str(self.screenshots_dir),
            "detailed_results": results
        }
        
        # Save report
        report_file = Path("e2e_navigation_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


async def main():
    """Main test runner for view navigation"""
    test_runner = AdvancedViewNavigationTest()
    
    # Check prerequisites
    logger.info("🔍 Checking prerequisites...")
    
    # Check if backend is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=5.0)
            if response.status_code != 200:
                logger.error("❌ Backend is not responding. Please start enhanced_backend.py first.")
                return 1
    except:
        logger.error("❌ Backend is not running. Please start enhanced_backend.py first.")
        return 1
    
    logger.info("✅ Backend is running")
    
    # Check if app exists
    if not os.path.exists(test_runner.app_path):
        logger.error(f"❌ App not found at {test_runner.app_path}")
        logger.error("Please build the macOS app first using Xcode")
        return 1
    
    logger.info("✅ App found")
    
    # Run navigation tests
    results = await test_runner.run_complete_navigation_test()
    
    # Generate report
    report = test_runner.generate_navigation_report(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 E2E VIEW NAVIGATION TEST REPORT")
    print("=" * 60)
    print(f"📈 Total Tests: {report['summary']['total_tests']}")
    print(f"✅ Passed: {report['summary']['passed']}")
    print(f"❌ Failed: {report['summary']['failed']}")
    print(f"💥 Errors: {report['summary']['errors']}")
    print(f"🎯 Success Rate: {report['summary']['success_rate']:.1f}%")
    
    print("\n📋 View Test Results:")
    for view, data in report['view_analysis'].items():
        print(f"  {view.title()}: {data['status']} ({data['duration']:.2f}s)")
        print(f"    UI Elements: {data['ui_score']:.1%}")
        print(f"    Backend Integration: {'✅' if data['backend_score'] else '❌'}")
    
    print(f"\n📸 Screenshots saved to: {report['screenshots_directory']}")
    print(f"📄 Detailed report: e2e_navigation_report.json")
    
    success = report['summary']['success_rate'] >= 80.0
    
    if success:
        logger.info("🎉 E2E navigation tests passed!")
        return 0
    else:
        logger.warning("⚠️ Some E2E navigation tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)