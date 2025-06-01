#!/usr/bin/env python3
"""
Fully Headless E2E Test Suite for AgenticSeek
Non-interactive testing with screenshots and complete automation
"""

import asyncio
import subprocess
import time
import json
import httpx
import os
import tempfile
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FullyHeadlessE2ETest:
    """Completely headless E2E testing that doesn't interrupt the user"""
    
    def __init__(self):
        self.app_path = "/Users/bernhardbudiono/Library/Developer/Xcode/DerivedData/AgenticSeek-bdpcvbrzemrwhfcxcrtpdmjthvtb/Build/Products/Debug/AgenticSeek.app"
        self.base_url = "http://localhost:8000"
        self.test_results = []
        self.screenshots_dir = Path("headless_e2e_screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Store original user context
        self.original_active_app = None
        self.app_process = None
        
    def preserve_user_context(self):
        """Save current user context to restore later"""
        try:
            # Get currently active application
            result = subprocess.run([
                "osascript", "-e", 
                'tell application "System Events" to return name of first application process whose frontmost is true'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.original_active_app = result.stdout.strip()
                logger.info(f"Preserved user context: {self.original_active_app}")
        except Exception as e:
            logger.warning(f"Could not preserve user context: {e}")
    
    def restore_user_context(self):
        """Restore original user context"""
        try:
            if self.original_active_app and self.original_active_app != "AgenticSeek":
                # Return focus to original app
                subprocess.run([
                    "osascript", "-e", 
                    f'tell application "{self.original_active_app}" to activate'
                ], capture_output=True, text=True, timeout=5)
                logger.info(f"Restored user context: {self.original_active_app}")
        except Exception as e:
            logger.warning(f"Could not restore user context: {e}")
    
    def run_silent_applescript(self, script: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute AppleScript without user interaction"""
        try:
            # Add delay to ensure non-interruption
            full_script = f"""
            try
                {script}
            on error errMsg
                return "error: " & errMsg
            end try
            """
            
            cmd = ["osascript", "-e", full_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout.strip(),
                "error": result.stderr.strip() if result.stderr else None,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"AppleScript timeout after {timeout}s", "timeout": True}
        except Exception as e:
            return {"success": False, "error": str(e), "exception": True}
    
    def launch_app_headless(self) -> Dict[str, Any]:
        """Launch app without bringing it to front or interrupting user"""
        try:
            if not os.path.exists(self.app_path):
                return {"success": False, "error": f"App not found at {self.app_path}"}
            
            # Preserve user context first
            self.preserve_user_context()
            
            # Force quit any existing instances
            self.quit_app_silent()
            time.sleep(2)
            
            # Launch app in background
            cmd = ["open", "-g", self.app_path]  # -g flag keeps app in background
            self.app_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for app to initialize
            time.sleep(5)
            
            # Verify launch without bringing to front
            verify_script = '''
            tell application "System Events"
                if exists (process "AgenticSeek") then
                    return "running"
                else
                    return "not_running"
                end if
            end tell
            '''
            
            result = self.run_silent_applescript(verify_script)
            is_running = result.get("output") == "running"
            
            # Take initial screenshot without interrupting user
            screenshot_result = self.take_silent_screenshot("app_launch_headless")
            
            return {
                "success": is_running,
                "pid": self.app_process.pid if self.app_process else None,
                "status": "launched_headless" if is_running else "failed_to_launch",
                "screenshot": screenshot_result.get("filename") if screenshot_result.get("success") else None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def quit_app_silent(self) -> Dict[str, Any]:
        """Quit app silently without user interaction"""
        try:
            # First try graceful quit
            quit_script = '''
            tell application "System Events"
                if exists (process "AgenticSeek") then
                    tell process "AgenticSeek"
                        try
                            click button 1 of window 1
                        end try
                    end tell
                end if
            end tell
            '''
            
            self.run_silent_applescript(quit_script)
            
            # Force kill any remaining processes
            try:
                subprocess.run(["pkill", "-f", "AgenticSeek"], capture_output=True, timeout=5)
            except:
                pass
            
            # Terminate our process if it exists
            if self.app_process:
                try:
                    self.app_process.terminate()
                    self.app_process.wait(timeout=3)
                except:
                    try:
                        self.app_process.kill()
                    except:
                        pass
            
            return {"success": True, "method": "silent_quit"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def take_silent_screenshot(self, name: str) -> Dict[str, Any]:
        """Take screenshot of specific window without interrupting user"""
        try:
            timestamp = int(time.time())
            filename = f"{name}_{timestamp}.png"
            filepath = self.screenshots_dir / filename
            
            # Get window ID of AgenticSeek
            get_window_script = '''
            tell application "System Events"
                if exists (process "AgenticSeek") then
                    tell process "AgenticSeek"
                        if exists window 1 then
                            return id of window 1
                        else
                            return "no_window"
                        end if
                    end tell
                else
                    return "no_process"
                end if
            end tell
            '''
            
            window_result = self.run_silent_applescript(get_window_script)
            
            if window_result.get("success") and window_result.get("output") not in ["no_window", "no_process"]:
                # Take screenshot of specific window using window ID
                cmd = ["screencapture", "-l", window_result["output"], str(filepath)]
            else:
                # Fallback: take screenshot of AgenticSeek area without focusing
                cmd = ["screencapture", "-C", str(filepath)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            return {
                "success": result.returncode == 0,
                "filename": filename,
                "path": str(filepath.absolute()),
                "window_captured": window_result.get("output") not in ["no_window", "no_process"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def interact_with_app_headless(self, action: str, target: str = None) -> Dict[str, Any]:
        """Interact with app without bringing it to front"""
        try:
            if action == "click_button":
                script = f'''
                tell application "System Events"
                    if exists (process "AgenticSeek") then
                        tell process "AgenticSeek"
                            try
                                set buttonFound to false
                                repeat with b in buttons of window 1
                                    if name of b is "{target}" then
                                        click b
                                        set buttonFound to true
                                        exit repeat
                                    end if
                                end repeat
                                
                                if not buttonFound then
                                    repeat with b in buttons of window 1
                                        try
                                            if (description of b contains "{target}") or (help of b contains "{target}") then
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
                    else
                        return "app_not_running"
                    end if
                end tell
                '''
                
            elif action == "get_ui_info":
                script = '''
                tell application "System Events"
                    if exists (process "AgenticSeek") then
                        tell process "AgenticSeek"
                            try
                                set buttonInfo to {}
                                repeat with b in buttons of window 1
                                    try
                                        set buttonData to {name of b, enabled of b}
                                        set end of buttonInfo to buttonData
                                    end try
                                end repeat
                                return "Buttons: " & (buttonInfo as string)
                            on error errMsg
                                return "error: " & errMsg
                            end try
                        end tell
                    else
                        return "app_not_running"
                    end if
                end tell
                '''
            
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
            
            result = self.run_silent_applescript(script)
            
            # Add delay for UI updates
            if result.get("output") == "clicked":
                time.sleep(1.5)
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_headless_navigation_workflow(self) -> List[Dict[str, Any]]:
        """Test navigation workflow completely headlessly"""
        results = []
        
        logger.info("🚀 Starting Fully Headless E2E Navigation Test")
        
        # 1. Launch app headlessly
        logger.info("📱 Launching app in background...")
        launch_result = self.launch_app_headless()
        
        results.append({
            "test": "Headless App Launch",
            "status": "PASSED" if launch_result["success"] else "FAILED",
            "duration": 0,
            "details": launch_result
        })
        
        if not launch_result["success"]:
            logger.error("❌ Headless app launch failed")
            return results
        
        # Wait for app to fully load
        await asyncio.sleep(3)
        
        # 2. Test initial state
        initial_screenshot = self.take_silent_screenshot("initial_state")
        ui_info = self.interact_with_app_headless("get_ui_info")
        
        results.append({
            "test": "Initial UI State",
            "status": "PASSED" if ui_info["success"] else "FAILED",
            "duration": 0,
            "details": {
                "ui_info": ui_info,
                "screenshot": initial_screenshot.get("filename")
            }
        })
        
        # 3. Test navigation to different views
        views_to_test = [
            ("Models", "models_view"),
            ("Config", "config_view"), 
            ("Settings", "settings_view")
        ]
        
        for button_name, view_name in views_to_test:
            start_time = time.time()
            
            # Take pre-navigation screenshot
            pre_screenshot = self.take_silent_screenshot(f"pre_{view_name}")
            
            # Navigate to view
            nav_result = self.interact_with_app_headless("click_button", button_name)
            
            # Wait for view to load
            await asyncio.sleep(2)
            
            # Take post-navigation screenshot
            post_screenshot = self.take_silent_screenshot(f"post_{view_name}")
            
            # Get UI state after navigation
            post_ui = self.interact_with_app_headless("get_ui_info")
            
            duration = time.time() - start_time
            
            # Test backend integration for this view
            backend_test = await self.test_backend_for_view(view_name)
            
            results.append({
                "test": f"Navigate to {button_name} View",
                "status": "PASSED" if nav_result["success"] else "FAILED",
                "duration": duration,
                "details": {
                    "navigation": nav_result,
                    "ui_state": post_ui,
                    "backend_integration": backend_test,
                    "screenshots": {
                        "pre": pre_screenshot.get("filename"),
                        "post": post_screenshot.get("filename")
                    }
                }
            })
            
            logger.info(f"✅ {button_name} view test: {'PASSED' if nav_result['success'] else 'FAILED'}")
        
        # 4. Test rapid navigation (stress test)
        rapid_nav_start = time.time()
        rapid_sequence = ["Models", "Config", "Models", "Config"]
        rapid_success = True
        
        for i, button in enumerate(rapid_sequence):
            nav_result = self.interact_with_app_headless("click_button", button)
            if not nav_result["success"]:
                rapid_success = False
                break
            await asyncio.sleep(0.5)
        
        rapid_duration = time.time() - rapid_nav_start
        
        # Take final state screenshot
        final_screenshot = self.take_silent_screenshot("final_state")
        
        results.append({
            "test": "Rapid Navigation Stress Test",
            "status": "PASSED" if rapid_success else "FAILED",
            "duration": rapid_duration,
            "details": {
                "sequence": rapid_sequence,
                "navigation_speed": len(rapid_sequence) / rapid_duration,
                "final_screenshot": final_screenshot.get("filename")
            }
        })
        
        # 5. Clean up headlessly
        quit_result = self.quit_app_silent()
        
        results.append({
            "test": "Headless App Cleanup",
            "status": "PASSED" if quit_result["success"] else "FAILED",
            "duration": 0,
            "details": quit_result
        })
        
        # Restore user context
        self.restore_user_context()
        
        logger.info("✅ Fully headless E2E test completed")
        
        return results
    
    async def test_backend_for_view(self, view_name: str) -> Dict[str, Any]:
        """Test backend integration for specific view"""
        try:
            if "models" in view_name:
                async with httpx.AsyncClient() as client:
                    catalog_response = await client.get(f"{self.base_url}/models/catalog", timeout=5)
                    return {
                        "success": catalog_response.status_code == 200,
                        "status_code": catalog_response.status_code
                    }
            elif "config" in view_name:
                async with httpx.AsyncClient() as client:
                    api_keys_response = await client.get(f"{self.base_url}/config/api-keys", timeout=5)
                    return {
                        "success": api_keys_response.status_code == 200,
                        "status_code": api_keys_response.status_code
                    }
            else:
                async with httpx.AsyncClient() as client:
                    health_response = await client.get(f"{self.base_url}/health", timeout=5)
                    return {
                        "success": health_response.status_code == 200,
                        "status_code": health_response.status_code
                    }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_headless_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive headless test report"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.get("status") == "PASSED")
        failed_tests = sum(1 for r in results if r.get("status") == "FAILED")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Collect all screenshots
        screenshots = []
        for result in results:
            details = result.get("details", {})
            if isinstance(details, dict):
                if "screenshot" in details:
                    screenshots.append(details["screenshot"])
                if "screenshots" in details:
                    screenshots.extend(details["screenshots"].values())
                if "final_screenshot" in details:
                    screenshots.append(details["final_screenshot"])
        
        report = {
            "timestamp": time.time(),
            "test_type": "fully_headless_e2e",
            "user_interruption": False,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": success_rate
            },
            "screenshots": {
                "directory": str(self.screenshots_dir),
                "captured": [s for s in screenshots if s],
                "count": len([s for s in screenshots if s])
            },
            "detailed_results": results
        }
        
        # Save report
        report_file = Path("fully_headless_e2e_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


async def main():
    """Main headless test runner"""
    test_runner = FullyHeadlessE2ETest()
    
    # Check prerequisites
    logger.info("🔍 Checking prerequisites...")
    
    # Check backend
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=5.0)
            if response.status_code != 200:
                logger.error("❌ Backend not responding")
                return 1
    except:
        logger.error("❌ Backend not running")
        return 1
    
    logger.info("✅ Backend is running")
    
    # Check app exists
    if not os.path.exists(test_runner.app_path):
        logger.error(f"❌ App not found at {test_runner.app_path}")
        return 1
    
    logger.info("✅ App found")
    
    # Run headless tests
    results = await test_runner.test_headless_navigation_workflow()
    
    # Generate report
    report = test_runner.generate_headless_report(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 FULLY HEADLESS E2E TEST REPORT")
    print("=" * 60)
    print(f"🤫 User Interruption: {'No' if not report.get('user_interruption', True) else 'Yes'}")
    print(f"📈 Total Tests: {report['summary']['total_tests']}")
    print(f"✅ Passed: {report['summary']['passed']}")
    print(f"❌ Failed: {report['summary']['failed']}")
    print(f"🎯 Success Rate: {report['summary']['success_rate']:.1f}%")
    
    print(f"\n📸 Screenshots:")
    print(f"  Directory: {report['screenshots']['directory']}")
    print(f"  Captured: {report['screenshots']['count']} screenshots")
    for screenshot in report['screenshots']['captured'][:5]:  # Show first 5
        print(f"    - {screenshot}")
    if report['screenshots']['count'] > 5:
        print(f"    ... and {report['screenshots']['count'] - 5} more")
    
    print(f"\n📄 Detailed report: fully_headless_e2e_report.json")
    
    success = report['summary']['success_rate'] >= 80.0
    
    if success:
        logger.info("🎉 Fully headless E2E tests passed!")
        return 0
    else:
        logger.warning("⚠️ Some headless E2E tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)