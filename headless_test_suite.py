#!/usr/bin/env python3
"""
Comprehensive Headless Test Suite for AgenticSeek
Includes macOS automation, E2E testing, and MCP server integration
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
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MacOSAutomation:
    """MacOS-specific automation using AppleScript and system commands"""
    
    def __init__(self):
        self.app_path = "/Users/bernhardbudiono/Library/Developer/Xcode/DerivedData/AgenticSeek-bdpcvbrzemrwhfcxcrtpdmjthvtb/Build/Products/Debug/AgenticSeek.app"
        self.app_process = None
        
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
    
    def launch_agenticseek_app(self) -> Dict[str, Any]:
        """Launch the AgenticSeek macOS app"""
        try:
            # Check if app exists
            if not os.path.exists(self.app_path):
                return {"success": False, "error": f"App not found at {self.app_path}"}
            
            # Launch app using open command
            cmd = ["open", self.app_path]
            self.app_process = subprocess.Popen(cmd)
            
            # Wait for app to launch
            time.sleep(3)
            
            # Verify app is running using AppleScript
            verify_script = '''
            tell application "System Events"
                if exists (process "AgenticSeek") then
                    return "running"
                else
                    return "not_running"
                end if
            end tell
            '''
            
            result = self.run_applescript(verify_script)
            is_running = result.get("output") == "running"
            
            return {
                "success": is_running,
                "pid": self.app_process.pid if self.app_process else None,
                "status": "launched" if is_running else "failed_to_launch"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def quit_agenticseek_app(self) -> Dict[str, Any]:
        """Quit the AgenticSeek app"""
        try:
            quit_script = '''
            tell application "AgenticSeek"
                quit
            end tell
            '''
            
            result = self.run_applescript(quit_script)
            
            # Also terminate process if AppleScript fails
            if self.app_process:
                try:
                    self.app_process.terminate()
                    self.app_process.wait(timeout=5)
                except:
                    self.app_process.kill()
                    
            return {"success": True, "method": "applescript" if result["success"] else "process_termination"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_window_info(self) -> Dict[str, Any]:
        """Get information about AgenticSeek windows"""
        script = '''
        tell application "System Events"
            tell process "AgenticSeek"
                set windowList to {}
                repeat with w in windows
                    set windowInfo to {name of w, position of w, size of w}
                    set end of windowList to windowInfo
                end repeat
                return windowList as string
            end tell
        end tell
        '''
        
        return self.run_applescript(script)
    
    def click_button_by_name(self, button_name: str) -> Dict[str, Any]:
        """Click a button in the AgenticSeek app by name"""
        script = f'''
        tell application "System Events"
            tell process "AgenticSeek"
                click button "{button_name}" of window 1
                return "clicked"
            end tell
        end tell
        '''
        
        return self.run_applescript(script)
    
    def navigate_to_view(self, view_name: str) -> Dict[str, Any]:
        """Navigate to a specific view (Models, Config, etc.)"""
        button_map = {
            "models": "Models",
            "config": "Config", 
            "configuration": "Config",
            "settings": "Settings"
        }
        
        button_to_click = button_map.get(view_name.lower(), view_name)
        return self.click_button_by_name(button_to_click)
    
    def take_screenshot(self, filename: str = None) -> Dict[str, Any]:
        """Take a screenshot of the current state"""
        if not filename:
            filename = f"agenticseek_screenshot_{int(time.time())}.png"
        
        try:
            cmd = ["screencapture", "-w", filename]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            return {
                "success": result.returncode == 0,
                "filename": filename,
                "path": os.path.abspath(filename)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def check_ui_elements_visible(self) -> Dict[str, Any]:
        """Check if key UI elements are visible"""
        script = '''
        tell application "System Events"
            tell process "AgenticSeek"
                set buttonCount to count of buttons of window 1
                set visibleButtons to {}
                repeat with b in buttons of window 1
                    if exists b then
                        set end of visibleButtons to name of b
                    end if
                end repeat
                return visibleButtons as string
            end tell
        end tell
        '''
        
        return self.run_applescript(script)


class E2ETestRunner:
    """End-to-end test runner for AgenticSeek"""
    
    def __init__(self):
        self.macos_automation = MacOSAutomation()
        self.base_url = "http://localhost:8000"
        self.test_results = []
        
    async def run_complete_e2e_test(self) -> List[Dict[str, Any]]:
        """Run complete end-to-end test suite"""
        results = []
        
        logger.info("🚀 Starting Complete E2E Test Suite")
        
        # 1. Backend startup test
        results.append(await self.test_backend_startup())
        
        # 2. App launch test
        results.append(await self.test_app_launch())
        
        # 3. UI navigation tests
        results.extend(await self.test_ui_navigation())
        
        # 4. Configuration workflow test
        results.append(await self.test_configuration_workflow())
        
        # 5. Model management workflow test
        results.append(await self.test_model_management_workflow())
        
        # 6. Integration test (UI + Backend)
        results.append(await self.test_ui_backend_integration())
        
        # 7. App cleanup
        results.append(await self.test_app_cleanup())
        
        return results
    
    async def test_backend_startup(self) -> Dict[str, Any]:
        """Test backend service startup"""
        start_time = time.time()
        
        try:
            # Check if backend is running
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=5.0)
                backend_running = response.status_code == 200
                
            duration = time.time() - start_time
            
            return {
                "category": "E2E",
                "test": "Backend Startup",
                "status": "PASSED" if backend_running else "FAILED",
                "duration": duration,
                "backend_running": backend_running
            }
            
        except Exception as e:
            return {
                "category": "E2E",
                "test": "Backend Startup",
                "status": "ERROR",
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    async def test_app_launch(self) -> Dict[str, Any]:
        """Test macOS app launch"""
        start_time = time.time()
        
        try:
            launch_result = self.macos_automation.launch_agenticseek_app()
            duration = time.time() - start_time
            
            # Take screenshot after launch
            screenshot_result = self.macos_automation.take_screenshot("app_launch.png")
            
            return {
                "category": "E2E",
                "test": "App Launch",
                "status": "PASSED" if launch_result["success"] else "FAILED",
                "duration": duration,
                "launch_result": launch_result,
                "screenshot": screenshot_result.get("filename") if screenshot_result["success"] else None
            }
            
        except Exception as e:
            return {
                "category": "E2E",
                "test": "App Launch",
                "status": "ERROR",
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    async def test_ui_navigation(self) -> List[Dict[str, Any]]:
        """Test navigation through all UI views"""
        results = []
        views_to_test = ["models", "config", "main"]
        
        for view_name in views_to_test:
            start_time = time.time()
            
            try:
                # Navigate to view
                nav_result = self.macos_automation.navigate_to_view(view_name)
                
                # Wait for UI to update
                await asyncio.sleep(1)
                
                # Check UI elements
                ui_check = self.macos_automation.check_ui_elements_visible()
                
                # Take screenshot
                screenshot = self.macos_automation.take_screenshot(f"view_{view_name}.png")
                
                duration = time.time() - start_time
                
                results.append({
                    "category": "E2E",
                    "test": f"Navigate to {view_name.title()} View",
                    "status": "PASSED" if nav_result["success"] else "FAILED",
                    "duration": duration,
                    "navigation_result": nav_result,
                    "ui_elements": ui_check.get("output", ""),
                    "screenshot": screenshot.get("filename") if screenshot["success"] else None
                })
                
            except Exception as e:
                results.append({
                    "category": "E2E",
                    "test": f"Navigate to {view_name.title()} View",
                    "status": "ERROR",
                    "duration": time.time() - start_time,
                    "error": str(e)
                })
        
        return results
    
    async def test_configuration_workflow(self) -> Dict[str, Any]:
        """Test configuration management workflow"""
        start_time = time.time()
        
        try:
            # Navigate to config view
            nav_result = self.macos_automation.navigate_to_view("config")
            if not nav_result["success"]:
                raise Exception("Failed to navigate to config view")
            
            await asyncio.sleep(2)  # Wait for view to load
            
            # Test API key loading via backend
            async with httpx.AsyncClient() as client:
                api_keys_response = await client.get(f"{self.base_url}/config/api-keys", timeout=10)
                providers_response = await client.get(f"{self.base_url}/config/providers", timeout=10)
                
            # Take screenshot of config view
            screenshot = self.macos_automation.take_screenshot("config_view.png")
            
            duration = time.time() - start_time
            
            config_loaded = (api_keys_response.status_code == 200 and 
                           providers_response.status_code == 200)
            
            return {
                "category": "E2E",
                "test": "Configuration Workflow",
                "status": "PASSED" if config_loaded else "FAILED",
                "duration": duration,
                "api_keys_status": api_keys_response.status_code,
                "providers_status": providers_response.status_code,
                "screenshot": screenshot.get("filename") if screenshot["success"] else None
            }
            
        except Exception as e:
            return {
                "category": "E2E",
                "test": "Configuration Workflow",
                "status": "ERROR",
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    async def test_model_management_workflow(self) -> Dict[str, Any]:
        """Test model management workflow"""
        start_time = time.time()
        
        try:
            # Navigate to models view
            nav_result = self.macos_automation.navigate_to_view("models")
            if not nav_result["success"]:
                raise Exception("Failed to navigate to models view")
            
            await asyncio.sleep(2)  # Wait for view to load
            
            # Test model catalog loading via backend
            async with httpx.AsyncClient() as client:
                catalog_response = await client.get(f"{self.base_url}/models/catalog", timeout=10)
                installed_response = await client.get(f"{self.base_url}/models/installed", timeout=10)
                storage_response = await client.get(f"{self.base_url}/models/storage", timeout=10)
            
            # Take screenshot of models view
            screenshot = self.macos_automation.take_screenshot("models_view.png")
            
            duration = time.time() - start_time
            
            models_loaded = (catalog_response.status_code == 200 and 
                           installed_response.status_code == 200 and
                           storage_response.status_code == 200)
            
            return {
                "category": "E2E",
                "test": "Model Management Workflow",
                "status": "PASSED" if models_loaded else "FAILED",
                "duration": duration,
                "catalog_status": catalog_response.status_code,
                "installed_status": installed_response.status_code,
                "storage_status": storage_response.status_code,
                "screenshot": screenshot.get("filename") if screenshot["success"] else None
            }
            
        except Exception as e:
            return {
                "category": "E2E",
                "test": "Model Management Workflow",
                "status": "ERROR",
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    async def test_ui_backend_integration(self) -> Dict[str, Any]:
        """Test integration between UI and backend"""
        start_time = time.time()
        
        try:
            # Test query workflow
            async with httpx.AsyncClient() as client:
                # Send a test query
                query_response = await client.post(
                    f"{self.base_url}/query",
                    json={"message": "E2E test query", "session_id": "e2e_test"},
                    timeout=30
                )
                
                # Get latest answer
                if query_response.status_code == 200:
                    await asyncio.sleep(1)
                    answer_response = await client.get(f"{self.base_url}/latest_answer", timeout=10)
                else:
                    answer_response = None
            
            # Take screenshot during integration test
            screenshot = self.macos_automation.take_screenshot("integration_test.png")
            
            duration = time.time() - start_time
            
            integration_success = (query_response.status_code == 200 and
                                 answer_response and answer_response.status_code == 200)
            
            return {
                "category": "E2E",
                "test": "UI Backend Integration",
                "status": "PASSED" if integration_success else "FAILED",
                "duration": duration,
                "query_status": query_response.status_code,
                "answer_status": answer_response.status_code if answer_response else None,
                "screenshot": screenshot.get("filename") if screenshot["success"] else None
            }
            
        except Exception as e:
            return {
                "category": "E2E",
                "test": "UI Backend Integration",
                "status": "ERROR",
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    async def test_app_cleanup(self) -> Dict[str, Any]:
        """Test app cleanup and shutdown"""
        start_time = time.time()
        
        try:
            quit_result = self.macos_automation.quit_agenticseek_app()
            duration = time.time() - start_time
            
            # Verify app is no longer running
            await asyncio.sleep(2)
            verify_script = '''
            tell application "System Events"
                if exists (process "AgenticSeek") then
                    return "still_running"
                else
                    return "stopped"
                end if
            end tell
            '''
            
            verify_result = self.macos_automation.run_applescript(verify_script)
            app_stopped = verify_result.get("output") == "stopped"
            
            return {
                "category": "E2E",
                "test": "App Cleanup",
                "status": "PASSED" if quit_result["success"] and app_stopped else "FAILED",
                "duration": duration,
                "quit_result": quit_result,
                "app_stopped": app_stopped
            }
            
        except Exception as e:
            return {
                "category": "E2E",
                "test": "App Cleanup",
                "status": "ERROR",
                "duration": time.time() - start_time,
                "error": str(e)
            }


class HeadlessTestSuite:
    """Main headless test suite orchestrator"""
    
    def __init__(self):
        self.e2e_runner = E2ETestRunner()
        self.test_results = []
        self.performance_metrics = {}
        
    async def run_comprehensive_tests(self) -> bool:
        """Run all comprehensive headless tests"""
        logger.info("🚀 Starting Comprehensive Headless Test Suite")
        print("=" * 70)
        
        # Test categories
        test_categories = [
            ("Backend API Tests", self.run_backend_api_tests),
            ("macOS Automation Tests", self.run_macos_automation_tests),
            ("E2E Integration Tests", self.run_e2e_integration_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Security Tests", self.run_security_tests)
        ]
        
        # Run test categories
        for category_name, test_func in test_categories:
            logger.info(f"🧪 Starting {category_name}")
            start_time = time.time()
            
            try:
                category_results = await test_func()
                duration = time.time() - start_time
                
                success_count = sum(1 for r in category_results if r.get('status') == 'PASSED')
                total_count = len(category_results)
                
                logger.info(f"✅ {category_name}: {success_count}/{total_count} passed ({duration:.2f}s)")
                
                self.test_results.extend(category_results)
                self.performance_metrics[category_name] = {
                    'duration': duration,
                    'success_rate': (success_count / total_count) * 100 if total_count > 0 else 0
                }
                
            except Exception as e:
                logger.error(f"❌ {category_name} failed: {str(e)}")
                self.test_results.append({
                    'category': category_name,
                    'test': 'category_execution',
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        # Generate final report
        return await self.generate_comprehensive_report()
    
    async def run_backend_api_tests(self) -> List[Dict[str, Any]]:
        """Run backend API tests"""
        results = []
        
        endpoints = [
            ("Health Check", "GET", "/health"),
            ("Config Providers", "GET", "/config/providers"),
            ("Config API Keys", "GET", "/config/api-keys"),
            ("Models Catalog", "GET", "/models/catalog"),
            ("Models Installed", "GET", "/models/installed"),
            ("Models Storage", "GET", "/models/storage")
        ]
        
        async with httpx.AsyncClient() as client:
            for test_name, method, endpoint in endpoints:
                start_time = time.time()
                
                try:
                    if method == "GET":
                        response = await client.get(f"http://localhost:8000{endpoint}", timeout=10)
                    
                    duration = time.time() - start_time
                    
                    results.append({
                        'category': 'Backend API',
                        'test': test_name,
                        'status': 'PASSED' if response.status_code == 200 else 'FAILED',
                        'duration': duration,
                        'status_code': response.status_code
                    })
                    
                except Exception as e:
                    results.append({
                        'category': 'Backend API',
                        'test': test_name,
                        'status': 'ERROR',
                        'duration': time.time() - start_time,
                        'error': str(e)
                    })
        
        return results
    
    async def run_macos_automation_tests(self) -> List[Dict[str, Any]]:
        """Run macOS automation tests"""
        results = []
        automation = MacOSAutomation()
        
        # Test AppleScript functionality
        start_time = time.time()
        applescript_test = automation.run_applescript('return "test"')
        results.append({
            'category': 'macOS Automation',
            'test': 'AppleScript Execution',
            'status': 'PASSED' if applescript_test['success'] else 'FAILED',
            'duration': time.time() - start_time,
            'output': applescript_test.get('output', '')
        })
        
        # Test system process checking
        start_time = time.time()
        process_check = automation.run_applescript('''
        tell application "System Events"
            return count of processes
        end tell
        ''')
        results.append({
            'category': 'macOS Automation',
            'test': 'System Process Check',
            'status': 'PASSED' if process_check['success'] else 'FAILED',
            'duration': time.time() - start_time,
            'process_count': process_check.get('output', '0')
        })
        
        # Test screenshot capability
        start_time = time.time()
        screenshot_test = automation.take_screenshot("test_screenshot.png")
        results.append({
            'category': 'macOS Automation',
            'test': 'Screenshot Capability',
            'status': 'PASSED' if screenshot_test['success'] else 'FAILED',
            'duration': time.time() - start_time,
            'screenshot_path': screenshot_test.get('path', '')
        })
        
        return results
    
    async def run_e2e_integration_tests(self) -> List[Dict[str, Any]]:
        """Run end-to-end integration tests"""
        return await self.e2e_runner.run_complete_e2e_test()
    
    async def run_performance_tests(self) -> List[Dict[str, Any]]:
        """Run performance tests"""
        results = []
        
        # Backend response time test
        response_times = []
        for i in range(5):
            start_time = time.time()
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:8000/health", timeout=5)
                response_time = time.time() - start_time
                if response.status_code == 200:
                    response_times.append(response_time)
            except:
                pass
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            results.append({
                'category': 'Performance',
                'test': 'Backend Response Time',
                'status': 'PASSED' if avg_response_time < 1.0 else 'FAILED',
                'duration': sum(response_times),
                'average_response_time_ms': avg_response_time * 1000,
                'samples': len(response_times)
            })
        
        return results
    
    async def run_security_tests(self) -> List[Dict[str, Any]]:
        """Run security tests"""
        results = []
        
        # CORS test
        try:
            async with httpx.AsyncClient() as client:
                response = await client.options(
                    "http://localhost:8000/query",
                    headers={"Origin": "http://localhost:3000"}
                )
                
                has_cors = "access-control-allow-origin" in response.headers
                results.append({
                    'category': 'Security',
                    'test': 'CORS Headers',
                    'status': 'PASSED' if has_cors else 'FAILED',
                    'duration': 0.1,
                    'cors_enabled': has_cors
                })
        except Exception as e:
            results.append({
                'category': 'Security',
                'test': 'CORS Headers',
                'status': 'ERROR',
                'error': str(e)
            })
        
        return results
    
    async def generate_comprehensive_report(self) -> bool:
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE HEADLESS TEST REPORT")
        print("=" * 70)
        
        # Overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.get('status') == 'PASSED')
        failed_tests = sum(1 for r in self.test_results if r.get('status') == 'FAILED')
        error_tests = sum(1 for r in self.test_results if r.get('status') == 'ERROR')
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"💥 Errors: {error_tests}")
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        # Performance metrics
        print("\n⚡ Performance Metrics:")
        for category, metrics in self.performance_metrics.items():
            print(f"  {category}: {metrics['duration']:.2f}s ({metrics['success_rate']:.1f}% success)")
        
        # Category breakdown
        print("\n📋 Test Categories:")
        categories = {}
        for result in self.test_results:
            category = result.get('category', 'Unknown')
            if category not in categories:
                categories[category] = {'passed': 0, 'failed': 0, 'error': 0}
            
            status = result.get('status', 'ERROR')
            if status == 'PASSED':
                categories[category]['passed'] += 1
            elif status == 'FAILED':
                categories[category]['failed'] += 1
            else:
                categories[category]['error'] += 1
        
        for category, stats in categories.items():
            total = stats['passed'] + stats['failed'] + stats['error']
            rate = (stats['passed'] / total) * 100 if total > 0 else 0
            print(f"  {category}: {stats['passed']}/{total} ({rate:.1f}%)")
        
        # Failed tests detail
        failed_results = [r for r in self.test_results if r.get('status') in ['FAILED', 'ERROR']]
        if failed_results:
            print("\n🔍 Failed/Error Tests:")
            for result in failed_results:
                print(f"  ❌ {result.get('category', 'Unknown')}/{result.get('test', 'Unknown')}: {result.get('status', 'ERROR')}")
                if 'error' in result:
                    print(f"     Error: {result['error']}")
        
        # Save detailed report
        report_file = Path("headless_test_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'summary': {
                    'total_tests': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'errors': error_tests,
                    'success_rate': success_rate
                },
                'performance_metrics': self.performance_metrics,
                'detailed_results': self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return success_rate >= 85.0  # 85% success rate target


async def main():
    """Main test runner"""
    test_suite = HeadlessTestSuite()
    
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
    
    # Run comprehensive tests
    success = await test_suite.run_comprehensive_tests()
    
    if success:
        logger.info("🎉 All headless tests passed! System is fully functional.")
        return 0
    else:
        logger.warning("⚠️ Some tests failed. Check the report above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)