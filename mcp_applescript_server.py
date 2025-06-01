#!/usr/bin/env python3
"""
MCP AppleScript Server for AgenticSeek Testing
Provides AppleScript automation capabilities through MCP protocol
"""

import asyncio
import json
import subprocess
import sys
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MCPMessage:
    """MCP message structure"""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

class AppleScriptMCPServer:
    """MCP Server for AppleScript automation"""
    
    def __init__(self):
        self.tools = {
            "applescript_execute": {
                "name": "applescript_execute",
                "description": "Execute AppleScript commands for macOS automation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "AppleScript code to execute"
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds (default: 30)",
                            "default": 30
                        }
                    },
                    "required": ["script"]
                }
            },
            "app_automation": {
                "name": "app_automation",
                "description": "High-level app automation commands",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["launch", "quit", "click_button", "get_ui_info", "take_screenshot"],
                            "description": "Automation action to perform"
                        },
                        "app_name": {
                            "type": "string",
                            "description": "Name of the application",
                            "default": "AgenticSeek"
                        },
                        "target": {
                            "type": "string",
                            "description": "Target element (for click_button action)"
                        },
                        "screenshot_path": {
                            "type": "string",
                            "description": "Path for screenshot (for take_screenshot action)"
                        }
                    },
                    "required": ["action"]
                }
            },
            "ui_validation": {
                "name": "ui_validation",
                "description": "Validate UI elements and state",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "app_name": {
                            "type": "string",
                            "description": "Name of the application",
                            "default": "AgenticSeek"
                        },
                        "expected_elements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of expected UI elements"
                        },
                        "validation_type": {
                            "type": "string",
                            "enum": ["buttons", "windows", "text_fields", "all"],
                            "description": "Type of elements to validate",
                            "default": "all"
                        }
                    },
                    "required": ["expected_elements"]
                }
            }
        }
        
        self.app_paths = {
            "AgenticSeek": "/Users/bernhardbudiono/Library/Developer/Xcode/DerivedData/AgenticSeek-bdpcvbrzemrwhfcxcrtpdmjthvtb/Build/Products/Debug/AgenticSeek.app"
        }
    
    def create_response(self, message_id: str, result: Any = None, error: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create MCP response message"""
        response = {
            "jsonrpc": "2.0",
            "id": message_id
        }
        
        if error:
            response["error"] = error
        else:
            response["result"] = result
            
        return response
    
    def create_error(self, code: int, message: str, data: Any = None) -> Dict[str, Any]:
        """Create MCP error object"""
        error = {
            "code": code,
            "message": message
        }
        if data:
            error["data"] = data
        return error
    
    def execute_applescript(self, script: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute AppleScript and return results"""
        try:
            cmd = ["osascript", "-e", script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout.strip(),
                "error": result.stderr.strip() if result.stderr else None,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"AppleScript execution timed out after {timeout} seconds",
                "timeout": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "exception": True
            }
    
    def app_launch(self, app_name: str) -> Dict[str, Any]:
        """Launch an application"""
        app_path = self.app_paths.get(app_name)
        if not app_path:
            return {
                "success": False,
                "error": f"Unknown application: {app_name}"
            }
        
        try:
            # Use open command to launch
            cmd = ["open", app_path]
            process = subprocess.Popen(cmd)
            
            # Wait a moment and verify launch
            time.sleep(3)
            
            verify_script = f'''
            tell application "System Events"
                if exists (process "{app_name}") then
                    tell process "{app_name}"
                        set frontmost to true
                    end tell
                    return "running"
                else
                    return "not_running"
                end if
            end tell
            '''
            
            verify_result = self.execute_applescript(verify_script)
            is_running = verify_result.get("output") == "running"
            
            return {
                "success": is_running,
                "pid": process.pid,
                "status": "launched" if is_running else "failed_to_launch",
                "verification": verify_result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def app_quit(self, app_name: str) -> Dict[str, Any]:
        """Quit an application"""
        quit_script = f'''
        tell application "{app_name}"
            try
                quit
            end try
        end tell
        
        tell application "System Events"
            try
                tell process "{app_name}"
                    click button 1 of window 1
                end try
            end try
        end tell
        '''
        
        result = self.execute_applescript(quit_script)
        
        # Also force kill if needed
        try:
            subprocess.run(["pkill", "-f", app_name], capture_output=True)
        except:
            pass
        
        return {
            "success": True,
            "applescript_result": result,
            "method": "comprehensive_quit"
        }
    
    def click_button(self, app_name: str, button_name: str) -> Dict[str, Any]:
        """Click a button in the application"""
        script = f'''
        tell application "System Events"
            tell process "{app_name}"
                try
                    set frontmost to true
                    delay 0.5
                    
                    set buttonFound to false
                    repeat with b in buttons of window 1
                        if name of b is "{button_name}" then
                            click b
                            set buttonFound to true
                            exit repeat
                        end if
                    end repeat
                    
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
        
        result = self.execute_applescript(script)
        
        # Add delay for UI update
        if result.get("output") == "clicked":
            time.sleep(1)
        
        return result
    
    def get_ui_info(self, app_name: str) -> Dict[str, Any]:
        """Get UI information for the application"""
        script = f'''
        tell application "System Events"
            tell process "{app_name}"
                try
                    set frontmost to true
                    
                    set windowInfo to {{}}
                    set buttonInfo to {{}}
                    set textInfo to {{}}
                    
                    repeat with w in windows
                        set windowData to {{name of w, position of w, size of w}}
                        set end of windowInfo to windowData
                    end repeat
                    
                    repeat with b in buttons of window 1
                        try
                            set buttonData to {{name of b, position of b, enabled of b}}
                            set end of buttonInfo to buttonData
                        end try
                    end repeat
                    
                    repeat with t in text fields of window 1
                        try
                            set textData to {{value of t, position of t}}
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
        
        return self.execute_applescript(script)
    
    def take_screenshot(self, screenshot_path: str = None) -> Dict[str, Any]:
        """Take a screenshot"""
        if not screenshot_path:
            screenshot_path = f"mcp_screenshot_{int(time.time())}.png"
        
        try:
            cmd = ["screencapture", "-w", screenshot_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            return {
                "success": result.returncode == 0,
                "path": screenshot_path,
                "absolute_path": subprocess.check_output(["realpath", screenshot_path], text=True).strip()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_ui_elements(self, app_name: str, expected_elements: List[str], validation_type: str = "all") -> Dict[str, Any]:
        """Validate UI elements"""
        ui_info_result = self.get_ui_info(app_name)
        
        if not ui_info_result["success"]:
            return {
                "success": False,
                "error": "Failed to get UI information",
                "ui_info_result": ui_info_result
            }
        
        ui_output = ui_info_result.get("output", "").lower()
        found_elements = []
        missing_elements = []
        
        for element in expected_elements:
            if element.lower() in ui_output:
                found_elements.append(element)
            else:
                missing_elements.append(element)
        
        validation_score = len(found_elements) / len(expected_elements) if expected_elements else 1.0
        
        return {
            "success": len(missing_elements) == 0,
            "found_elements": found_elements,
            "missing_elements": missing_elements,
            "validation_score": validation_score,
            "ui_output": ui_info_result.get("output", ""),
            "validation_type": validation_type
        }
    
    def handle_tool_call(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call based on tool name"""
        try:
            if tool_name == "applescript_execute":
                script = params.get("script", "")
                timeout = params.get("timeout", 30)
                
                if not script:
                    return self.create_error(-32602, "Missing required parameter: script")
                
                result = self.execute_applescript(script, timeout)
                return result
            
            elif tool_name == "app_automation":
                action = params.get("action", "")
                app_name = params.get("app_name", "AgenticSeek")
                
                if not action:
                    return self.create_error(-32602, "Missing required parameter: action")
                
                if action == "launch":
                    return self.app_launch(app_name)
                elif action == "quit":
                    return self.app_quit(app_name)
                elif action == "click_button":
                    target = params.get("target", "")
                    if not target:
                        return self.create_error(-32602, "Missing required parameter: target for click_button action")
                    return self.click_button(app_name, target)
                elif action == "get_ui_info":
                    return self.get_ui_info(app_name)
                elif action == "take_screenshot":
                    screenshot_path = params.get("screenshot_path")
                    return self.take_screenshot(screenshot_path)
                else:
                    return self.create_error(-32602, f"Unknown action: {action}")
            
            elif tool_name == "ui_validation":
                app_name = params.get("app_name", "AgenticSeek")
                expected_elements = params.get("expected_elements", [])
                validation_type = params.get("validation_type", "all")
                
                if not expected_elements:
                    return self.create_error(-32602, "Missing required parameter: expected_elements")
                
                return self.validate_ui_elements(app_name, expected_elements, validation_type)
            
            else:
                return self.create_error(-32601, f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Error handling tool call {tool_name}: {str(e)}")
            return self.create_error(-32603, f"Internal error: {str(e)}")
    
    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming MCP message"""
        try:
            method = message.get("method")
            params = message.get("params", {})
            message_id = message.get("id")
            
            if method == "initialize":
                return self.create_response(message_id, {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "applescript-automation",
                        "version": "1.0.0"
                    }
                })
            
            elif method == "tools/list":
                return self.create_response(message_id, {
                    "tools": list(self.tools.values())
                })
            
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_params = params.get("arguments", {})
                
                if not tool_name:
                    return self.create_response(message_id, error=self.create_error(-32602, "Missing tool name"))
                
                result = self.handle_tool_call(tool_name, tool_params)
                
                return self.create_response(message_id, {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                })
            
            else:
                return self.create_response(message_id, error=self.create_error(-32601, f"Unknown method: {method}"))
                
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            return self.create_response(message.get("id"), error=self.create_error(-32603, f"Internal error: {str(e)}"))
    
    async def run_server(self):
        """Run the MCP server"""
        logger.info("🚀 Starting AppleScript MCP Server")
        logger.info("📱 Available tools: applescript_execute, app_automation, ui_validation")
        
        try:
            while True:
                # Read from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                
                if not line:
                    break
                
                try:
                    message = json.loads(line.strip())
                    response = await self.handle_message(message)
                    
                    if response:
                        print(json.dumps(response))
                        sys.stdout.flush()
                        
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    
        except KeyboardInterrupt:
            logger.info("🛑 Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {str(e)}")

# Standalone test runner
async def test_mcp_server():
    """Test the MCP server functionality"""
    server = AppleScriptMCPServer()
    
    print("🧪 Testing AppleScript MCP Server")
    print("=" * 50)
    
    # Test AppleScript execution
    test_script = 'tell application "System Events" to return (count of processes)'
    result = server.execute_applescript(test_script)
    print(f"✅ AppleScript test: {result}")
    
    # Test app automation
    ui_info = server.get_ui_info("Finder")
    print(f"✅ UI Info test: {ui_info.get('success', False)}")
    
    # Test screenshot
    screenshot_result = server.take_screenshot("test_mcp_screenshot.png")
    print(f"✅ Screenshot test: {screenshot_result}")
    
    print("\n🎉 MCP Server tests completed!")

async def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        await test_mcp_server()
    else:
        server = AppleScriptMCPServer()
        await server.run_server()

if __name__ == "__main__":
    asyncio.run(main())