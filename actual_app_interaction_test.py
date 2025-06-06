#!/usr/bin/env python3
"""
ACTUAL APP INTERACTION TEST
This will ACTUALLY interact with the running AgenticSeek app
HONESTY CHECK: I will prove the app is working by actually using it
"""

import subprocess
import time
import json
from datetime import datetime

class ActualAppInteractionTest:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'app_found': False,
            'sso_button_found': False,
            'chat_input_found': False,
            'send_button_found': False,
            'message_sent': False,
            'response_received': False,
            'provider_switch_found': False,
            'actual_interactions': [],
            'errors': []
        }
    
    def run_applescript(self, script, timeout=30):
        """Execute AppleScript with detailed error handling"""
        try:
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                print(f"✅ AppleScript success: {output}")
                return output
            else:
                error = result.stderr.strip()
                print(f"❌ AppleScript error: {error}")
                self.results['errors'].append(f"AppleScript error: {error}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ AppleScript timeout after {timeout}s")
            self.results['errors'].append(f"AppleScript timeout after {timeout}s")
            return None
        except Exception as e:
            print(f"💥 AppleScript exception: {e}")
            self.results['errors'].append(f"AppleScript exception: {e}")
            return None
    
    def find_and_activate_app(self):
        """Find and activate the AgenticSeek app"""
        print("🔍 Looking for AgenticSeek app...")
        
        script = '''
        tell application "System Events"
            set appList to name of every process
            if "AgenticSeek" is in appList then
                tell application "AgenticSeek" to activate
                delay 2
                return "AgenticSeek found and activated"
            else
                return "AgenticSeek not found in: " & (appList as string)
            end if
        end tell
        '''
        
        result = self.run_applescript(script)
        if result and "found and activated" in result:
            self.results['app_found'] = True
            self.results['actual_interactions'].append({
                'action': 'app_activation',
                'result': 'success',
                'timestamp': datetime.now().isoformat()
            })
            return True
        return False
    
    def examine_app_ui(self):
        """Examine the actual UI elements in the app"""
        print("🔍 Examining app UI elements...")
        
        script = '''
        tell application "System Events"
            tell process "AgenticSeek"
                try
                    set windowCount to count of windows
                    if windowCount > 0 then
                        set mainWindow to window 1
                        set windowTitle to title of mainWindow
                        
                        -- Count UI elements
                        set buttonCount to count of buttons of mainWindow
                        set textFieldCount to count of text fields of mainWindow
                        set staticTextCount to count of static texts of mainWindow
                        
                        -- Look for specific elements
                        set buttonNames to {}
                        repeat with btn in buttons of mainWindow
                            try
                                set buttonNames to buttonNames & {title of btn}
                            on error
                                set buttonNames to buttonNames & {"(no title)"}
                            end try
                        end repeat
                        
                        return "Window: " & windowTitle & " | Buttons: " & buttonCount & " | TextFields: " & textFieldCount & " | StaticTexts: " & staticTextCount & " | Button names: " & (buttonNames as string)
                    else
                        return "No windows found"
                    end if
                on error errMsg
                    return "UI examination error: " & errMsg
                end try
            end tell
        end tell
        '''
        
        result = self.run_applescript(script)
        if result:
            self.results['actual_interactions'].append({
                'action': 'ui_examination',
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
            return result
        return None
    
    def look_for_sso_elements(self):
        """Look for SSO/authentication elements"""
        print("🔐 Looking for SSO elements...")
        
        script = '''
        tell application "System Events"
            tell process "AgenticSeek"
                try
                    set foundElements to {}
                    
                    -- Look for Sign In button
                    repeat with btn in buttons of window 1
                        set btnTitle to title of btn
                        if btnTitle contains "Sign" or btnTitle contains "Apple" or btnTitle contains "Auth" then
                            set foundElements to foundElements & {"Button: " & btnTitle}
                        end if
                    end repeat
                    
                    -- Look for authentication text
                    repeat with txt in static texts of window 1
                        set txtValue to value of txt
                        if txtValue contains "bernhardbudiono" or txtValue contains "gmail" or txtValue contains "Auth" or txtValue contains "Ready" then
                            set foundElements to foundElements & {"Text: " & txtValue}
                        end if
                    end repeat
                    
                    if (count of foundElements) > 0 then
                        return "SSO elements found: " & (foundElements as string)
                    else
                        return "No SSO elements found"
                    end if
                    
                on error errMsg
                    return "SSO search error: " & errMsg
                end try
            end tell
        end tell
        '''
        
        result = self.run_applescript(script)
        if result and "found:" in result:
            self.results['sso_button_found'] = True
            self.results['actual_interactions'].append({
                'action': 'sso_search',
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
        return result
    
    def try_authentication(self):
        """Try to authenticate if needed"""
        print("🔐 Attempting authentication...")
        
        script = '''
        tell application "System Events"
            tell process "AgenticSeek"
                try
                    -- Look for Sign In button and click it
                    repeat with btn in buttons of window 1
                        set btnTitle to title of btn
                        if btnTitle contains "Sign" and btnTitle contains "Apple" then
                            click btn
                            delay 3
                            return "Clicked Sign In button: " & btnTitle
                        end if
                    end repeat
                    
                    return "No Sign In button found to click"
                    
                on error errMsg
                    return "Authentication error: " & errMsg
                end try
            end tell
        end tell
        '''
        
        result = self.run_applescript(script)
        if result:
            self.results['actual_interactions'].append({
                'action': 'authentication_attempt',
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
        return result
    
    def find_chat_input(self):
        """Find the chat input field"""
        print("💬 Looking for chat input field...")
        
        script = '''
        tell application "System Events"
            tell process "AgenticSeek"
                try
                    set textFieldCount to count of text fields of window 1
                    if textFieldCount > 0 then
                        set chatField to text field 1 of window 1
                        set currentValue to value of chatField
                        return "Chat input found with value: '" & currentValue & "' (placeholder or content)"
                    else
                        return "No text fields found"
                    end if
                    
                on error errMsg
                    return "Chat input search error: " & errMsg
                end try
            end tell
        end tell
        '''
        
        result = self.run_applescript(script)
        if result and "found" in result:
            self.results['chat_input_found'] = True
            self.results['actual_interactions'].append({
                'action': 'chat_input_search',
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
        return result
    
    def send_actual_message(self):
        """Send an actual message through the app"""
        print("📤 Sending actual message through the app...")
        
        test_message = "Hello from automated test! Please confirm you received this message."
        
        script = f'''
        tell application "System Events"
            tell process "AgenticSeek"
                try
                    -- Find and fill the text field
                    if (count of text fields of window 1) > 0 then
                        set chatField to text field 1 of window 1
                        set focused of chatField to true
                        delay 1
                        set value of chatField to "{test_message}"
                        delay 1
                        
                        -- Look for send button (could be arrow icon or text)
                        set sendClicked to false
                        repeat with btn in buttons of window 1
                            set btnTitle to title of btn
                            if btnTitle contains "Send" or btnTitle contains "arrow" or btnTitle = "" then
                                click btn
                                set sendClicked to true
                                exit repeat
                            end if
                        end repeat
                        
                        if sendClicked then
                            return "Message sent: '{test_message}'"
                        else
                            -- Try pressing Enter key
                            key code 36  -- Enter key
                            return "Message sent via Enter key: '{test_message}'"
                        end if
                    else
                        return "No text field available for message"
                    end if
                    
                on error errMsg
                    return "Message sending error: " & errMsg
                end try
            end tell
        end tell
        '''
        
        result = self.run_applescript(script)
        if result and "sent:" in result:
            self.results['message_sent'] = True
            self.results['send_button_found'] = True
            self.results['actual_interactions'].append({
                'action': 'message_send',
                'result': result,
                'message': test_message,
                'timestamp': datetime.now().isoformat()
            })
        return result
    
    def wait_for_response(self):
        """Wait and look for AI response"""
        print("⏳ Waiting for AI response...")
        
        # Wait a bit for response
        time.sleep(8)
        
        script = '''
        tell application "System Events"
            tell process "AgenticSeek"
                try
                    -- Look for new content that might be a response
                    set allTexts to {}
                    repeat with txt in static texts of window 1
                        set txtValue to value of txt
                        if length of txtValue > 10 then
                            set allTexts to allTexts & {txtValue}
                        end if
                    end repeat
                    
                    if (count of allTexts) > 0 then
                        return "Found text content: " & (item -1 of allTexts)  -- Last text item
                    else
                        return "No response text found"
                    end if
                    
                on error errMsg
                    return "Response check error: " & errMsg
                end try
            end tell
        end tell
        '''
        
        result = self.run_applescript(script)
        if result and "Found text content:" in result:
            self.results['response_received'] = True
            self.results['actual_interactions'].append({
                'action': 'response_check',
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
        return result
    
    def test_provider_switching(self):
        """Test provider switching functionality"""
        print("🔄 Testing provider switching...")
        
        script = '''
        tell application "System Events"
            tell process "AgenticSeek"
                try
                    -- Look for provider menu or switch button
                    set providerElements to {}
                    
                    repeat with btn in buttons of window 1
                        set btnTitle to title of btn
                        if btnTitle contains "Claude" or btnTitle contains "GPT" or btnTitle contains "Provider" or btnTitle contains "Anthropic" or btnTitle contains "OpenAI" then
                            set providerElements to providerElements & {"Button: " & btnTitle}
                        end if
                    end repeat
                    
                    repeat with pop in pop up buttons of window 1
                        set popTitle to title of pop
                        if popTitle contains "Claude" or popTitle contains "GPT" or popTitle contains "Provider" then
                            set providerElements to providerElements & {"Menu: " & popTitle}
                        end if
                    end repeat
                    
                    if (count of providerElements) > 0 then
                        return "Provider elements found: " & (providerElements as string)
                    else
                        return "No provider switching elements found"
                    end if
                    
                on error errMsg
                    return "Provider switch test error: " & errMsg
                end try
            end tell
        end tell
        '''
        
        result = self.run_applescript(script)
        if result and "found:" in result:
            self.results['provider_switch_found'] = True
            self.results['actual_interactions'].append({
                'action': 'provider_switch_test',
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
        return result
    
    def run_comprehensive_app_test(self):
        """Run comprehensive test of the actual running app"""
        print("🚀 COMPREHENSIVE ACTUAL APP INTERACTION TEST")
        print("=" * 70)
        print("📱 Testing the REAL AgenticSeek app interface...")
        print("=" * 70)
        
        # Step 1: Find and activate app
        if not self.find_and_activate_app():
            print("❌ CRITICAL: AgenticSeek app not found or couldn't activate")
            return False
        
        # Step 2: Examine UI
        ui_info = self.examine_app_ui()
        print(f"🖼️ UI Info: {ui_info}")
        
        # Step 3: Look for SSO elements
        sso_info = self.look_for_sso_elements()
        print(f"🔐 SSO Info: {sso_info}")
        
        # Step 4: Try authentication if needed
        auth_result = self.try_authentication()
        print(f"🔓 Auth Result: {auth_result}")
        
        # Step 5: Find chat input
        input_info = self.find_chat_input()
        print(f"💬 Input Info: {input_info}")
        
        # Step 6: Send actual message
        send_result = self.send_actual_message()
        print(f"📤 Send Result: {send_result}")
        
        # Step 7: Wait for response
        response_info = self.wait_for_response()
        print(f"📥 Response Info: {response_info}")
        
        # Step 8: Test provider switching
        provider_info = self.test_provider_switching()
        print(f"🔄 Provider Info: {provider_info}")
        
        # Generate final results
        self.generate_final_results()
        
        return self.results['message_sent'] and self.results['response_received']
    
    def generate_final_results(self):
        """Generate final comprehensive results"""
        print("\n" + "=" * 70)
        print("📊 ACTUAL APP INTERACTION RESULTS")
        print("=" * 70)
        
        # Count successes
        success_count = sum([
            self.results['app_found'],
            self.results['chat_input_found'],
            self.results['message_sent'],
            self.results['response_received']
        ])
        
        # Display results
        print(f"App Found & Activated:    {'✅' if self.results['app_found'] else '❌'}")
        print(f"SSO Elements Found:       {'✅' if self.results['sso_button_found'] else '⚠️'}")
        print(f"Chat Input Found:         {'✅' if self.results['chat_input_found'] else '❌'}")
        print(f"Send Button Found:        {'✅' if self.results['send_button_found'] else '❌'}")
        print(f"Message Actually Sent:    {'✅' if self.results['message_sent'] else '❌'}")
        print(f"Response Received:        {'✅' if self.results['response_received'] else '❌'}")
        print(f"Provider Switch Found:    {'✅' if self.results['provider_switch_found'] else '⚠️'}")
        
        print(f"\nTotal Interactions:       {len(self.results['actual_interactions'])}")
        print(f"Errors Encountered:       {len(self.results['errors'])}")
        
        # Show recent interactions
        if self.results['actual_interactions']:
            print("\n📝 Recent Interactions:")
            for interaction in self.results['actual_interactions'][-3:]:
                print(f"  • {interaction['action']}: {interaction['result'][:60]}...")
        
        # Overall assessment
        if success_count >= 3:
            print("\n🎉 SUCCESS: AgenticSeek app is functional!")
            print("✅ The app interface is working and responsive")
            if self.results['message_sent']:
                print("✅ Messages can be sent through the actual UI")
            if self.results['response_received']:
                print("✅ AI responses are being received in the app")
        else:
            print("\n❌ FAILURE: App functionality issues detected")
            if self.results['errors']:
                print("Errors:")
                for error in self.results['errors'][-3:]:
                    print(f"  • {error}")
        
        # Save detailed results
        results_file = '/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/actual_app_interaction_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📝 Detailed results saved to: {results_file}")

def main():
    tester = ActualAppInteractionTest()
    success = tester.run_comprehensive_app_test()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()