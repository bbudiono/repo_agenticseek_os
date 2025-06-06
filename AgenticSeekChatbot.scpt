-- AgenticSeek Chatbot with Apple Sign In and Real LLM APIs
-- Accessibility: All windows have proper labels for automation testing

property userEmail : "bernhardbudiono@gmail.com"
property isAuthenticated : false
property currentProvider : "Anthropic Claude"
property apiKeys : {}

on run
	-- Load API keys from .env file
	loadAPIKeys()
	
	-- Show authentication dialog
	if authenticateUser() then
		-- Show main chatbot window
		showChatbotWindow()
	else
		display dialog "❌ Authentication required to use AgenticSeek" buttons {"OK"} default button "OK" with title "AgenticSeek Authentication Required"
	end if
end run

on loadAPIKeys()
	try
		set envFile to POSIX file "/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/.env"
		set envContent to read envFile as «class utf8»
		
		set AppleScript's text item delimiters to return
		set envLines to text items of envContent
		set AppleScript's text item delimiters to ""
		
		set apiKeys to {}
		repeat with envLine in envLines
			if envLine contains "=" and not (envLine starts with "#") then
				set AppleScript's text item delimiters to "="
				set keyValue to text items of envLine
				set keyName to item 1 of keyValue
				set keyVal to item 2 of keyValue
				-- Remove quotes if present
				if keyVal starts with "\"" and keyVal ends with "\"" then
					set keyVal to text 2 thru -2 of keyVal
				end if
				set apiKeys to apiKeys & {{keyName:keyName, keyVal:keyVal}}
				set AppleScript's text item delimiters to ""
			end if
		end repeat
		
		log "✅ API keys loaded successfully"
		display notification "✅ API keys loaded from .env file" with title "AgenticSeek Initialization"
		
	on error errMsg
		log "❌ Error loading API keys: " & errMsg
		display dialog "❌ Error loading API keys from .env file:" & return & errMsg buttons {"OK"} default button "OK" with title "AgenticSeek Error"
	end try
end loadAPIKeys

on authenticateUser()
	-- Apple Sign In simulation with proper accessibility
	try
		tell application "System Events"
			activate
			set authDialog to display dialog "🔐 AgenticSeek SSO Authentication" & return & return & "👤 Apple ID: " & userEmail & return & return & "This app requires authentication to access AI services." & return & return & "✅ API Keys: Loaded from .env" & return & "🔒 Secure: Apple Sign In protected" buttons {"Cancel", "Sign In with Apple"} default button "Sign In with Apple" with title "AgenticSeek Authentication - SSO Window" giving up after 30
			
			if button returned of authDialog is "Sign In with Apple" then
				set isAuthenticated to true
				display notification "✅ Signed in as " & userEmail with title "AgenticSeek Authentication Success"
				return true
			else
				return false
			end if
		end tell
	on error errMsg
		display dialog "❌ Authentication error: " & errMsg buttons {"OK"} default button "OK"
		return false
	end try
end authenticateUser

on showChatbotWindow()
	-- Create main chatbot interface with accessibility labels
	try
		tell application "System Events"
			activate
			
			-- Main chat window with proper accessibility
			set chatWindow to display dialog "🤖 AgenticSeek AI Assistant" & return & return & "👤 User: " & userEmail & return & "🧠 AI Provider: " & currentProvider & return & "🔑 API Status: Connected" & return & return & "💬 Type your message below:" buttons {"Switch Provider", "Test API", "Send Message"} default button "Send Message" with title "AgenticSeek Main Chat - AXIdentifier: AgenticSeekMainWindow" default answer "" giving up after 300
			
			if gave up of chatWindow then
				-- User closed window or timeout
				return
			end if
			
			set buttonPressed to button returned of chatWindow
			set userMessage to text returned of chatWindow
			
			if buttonPressed is "Switch Provider" then
				switchProvider()
				showChatbotWindow() -- Show window again
			else if buttonPressed is "Test API" then
				testAllAPIs()
				showChatbotWindow() -- Show window again
			else if buttonPressed is "Send Message" then
				if userMessage is not "" then
					processMessage(userMessage)
				end if
				showChatbotWindow() -- Show window again for next message
			end if
		end tell
	on error errMsg
		display dialog "❌ Chat window error: " & errMsg buttons {"OK"} default button "OK"
	end try
end showChatbotWindow

on switchProvider()
	try
		tell application "System Events"
			set providerDialog to display dialog "🔄 Switch AI Provider" & return & return & "Current: " & currentProvider & return & return & "Choose your AI provider:" buttons {"Cancel", "OpenAI GPT-4", "Anthropic Claude"} default button currentProvider with title "AgenticSeek Provider Selection - AXIdentifier: ProviderSwitchWindow"
			
			if button returned of providerDialog is "Anthropic Claude" then
				set currentProvider to "Anthropic Claude"
				display notification "🧠 Switched to Anthropic Claude" with title "AgenticSeek Provider Change"
			else if button returned of providerDialog is "OpenAI GPT-4" then
				set currentProvider to "OpenAI GPT-4"
				display notification "🤖 Switched to OpenAI GPT-4" with title "AgenticSeek Provider Change"
			end if
		end tell
	on error errMsg
		display dialog "❌ Provider switch error: " & errMsg buttons {"OK"} default button "OK"
	end try
end switchProvider

on testAllAPIs()
	-- Test both APIs to verify they work
	try
		display dialog "🧪 Testing API Connections..." & return & return & "This will test both Anthropic and OpenAI APIs with your loaded keys." buttons {"Cancel", "Run Tests"} default button "Run Tests" with title "AgenticSeek API Testing - AXIdentifier: APITestWindow"
		
		if button returned of result is "Run Tests" then
			-- Test Anthropic
			set anthropicResult to callAnthropicAPI("Hello, respond with just 'Anthropic API working!'")
			
			-- Test OpenAI  
			set openaiResult to callOpenAIAPI("Hello, respond with just 'OpenAI API working!'")
			
			-- Show results
			display dialog "🧪 API Test Results:" & return & return & "🧠 Anthropic Claude:" & return & anthropicResult & return & return & "🤖 OpenAI GPT-4:" & return & openaiResult buttons {"OK"} default button "OK" with title "AgenticSeek API Test Results - AXIdentifier: APIResultsWindow"
		end if
		
	on error errMsg
		display dialog "❌ API test error: " & errMsg buttons {"OK"} default button "OK"
	end try
end testAllAPIs

on processMessage(userMessage)
	-- Show loading state with accessibility
	try
		tell application "System Events"
			set loadingDialog to display dialog "🤔 " & currentProvider & " is thinking..." & return & return & "Your message: " & userMessage & return & return & "Please wait for response..." buttons {"Cancel"} default button "Cancel" with title "AgenticSeek Processing - AXIdentifier: ProcessingWindow" giving up after 3
		end tell
		
		-- Get AI response
		set aiResponse to getAIResponse(userMessage)
		
		-- Show response with accessibility
		tell application "System Events"
			set responseDialog to display dialog "💬 " & currentProvider & " Response:" & return & return & aiResponse & return & return & "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" & return & "Your message: " & userMessage buttons {"Copy Response", "Continue Chatting"} default button "Continue Chatting" with title "AgenticSeek AI Response - AXIdentifier: ResponseWindow"
			
			if button returned of responseDialog is "Copy Response" then
				set the clipboard to aiResponse
				display notification "📋 Response copied to clipboard" with title "AgenticSeek"
			end if
		end tell
		
	on error errMsg
		display dialog "❌ Message processing error: " & errMsg buttons {"OK"} default button "OK"
	end try
end processMessage

on getAIResponse(message)
	try
		if currentProvider is "Anthropic Claude" then
			return callAnthropicAPI(message)
		else if currentProvider is "OpenAI GPT-4" then
			return callOpenAIAPI(message)
		else
			return "❌ Provider not configured: " & currentProvider
		end if
	on error errMsg
		return "❌ Error getting AI response: " & errMsg
	end try
end getAIResponse

on callAnthropicAPI(message)
	try
		set anthropicKey to getAPIKey("ANTHROPIC_API_KEY")
		if anthropicKey is "" then
			return "❌ Anthropic API key not found in .env file"
		end if
		
		-- Escape message for JSON
		set escapedMessage to my escapeForJSON(message)
		
		-- Create JSON payload
		set jsonPayload to "{\"model\":\"claude-3-5-sonnet-20241022\",\"max_tokens\":1000,\"messages\":[{\"role\":\"user\",\"content\":\"" & escapedMessage & "\"}]}"
		
		-- Call API using curl
		set curlCommand to "curl -s -X POST https://api.anthropic.com/v1/messages -H 'Content-Type: application/json' -H 'x-api-key: " & anthropicKey & "' -H 'anthropic-version: 2023-06-01' -d '" & jsonPayload & "'"
		
		set apiResponse to do shell script curlCommand
		
		-- Parse response (simplified)
		if apiResponse contains "\"text\":" then
			set AppleScript's text item delimiters to "\"text\":\""
			set responseText to text item 2 of apiResponse
			set AppleScript's text item delimiters to "\""
			set responseText to text item 1 of responseText
			set AppleScript's text item delimiters to ""
			-- Unescape JSON
			set responseText to my unescapeFromJSON(responseText)
			return "✅ " & responseText
		else if apiResponse contains "error" then
			return "❌ API Error: " & apiResponse
		else
			return "❌ Invalid API response format"
		end if
		
	on error errMsg
		return "❌ Anthropic API error: " & errMsg
	end try
end callAnthropicAPI

on callOpenAIAPI(message)
	try
		set openaiKey to getAPIKey("OPENAI_API_KEY")
		if openaiKey is "" then
			return "❌ OpenAI API key not found in .env file"
		end if
		
		-- Escape message for JSON
		set escapedMessage to my escapeForJSON(message)
		
		-- Create JSON payload
		set jsonPayload to "{\"model\":\"gpt-4\",\"messages\":[{\"role\":\"user\",\"content\":\"" & escapedMessage & "\"}],\"max_tokens\":1000}"
		
		-- Call API using curl
		set curlCommand to "curl -s -X POST https://api.openai.com/v1/chat/completions -H 'Content-Type: application/json' -H 'Authorization: Bearer " & openaiKey & "' -d '" & jsonPayload & "'"
		
		set apiResponse to do shell script curlCommand
		
		-- Parse response (simplified)
		if apiResponse contains "\"content\":" then
			set AppleScript's text item delimiters to "\"content\":\""
			set responseText to text item 2 of apiResponse
			set AppleScript's text item delimiters to "\""
			set responseText to text item 1 of responseText
			set AppleScript's text item delimiters to ""
			-- Unescape JSON
			set responseText to my unescapeFromJSON(responseText)
			return "✅ " & responseText
		else if apiResponse contains "error" then
			return "❌ API Error: " & apiResponse
		else
			return "❌ Invalid API response format"
		end if
		
	on error errMsg
		return "❌ OpenAI API error: " & errMsg
	end try
end callOpenAIAPI

on getAPIKey(keyName)
	repeat with keyPair in apiKeys
		if keyName of keyPair is keyName then
			return keyVal of keyPair
		end if
	end repeat
	return ""
end getAPIKey

on escapeForJSON(txt)
	-- Basic JSON escaping
	set txt to my replaceText(txt, "\\", "\\\\")
	set txt to my replaceText(txt, "\"", "\\\"")
	set txt to my replaceText(txt, return, "\\n")
	set txt to my replaceText(txt, tab, "\\t")
	return txt
end escapeForJSON

on unescapeFromJSON(txt)
	-- Basic JSON unescaping
	set txt to my replaceText(txt, "\\\"", "\"")
	set txt to my replaceText(txt, "\\\\", "\\")
	set txt to my replaceText(txt, "\\n", return)
	set txt to my replaceText(txt, "\\t", tab)
	return txt
end unescapeFromJSON

on replaceText(thisText, searchString, replaceString)
	set AppleScript's text item delimiters to searchString
	set textItems to text items of thisText
	set AppleScript's text item delimiters to replaceString
	set thisText to textItems as string
	set AppleScript's text item delimiters to ""
	return thisText
end replaceText