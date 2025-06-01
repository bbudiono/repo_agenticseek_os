// SANDBOX FILE: For testing/development. See .cursorrules.
import SwiftUI
import WebKit

//
// FILE-LEVEL CODE REVIEW & RATING
//
// Purpose: Manages the `WKWebView` instance for the AgenticSeek application, handling web content loading, navigation, and bridging between native macOS and the web-based frontend (React).
//
// Issues & Complexity: `WebViewManager.swift` is a critical component for integrating the web frontend. Its primary strength lies in establishing a robust communication bridge and handling basic webview lifecycle events. However, it exhibits several architectural issues:
// - **Direct Service Health Checks**: The `WebViewManager` directly performs HTTP requests to check backend/frontend health, which is a concern for separation of concerns. This logic should ideally reside in a dedicated `ServiceManager` (which already exists and is being used by `ContentView`). Duplication and inconsistency in service monitoring can lead to hard-to-debug issues.
// - **Hardcoded URLs**: `http://localhost:3000` and `http://localhost:8001` are hardcoded. While acceptable for development, they should be configurable for different environments.
// - **Incomplete Error Handling**: While it attempts to load service instructions on failure, the error handling for webview-specific issues (e.g., network errors, content loading failures) could be more robust and provide better user feedback.
// - **Tight Coupling**: The `ServiceStatus` struct and related logic are defined within or directly managed by `WebViewManager`, creating tight coupling instead of relying on the existing `ServiceManager` as a single source of truth for service status.
// - **Prevention of Reward Hacking**: The file's quality is moderately susceptible to 'reward hacking' if its checks for service health are primarily to pass internal tests rather than truly robustly determine service availability and responsiveness. This is mitigated somewhat by the existence of a separate `ServiceManager` that *should* be the authoritative source of truth.
//
// Key strengths include:
// - **JavaScript Bridge**: Effective implementation of `WKUserContentController` for `window.AgenticSeekNative` bridge, allowing native functionality (notifications, external links, clipboard) to be called from web content.
// - **User Agent Configuration**: Custom user agent helps identify the native application.
// - **Basic UI/Navigation Management**: Handles loading states, back/forward navigation, and title updates.
// - **CSS Injection**: Injects custom CSS for scrollbar styling and user-select, improving the native feel.
//
// Ranking/Rating:
// - Modularity/Separation of Concerns: 5/10 (Mixes webview management with service health checks)
// - Readability: 7/10 (Generally clear, but service logic complicates it)
// - Maintainability: 6/10 (Refactoring service checks would significantly improve it)
// - Architectural Contribution: Medium (Essential for web integration, but could be cleaner)
//
// Overall Code Quality Score: 6/10
//
// Summary: `WebViewManager.swift` is a functional but architecturally flawed component. Its direct involvement in checking service health is a significant design smell that should be addressed by fully delegating service status management to the `ServiceManager`. Refactoring this aspect would greatly improve its modularity, maintainability, and reduce the risk of subtle bugs related to inconsistent service state.

struct ServiceStatus {
    var backendRunning = false
    var frontendRunning = false
    var redisRunning = false
    
    var allServicesRunning: Bool {
        return backendRunning && frontendRunning && redisRunning
    }
}

class WebViewManager: NSObject, ObservableObject {
    @Published var isLoading = false
    @Published var canGoBack = false
    @Published var canGoForward = false
    @Published var title = ""
    
    private var webView: WKWebView?
    
    func setWebView(_ webView: WKWebView) {
        self.webView = webView
        setupWebView()
    }
    
    private func setupWebView() {
        guard let webView = webView else { return }
        
        webView.navigationDelegate = self
        webView.uiDelegate = self
        
        // Configure user agent to identify as AgenticSeek native app
        webView.customUserAgent = "AgenticSeek-macOS/1.0"
        
        // Enable developer tools in debug builds
        #if DEBUG
        webView.configuration.preferences.setValue(true, forKey: "developerExtrasEnabled")
        #endif
        
        // Configure content blocking and security
        let contentController = webView.configuration.userContentController
        
        // Inject CSS for native app styling
        let cssString = """
        body {
            -webkit-user-select: none;
            -webkit-touch-callout: none;
        }
        
        /* Hide scrollbars for more native feel */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(0,0,0,0.3);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0,0,0,0.5);
        }
        """
        
        let cssScript = WKUserScript(
            source: """
            var style = document.createElement('style');
            style.textContent = `\(cssString)`;
            document.head.appendChild(style);
            """,
            injectionTime: .atDocumentEnd,
            forMainFrameOnly: false
        )
        
        contentController.addUserScript(cssScript)
        
        // Add JavaScript bridge for native integration
        let bridgeScript = WKUserScript(
            source: """
            window.AgenticSeekNative = {
                platform: 'macOS',
                version: '1.0',
                
                // Native functions that can be called from React
                showNotification: function(title, message) {
                    window.webkit.messageHandlers.notification.postMessage({
                        title: title,
                        message: message
                    });
                },
                
                openExternal: function(url) {
                    window.webkit.messageHandlers.openExternal.postMessage({
                        url: url
                    });
                },
                
                copyToClipboard: function(text) {
                    window.webkit.messageHandlers.clipboard.postMessage({
                        text: text
                    });
                }
            };
            
            // Notify React that native bridge is ready
            window.dispatchEvent(new CustomEvent('agenticseek-native-ready'));
            """,
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true
        )
        
        contentController.addUserScript(bridgeScript)
        
        // Add message handlers for native integration
        contentController.add(self, name: "notification")
        contentController.add(self, name: "openExternal")
        contentController.add(self, name: "clipboard")
    }
    
    func loadAgenticSeek() {
        guard let webView = webView else { return }
        
        // Try to load local React frontend first
        let frontendURL = URL(string: "http://localhost:3000")!
        
        // Give it a longer timeout and retry logic
        var urlRequest = URLRequest(url: frontendURL)
        urlRequest.timeoutInterval = 10.0
        
        // Check if frontend is running
        let task = URLSession.shared.dataTask(with: urlRequest) { [weak self] _, response, error in
            DispatchQueue.main.async {
                if let httpResponse = response as? HTTPURLResponse,
                   httpResponse.statusCode == 200 {
                    // Frontend is running, load it
                    webView.load(URLRequest(url: frontendURL))
                } else {
                    // Frontend not running, show helpful instructions
                    self?.loadServiceInstructions()
                }
            }
        }
        task.resume()
    }
    
    private func loadServiceInstructions() {
        guard let webView = webView else { return }
        
        // Check actual service status instead of showing static content
        checkServicesAndDisplayStatus { [weak self] serviceStatus in
            DispatchQueue.main.async {
                self?.displayServiceStatus(serviceStatus)
            }
        }
    }
    
    private func checkServicesAndDisplayStatus(completion: @escaping (ServiceStatus) -> Void) {
        var serviceStatus = ServiceStatus()
        let group = DispatchGroup()
        
        // Check backend status
        group.enter()
        checkServiceHealth(url: "http://localhost:8001/health") { isRunning in
            serviceStatus.backendRunning = isRunning
            group.leave()
        }
        
        // Check frontend status
        group.enter()
        checkServiceHealth(url: "http://localhost:3000") { isRunning in
            serviceStatus.frontendRunning = isRunning
            group.leave()
        }
        
        // Check Redis status (backend dependency)
        group.enter()
        checkServiceHealth(url: "http://localhost:8001/config/health") { isRunning in
            serviceStatus.redisRunning = isRunning
            group.leave()
        }
        
        group.notify(queue: .main) {
            completion(serviceStatus)
        }
    }
    
    private func checkServiceHealth(url: String, completion: @escaping (Bool) -> Void) {
        guard let serviceURL = URL(string: url) else {
            completion(false)
            return
        }
        
        let task = URLSession.shared.dataTask(with: serviceURL) { _, response, error in
            if let httpResponse = response as? HTTPURLResponse {
                completion(httpResponse.statusCode == 200)
            } else {
                completion(false)
            }
        }
        task.resume()
        
        // Timeout after 3 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
            task.cancel()
            completion(false)
        }
    }
    
    private func displayServiceStatus(_ status: ServiceStatus) {
        guard let webView = webView else { return }
        
        let statusHTML = generateServiceStatusHTML(status)
        webView.loadHTMLString(statusHTML, baseURL: nil)
    }
    
    private func generateServiceStatusHTML(_ status: ServiceStatus) -> String {
        let backendIcon = status.backendRunning ? "✅" : "❌"
        let frontendIcon = status.frontendRunning ? "✅" : "❌"
        let redisIcon = status.redisRunning ? "✅" : "❌"
        
        let mainMessage = status.allServicesRunning ? 
            "AgenticSeek is running! Redirecting..." : 
            "Some services need to be started"
            
        let actionButton = status.allServicesRunning ?
            """
            <button class="success-btn" onclick="window.location.href='http://localhost:3000'">
                🚀 Open AgenticSeek
            </button>
            """ :
            """
            <button class="refresh-btn" onclick="window.location.reload()">
                🔄 Check Again
            </button>
            """
        
        // Auto-redirect if all services are running
        let autoRedirect = status.allServicesRunning ? 
            "<script>setTimeout(() => window.location.href='http://localhost:3000', 2000);</script>" : ""
            
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AgenticSeek - Service Status</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
                    margin: 0; padding: 40px; text-align: center; min-height: 100vh;
                    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
                    color: white; display: flex; align-items: center; justify-content: center;
                }
                .container {
                    max-width: 500px; background: rgba(255,255,255,0.15);
                    padding: 40px; border-radius: 20px; backdrop-filter: blur(20px);
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                }
                h1 { font-size: 2.2em; margin-bottom: 15px; font-weight: 600; }
                p { font-size: 1.1em; line-height: 1.5; margin-bottom: 25px; opacity: 0.9; }
                .status-grid {
                    display: grid; gap: 15px; margin: 25px 0; text-align: left;
                    background: rgba(0,0,0,0.2); padding: 20px; border-radius: 12px;
                }
                .status-item {
                    display: flex; justify-content: space-between; align-items: center;
                    padding: 8px 0; font-size: 1.1em;
                }
                .refresh-btn, .success-btn {
                    background: #059669; border: none; color: white; padding: 15px 30px;
                    font-size: 16px; border-radius: 25px; cursor: pointer; margin-top: 20px;
                    transition: all 0.3s ease; font-weight: 500;
                }
                .success-btn { background: #16a34a; }
                .refresh-btn:hover { background: #047857; transform: translateY(-1px); }
                .success-btn:hover { background: #15803d; transform: translateY(-1px); }
                .help-text {
                    margin-top: 20px; font-size: 0.9em; opacity: 0.8;
                    background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 AgenticSeek</h1>
                <p>\(mainMessage)</p>
                
                <div class="status-grid">
                    <div class="status-item">
                        <span>Backend API</span>
                        <span>\(backendIcon) \(status.backendRunning ? "Running" : "Stopped")</span>
                    </div>
                    <div class="status-item">
                        <span>React Frontend</span>
                        <span>\(frontendIcon) \(status.frontendRunning ? "Running" : "Stopped")</span>
                    </div>
                    <div class="status-item">
                        <span>Redis Cache</span>
                        <span>\(redisIcon) \(status.redisRunning ? "Connected" : "Disconnected")</span>
                    </div>
                </div>
                
                \(actionButton)
                
                \(status.allServicesRunning ? "" : """
                <div class="help-text">
                    To start services: Open Terminal → Navigate to AgenticSeek folder → Run <code>./start_services.sh</code>
                </div>
                """)
            </div>
            \(autoRedirect)
        </body>
        </html>
        """
    }
    
    func reload() {
        webView?.reload()
    }
    
    func goBack() {
        webView?.goBack()
    }
    
    func goForward() {
        webView?.goForward()
    }
}

// MARK: - WKNavigationDelegate
extension WebViewManager: WKNavigationDelegate {
    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        DispatchQueue.main.async {
            self.isLoading = true
        }
    }
    
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        DispatchQueue.main.async {
            self.isLoading = false
            self.canGoBack = webView.canGoBack
            self.canGoForward = webView.canGoForward
            self.title = webView.title ?? ""
        }
    }
    
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        DispatchQueue.main.async {
            self.isLoading = false
            print("WebView navigation failed: \(error.localizedDescription)")
        }
    }
}

// MARK: - WKUIDelegate
extension WebViewManager: WKUIDelegate {
    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration, for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        // Handle new window requests by opening in default browser
        if let url = navigationAction.request.url {
            NSWorkspace.shared.open(url)
        }
        return nil
    }
}

// MARK: - WKScriptMessageHandler
extension WebViewManager: WKScriptMessageHandler {
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let body = message.body as? [String: Any] else { return }
        
        switch message.name {
        case "notification":
            if let title = body["title"] as? String,
               let message = body["message"] as? String {
                showNotification(title: title, message: message)
            }
            
        case "openExternal":
            if let urlString = body["url"] as? String,
               let url = URL(string: urlString) {
                NSWorkspace.shared.open(url)
            }
            
        case "clipboard":
            if let text = body["text"] as? String {
                let pasteboard = NSPasteboard.general
                pasteboard.clearContents()
                pasteboard.setString(text, forType: .string)
                
                showNotification(title: "Copied", message: "Text copied to clipboard")
            }
            
        default:
            break
        }
    }
    
    private func showNotification(title: String, message: String) {
        let notification = NSUserNotification()
        notification.title = title
        notification.informativeText = message
        notification.soundName = NSUserNotificationDefaultSoundName
        
        NSUserNotificationCenter.default.deliver(notification)
    }
}

// MARK: - SwiftUI WebView Wrapper
struct WebViewRepresentable: NSViewRepresentable {
    let webViewManager: WebViewManager
    
    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        
        // Configure for better performance
        configuration.processPool = WKProcessPool()
        configuration.websiteDataStore = .default()
        
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webViewManager.setWebView(webView)
        
        return webView
    }
    
    func updateNSView(_ nsView: WKWebView, context: Context) {
        // Updates handled by WebViewManager
    }
}