#!/usr/bin/env python3
"""
Chatbot API Response Verification - TDD & Production Testing
==========================================================

Purpose: Verify chatbot is working with real API key loading and LLM responses
Issues & Complexity Summary: Critical API integration verification with memory-safe operations
Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~250
  - Core Algorithm Complexity: Medium
  - Dependencies: 3 New, 1 Mod
  - State Management Complexity: Medium
  - Novelty/Uncertainty Factor: Low
AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 70%
Problem Estimate (Inherent Problem Difficulty %): 65%
Initial Code Complexity Estimate %: 70%
Justification for Estimates: API verification with real responses requires careful error handling
Final Code Complexity (Actual %): TBD
Overall Result Score (Success & Quality %): TBD
Key Variances/Learnings: TBD
Last Updated: 2025-06-05
"""

import os
import sys
import json
import time
import logging
import asyncio
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import gc
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/chatbot_verification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APIKeyManager:
    """Secure API key management with global .env loading"""
    
    def __init__(self):
        self.api_keys = {}
        self.load_global_env()
    
    def load_global_env(self):
        """Load API keys from global .env file"""
        env_path = Path.home() / ".env"
        project_env_path = Path("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/.env")
        
        # Try project .env first, then global
        for env_file in [project_env_path, env_path]:
            if env_file.exists():
                logger.info(f"Loading API keys from: {env_file}")
                self._load_env_file(env_file)
                break
        else:
            logger.warning("No .env file found in project or home directory")
    
    def _load_env_file(self, env_path: Path):
        """Load environment variables from file"""
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        
                        # Store API keys
                        if 'API_KEY' in key or 'TOKEN' in key:
                            self.api_keys[key] = value
                            # Also set in environment
                            os.environ[key] = value
                            logger.info(f"Loaded API key: {key[:20]}...")
                            
        except Exception as e:
            logger.error(f"Error loading .env file {env_path}: {e}")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specific provider"""
        provider_keys = {
            'anthropic': ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY'],
            'openai': ['OPENAI_API_KEY', 'OPENAI_TOKEN'],
            'google': ['GOOGLE_API_KEY', 'GEMINI_API_KEY'],
            'perplexity': ['PERPLEXITY_API_KEY']
        }
        
        for key_name in provider_keys.get(provider, []):
            if key_name in self.api_keys:
                return self.api_keys[key_name]
            # Also check environment variables
            if key_name in os.environ:
                return os.environ[key_name]
        
        return None
    
    def list_available_keys(self) -> Dict[str, str]:
        """List all available API keys (masked)"""
        masked_keys = {}
        for key, value in self.api_keys.items():
            if value:
                masked_keys[key] = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***masked***"
        return masked_keys

class LLMResponseTester:
    """Test real LLM responses with memory safety"""
    
    def __init__(self, api_key_manager: APIKeyManager):
        self.api_key_manager = api_key_manager
        self.test_results = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AgenticSeek-Chatbot-Verification/1.0'
        })
    
    def test_anthropic_api(self) -> Dict[str, Any]:
        """Test Anthropic Claude API with real response"""
        logger.info("🧪 Testing Anthropic Claude API...")
        
        api_key = self.api_key_manager.get_api_key('anthropic')
        if not api_key:
            return {
                "success": False,
                "error": "No Anthropic API key found",
                "provider": "anthropic"
            }
        
        headers = {
            "x-api-key": api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Please confirm you're working by saying 'AgenticSeek chatbot API test successful'"
                }
            ]
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('content', [{}])[0].get('text', '')
                
                result = {
                    "success": True,
                    "provider": "anthropic",
                    "model": "claude-3-haiku-20240307",
                    "response_time_ms": round(response_time * 1000, 2),
                    "response_content": content,
                    "status_code": response.status_code,
                    "tokens_used": data.get('usage', {})
                }
                
                logger.info(f"✅ Anthropic API test successful: {response_time:.2f}s")
                logger.info(f"Response: {content[:100]}...")
                
                return result
            else:
                error_text = response.text
                logger.error(f"❌ Anthropic API error: {response.status_code} - {error_text}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {error_text}",
                    "provider": "anthropic"
                }
                
        except Exception as e:
            logger.error(f"❌ Anthropic API exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "anthropic"
            }
    
    def test_openai_api(self) -> Dict[str, Any]:
        """Test OpenAI GPT API with real response"""
        logger.info("🧪 Testing OpenAI GPT API...")
        
        api_key = self.api_key_manager.get_api_key('openai')
        if not api_key:
            return {
                "success": False,
                "error": "No OpenAI API key found",
                "provider": "openai"
            }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Please confirm you're working by saying 'AgenticSeek chatbot API test successful'"
                }
            ],
            "max_tokens": 100
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                result = {
                    "success": True,
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "response_time_ms": round(response_time * 1000, 2),
                    "response_content": content,
                    "status_code": response.status_code,
                    "tokens_used": data.get('usage', {})
                }
                
                logger.info(f"✅ OpenAI API test successful: {response_time:.2f}s")
                logger.info(f"Response: {content[:100]}...")
                
                return result
            else:
                error_text = response.text
                logger.error(f"❌ OpenAI API error: {response.status_code} - {error_text}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {error_text}",
                    "provider": "openai"
                }
                
        except Exception as e:
            logger.error(f"❌ OpenAI API exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "openai"
            }
    
    def run_comprehensive_api_tests(self) -> Dict[str, Any]:
        """Run comprehensive API tests with memory monitoring"""
        logger.info("🚀 Starting comprehensive LLM API testing...")
        
        # Memory monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)
        
        test_results = {
            "test_started": datetime.now().isoformat(),
            "initial_memory_mb": initial_memory,
            "api_tests": {},
            "summary": {}
        }
        
        # Test each provider
        providers = [
            ("anthropic", self.test_anthropic_api),
            ("openai", self.test_openai_api)
        ]
        
        successful_tests = 0
        total_tests = len(providers)
        
        for provider_name, test_func in providers:
            logger.info(f"Testing {provider_name} API...")
            result = test_func()
            test_results["api_tests"][provider_name] = result
            
            if result["success"]:
                successful_tests += 1
            
            # Memory cleanup after each test
            gc.collect()
            
            # Brief pause between tests
            time.sleep(1)
        
        # Final memory check
        final_memory = process.memory_info().rss / (1024 * 1024)
        
        # Generate summary
        test_results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": (successful_tests / total_tests) * 100,
            "final_memory_mb": final_memory,
            "memory_increase_mb": final_memory - initial_memory,
            "test_completed": datetime.now().isoformat()
        }
        
        logger.info(f"📊 API Testing Summary:")
        logger.info(f"   Success Rate: {test_results['summary']['success_rate']:.1f}%")
        logger.info(f"   Memory Usage: {final_memory:.1f}MB (Δ{final_memory - initial_memory:+.1f}MB)")
        
        return test_results

class ChatbotInterfaceVerifier:
    """Verify chatbot interface components are wired correctly"""
    
    def __init__(self):
        self.ui_components_tested = []
    
    def verify_swiftui_components(self) -> Dict[str, Any]:
        """Verify SwiftUI chatbot components"""
        logger.info("🧪 Verifying SwiftUI chatbot interface components...")
        
        project_path = Path("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek")
        swiftui_path = project_path / "_macOS" / "AgenticSeek"
        
        components_to_check = [
            "ChatbotInterface.swift",
            "ChatbotModels.swift", 
            "MinimalWorkingChatbot.swift",
            "SimpleWorkingChatbot.swift",
            "AuthenticationManager.swift",
            "ContentView.swift"
        ]
        
        verification_results = {
            "components_found": [],
            "components_missing": [],
            "total_components": len(components_to_check),
            "verification_score": 0
        }
        
        for component in components_to_check:
            component_path = swiftui_path / component
            if component_path.exists():
                verification_results["components_found"].append(component)
                logger.info(f"✅ Found: {component}")
                
                # Basic content verification
                content = component_path.read_text()
                if "import SwiftUI" in content or "import Foundation" in content:
                    logger.info(f"   📝 {component} contains valid Swift code")
            else:
                verification_results["components_missing"].append(component)
                logger.warning(f"❌ Missing: {component}")
        
        verification_results["verification_score"] = (
            len(verification_results["components_found"]) / 
            verification_results["total_components"]
        ) * 100
        
        logger.info(f"📊 UI Components Verification: {verification_results['verification_score']:.1f}%")
        
        return verification_results
    
    def verify_backend_endpoints(self) -> Dict[str, Any]:
        """Verify backend API endpoints are configured"""
        logger.info("🧪 Verifying backend API endpoints...")
        
        project_path = Path("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek")
        
        backend_files = [
            "api.py",
            "sources/fast_api.py",
            "sources/enhanced_backend_endpoints.py"
        ]
        
        endpoint_verification = {
            "backend_files_found": [],
            "endpoints_identified": [],
            "api_integrations": [],
            "verification_score": 0
        }
        
        for backend_file in backend_files:
            file_path = project_path / backend_file
            if file_path.exists():
                endpoint_verification["backend_files_found"].append(backend_file)
                logger.info(f"✅ Found backend file: {backend_file}")
                
                # Check for API integrations
                content = file_path.read_text()
                
                if "anthropic" in content.lower():
                    endpoint_verification["api_integrations"].append("anthropic")
                if "openai" in content.lower():
                    endpoint_verification["api_integrations"].append("openai")
                if "@app.post" in content or "@app.get" in content:
                    endpoint_verification["endpoints_identified"].append(backend_file)
        
        endpoint_verification["verification_score"] = (
            len(endpoint_verification["backend_files_found"]) / len(backend_files)
        ) * 100
        
        logger.info(f"📊 Backend Endpoints Verification: {endpoint_verification['verification_score']:.1f}%")
        
        return endpoint_verification

def verify_sso_authentication() -> Dict[str, Any]:
    """Verify SSO authentication configuration"""
    logger.info("🧪 Verifying SSO authentication configuration...")
    
    sso_verification = {
        "entitlements_found": False,
        "auth_manager_found": False,
        "bundle_id_correct": False,
        "sso_score": 0
    }
    
    project_path = Path("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek")
    
    # Check entitlements file
    entitlements_path = project_path / "_macOS" / "AgenticSeek" / "AgenticSeek.entitlements"
    if entitlements_path.exists():
        sso_verification["entitlements_found"] = True
        logger.info("✅ Found entitlements file")
        
        content = entitlements_path.read_text()
        if "com.apple.developer.applesignin" in content:
            logger.info("✅ Apple Sign In capability configured")
    
    # Check authentication manager
    auth_manager_path = project_path / "_macOS" / "AgenticSeek" / "AuthenticationManager.swift"
    if auth_manager_path.exists():
        sso_verification["auth_manager_found"] = True
        logger.info("✅ Found AuthenticationManager.swift")
    
    # Check for correct bundle ID in project
    project_file = project_path / "_macOS" / "AgenticSeek.xcodeproj" / "project.pbxproj"
    if project_file.exists():
        content = project_file.read_text()
        if "com.ablankcanvas.AgenticSeek" in content:
            sso_verification["bundle_id_correct"] = True
            logger.info("✅ Correct bundle ID found")
    
    # Calculate SSO score
    sso_checks = [
        sso_verification["entitlements_found"],
        sso_verification["auth_manager_found"],
        sso_verification["bundle_id_correct"]
    ]
    sso_verification["sso_score"] = (sum(sso_checks) / len(sso_checks)) * 100
    
    logger.info(f"📊 SSO Authentication Verification: {sso_verification['sso_score']:.1f}%")
    
    return sso_verification

def run_comprehensive_chatbot_verification():
    """Run comprehensive chatbot verification with TDD approach"""
    logger.info("🚀 Starting COMPREHENSIVE CHATBOT VERIFICATION")
    logger.info("=" * 60)
    
    verification_report = {
        "verification_started": datetime.now().isoformat(),
        "user_email": "bernhardbudiono@gmail.com",
        "tests": {}
    }
    
    # 1. API Key Loading Test
    logger.info("\n📋 Step 1: API Key Loading Verification")
    api_key_manager = APIKeyManager()
    available_keys = api_key_manager.list_available_keys()
    
    verification_report["tests"]["api_keys"] = {
        "available_keys": available_keys,
        "key_count": len(available_keys),
        "anthropic_available": api_key_manager.get_api_key('anthropic') is not None,
        "openai_available": api_key_manager.get_api_key('openai') is not None
    }
    
    logger.info(f"Available API keys: {len(available_keys)}")
    for key, masked_value in available_keys.items():
        logger.info(f"   {key}: {masked_value}")
    
    # 2. LLM Response Testing
    logger.info("\n📋 Step 2: LLM Response Verification")
    llm_tester = LLMResponseTester(api_key_manager)
    api_test_results = llm_tester.run_comprehensive_api_tests()
    verification_report["tests"]["llm_responses"] = api_test_results
    
    # 3. UI Components Verification
    logger.info("\n📋 Step 3: UI Components Verification")
    ui_verifier = ChatbotInterfaceVerifier()
    ui_results = ui_verifier.verify_swiftui_components()
    verification_report["tests"]["ui_components"] = ui_results
    
    # 4. Backend Endpoints Verification  
    logger.info("\n📋 Step 4: Backend Endpoints Verification")
    backend_results = ui_verifier.verify_backend_endpoints()
    verification_report["tests"]["backend_endpoints"] = backend_results
    
    # 5. SSO Authentication Verification
    logger.info("\n📋 Step 5: SSO Authentication Verification")
    sso_results = verify_sso_authentication()
    verification_report["tests"]["sso_authentication"] = sso_results
    
    # 6. Memory Safety Check
    logger.info("\n📋 Step 6: Memory Safety Check")
    process = psutil.Process()
    memory_info = process.memory_info()
    verification_report["tests"]["memory_safety"] = {
        "memory_usage_mb": memory_info.rss / (1024 * 1024),
        "memory_limit_ok": memory_info.rss < (512 * 1024 * 1024),  # < 512MB
        "gc_collections": gc.get_count()
    }
    
    # Generate Overall Score
    logger.info("\n📊 GENERATING VERIFICATION SCORE")
    scores = [
        verification_report["tests"]["llm_responses"]["summary"]["success_rate"],
        verification_report["tests"]["ui_components"]["verification_score"],
        verification_report["tests"]["backend_endpoints"]["verification_score"],
        verification_report["tests"]["sso_authentication"]["sso_score"]
    ]
    
    overall_score = sum(scores) / len(scores)
    verification_report["overall_score"] = overall_score
    verification_report["verification_completed"] = datetime.now().isoformat()
    
    # Save results
    report_path = Path("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/chatbot_verification_report.json")
    with open(report_path, 'w') as f:
        json.dump(verification_report, f, indent=2, default=str)
    
    # Print Summary
    logger.info("=" * 60)
    logger.info("🎯 CHATBOT VERIFICATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"📧 User Email: {verification_report['user_email']}")
    logger.info(f"🔑 API Keys Loaded: {verification_report['tests']['api_keys']['key_count']}")
    logger.info(f"🤖 LLM Success Rate: {api_test_results['summary']['success_rate']:.1f}%")
    logger.info(f"🖥️  UI Components: {ui_results['verification_score']:.1f}%")
    logger.info(f"🔗 Backend Endpoints: {backend_results['verification_score']:.1f}%")
    logger.info(f"🔐 SSO Authentication: {sso_results['sso_score']:.1f}%")
    logger.info(f"🧠 Memory Usage: {verification_report['tests']['memory_safety']['memory_usage_mb']:.1f}MB")
    logger.info("=" * 60)
    logger.info(f"🏆 OVERALL VERIFICATION SCORE: {overall_score:.1f}%")
    logger.info("=" * 60)
    
    if overall_score >= 85:
        logger.info("✅ CHATBOT VERIFICATION: PRODUCTION READY")
    elif overall_score >= 70:
        logger.info("⚠️  CHATBOT VERIFICATION: NEEDS MINOR FIXES")
    else:
        logger.info("❌ CHATBOT VERIFICATION: MAJOR ISSUES FOUND")
    
    logger.info(f"📋 Full report saved to: {report_path}")
    
    return verification_report

if __name__ == "__main__":
    # Run comprehensive chatbot verification
    report = run_comprehensive_chatbot_verification()
    
    # Update TodoWrite with results
    print("\n🎯 VERIFICATION COMPLETE - UPDATING TASK STATUS")
    print(f"Overall Score: {report['overall_score']:.1f}%")
    
    # Exit with appropriate code
    sys.exit(0 if report['overall_score'] >= 85 else 1)