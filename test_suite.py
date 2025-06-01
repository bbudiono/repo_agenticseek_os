#!/usr/bin/env python3
"""
Comprehensive Test Suite for AgenticSeek Backend
Tests all endpoints and functionality
"""

import asyncio
import httpx
import json
import time
from pathlib import Path

class AgenticSeekTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
        self.test_results = []
        
    async def run_test(self, test_name, test_func):
        """Run a single test and record results"""
        print(f"🧪 Running test: {test_name}")
        try:
            start_time = time.time()
            result = await test_func()
            duration = time.time() - start_time
            
            if result:
                print(f"✅ {test_name} - PASSED ({duration:.2f}s)")
                self.test_results.append({"test": test_name, "status": "PASSED", "duration": duration})
            else:
                print(f"❌ {test_name} - FAILED ({duration:.2f}s)")
                self.test_results.append({"test": test_name, "status": "FAILED", "duration": duration})
                
        except Exception as e:
            print(f"💥 {test_name} - ERROR: {str(e)}")
            self.test_results.append({"test": test_name, "status": "ERROR", "error": str(e)})
    
    async def test_server_running(self):
        """Test if the server is running"""
        try:
            response = await self.client.get(f"{self.base_url}/")
            return response.status_code == 200
        except:
            return False
    
    async def test_health_check(self):
        """Test health check endpoint"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                return "backend" in data and data["backend"] == "running"
            return False
        except:
            return False
    
    async def test_query_endpoint(self):
        """Test the query endpoint"""
        try:
            payload = {
                "message": "Hello, this is a test message",
                "session_id": "test_session"
            }
            response = await self.client.post(f"{self.base_url}/query", json=payload)
            if response.status_code == 200:
                data = response.json()
                return ("answer" in data and 
                       "agent_name" in data and 
                       "blocks" in data and
                       "timestamp" in data and
                       "test message" in data["answer"])
            return False
        except:
            return False
    
    async def test_latest_answer_endpoint(self):
        """Test the latest answer endpoint"""
        try:
            # First send a query
            payload = {"message": "Test for latest answer", "session_id": "test_latest"}
            await self.client.post(f"{self.base_url}/query", json=payload)
            
            # Then get the latest answer
            response = await self.client.get(f"{self.base_url}/latest_answer")
            if response.status_code == 200:
                data = response.json()
                return ("answer" in data and 
                       "Test for latest answer" in data["answer"])
            return False
        except:
            return False
    
    async def test_screenshots_endpoint(self):
        """Test screenshots endpoint (404 expected since no screenshots exist)"""
        try:
            response = await self.client.get(f"{self.base_url}/screenshots/test.png")
            # 404 is expected for non-existent screenshots
            return response.status_code == 404
        except:
            return False
    
    async def test_cors_headers(self):
        """Test CORS headers are present"""
        try:
            response = await self.client.options(f"{self.base_url}/query", 
                                                headers={"Origin": "http://localhost:3000"})
            headers = response.headers
            return ("access-control-allow-origin" in headers and 
                   "access-control-allow-methods" in headers)
        except:
            return False
    
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        try:
            tasks = []
            for i in range(5):
                payload = {"message": f"Concurrent test {i}", "session_id": f"session_{i}"}
                task = self.client.post(f"{self.base_url}/query", json=payload)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            return all(r.status_code == 200 for r in responses)
        except:
            return False
    
    async def test_error_handling(self):
        """Test error handling with invalid data"""
        try:
            # Send invalid JSON
            response = await self.client.post(f"{self.base_url}/query", 
                                            data="invalid json")
            # Should return an error status
            return response.status_code >= 400
        except:
            return True  # Exception is also valid error handling
    
    async def run_all_tests(self):
        """Run the complete test suite"""
        print("🚀 Starting AgenticSeek Backend Test Suite")
        print("=" * 50)
        
        tests = [
            ("Server Running", self.test_server_running),
            ("Health Check", self.test_health_check),
            ("Query Endpoint", self.test_query_endpoint),
            ("Latest Answer Endpoint", self.test_latest_answer_endpoint),
            ("Screenshots Endpoint", self.test_screenshots_endpoint),
            ("CORS Headers", self.test_cors_headers),
            ("Concurrent Requests", self.test_concurrent_requests),
            ("Error Handling", self.test_error_handling),
        ]
        
        for test_name, test_func in tests:
            await self.run_test(test_name, test_func)
            await asyncio.sleep(0.5)  # Small delay between tests
        
        print("\n" + "=" * 50)
        print("📊 Test Results Summary:")
        
        passed = sum(1 for r in self.test_results if r["status"] == "PASSED")
        failed = sum(1 for r in self.test_results if r["status"] == "FAILED")
        errors = sum(1 for r in self.test_results if r["status"] == "ERROR")
        total = len(self.test_results)
        
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"💥 Errors: {errors}")
        print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
        
        if failed > 0 or errors > 0:
            print("\n🔍 Failed/Error Tests:")
            for result in self.test_results:
                if result["status"] in ["FAILED", "ERROR"]:
                    print(f"  - {result['test']}: {result['status']}")
                    if "error" in result:
                        print(f"    Error: {result['error']}")
        
        await self.client.aclose()
        return passed == total

async def main():
    """Main test runner"""
    tester = AgenticSeekTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! Backend is fully functional.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the results above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())