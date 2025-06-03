#!/usr/bin/env python3
"""
Production Health Check Script for CopilotKit Backend
Comprehensive health monitoring with detailed component checks
"""

import asyncio
import json
import sys
import time
import logging
from typing import Dict, Any, List
import aiohttp
import redis
import psycopg2
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health check system for production monitoring"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.checks = {
            "api_health": self._check_api_health,
            "database": self._check_database,
            "redis": self._check_redis,
            "websocket": self._check_websocket,
            "memory_usage": self._check_memory_usage,
            "error_rates": self._check_error_rates,
            "action_handlers": self._check_action_handlers
        }
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status"""
        start_time = time.time()
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "checks": {},
            "execution_time_ms": 0,
            "version": "1.0.0"
        }
        
        failed_checks = []
        
        for check_name, check_func in self.checks.items():
            try:
                check_result = await check_func()
                results["checks"][check_name] = check_result
                
                if not check_result.get("healthy", False):
                    failed_checks.append(check_name)
                    
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                results["checks"][check_name] = {
                    "healthy": False,
                    "error": str(e),
                    "status": "error"
                }
                failed_checks.append(check_name)
        
        # Determine overall status
        if failed_checks:
            if len(failed_checks) >= len(self.checks) // 2:
                results["overall_status"] = "unhealthy"
            else:
                results["overall_status"] = "degraded"
            results["failed_checks"] = failed_checks
        
        results["execution_time_ms"] = round((time.time() - start_time) * 1000, 2)
        return results
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check basic API health endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/copilotkit/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "healthy": True,
                            "status": "ok",
                            "response_time_ms": response.headers.get("X-Response-Time", "unknown"),
                            "active_components": {
                                "agents": data.get("active_agents", 0),
                                "workflows": data.get("active_workflows", 0),
                                "websockets": data.get("websocket_connections", 0)
                            }
                        }
                    else:
                        return {
                            "healthy": False,
                            "status": f"http_error_{response.status}",
                            "error": f"HTTP {response.status}"
                        }
        except Exception as e:
            return {
                "healthy": False,
                "status": "connection_error",
                "error": str(e)
            }
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check PostgreSQL database connectivity"""
        try:
            import os
            db_url = os.getenv("POSTGRES_URL", "postgresql://copilotkit:password@postgres:5432/copilotkit")
            
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            # Test basic query
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            # Check connection pool status
            cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = 'copilotkit'")
            active_connections = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                "healthy": True,
                "status": "connected",
                "active_connections": active_connections,
                "query_result": result[0] if result else None
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "status": "connection_error",
                "error": str(e)
            }
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance"""
        try:
            import os
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
            
            r = redis.from_url(redis_url, decode_responses=True)
            
            # Test basic operations
            start_time = time.time()
            r.ping()
            ping_time = (time.time() - start_time) * 1000
            
            # Check memory usage
            info = r.info("memory")
            
            return {
                "healthy": True,
                "status": "connected",
                "ping_time_ms": round(ping_time, 2),
                "memory_usage_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                "connected_clients": info.get("connected_clients", 0)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "status": "connection_error",
                "error": str(e)
            }
    
    async def _check_websocket(self) -> Dict[str, Any]:
        """Check WebSocket endpoint availability"""
        try:
            import websockets
            
            uri = f"ws://localhost:8000/api/copilotkit/ws"
            
            start_time = time.time()
            async with websockets.connect(uri, timeout=5) as websocket:
                # Send ping message
                await websocket.send(json.dumps({"type": "ping"}))
                
                # Wait for response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2)
                    connection_time = (time.time() - start_time) * 1000
                    
                    return {
                        "healthy": True,
                        "status": "connected",
                        "connection_time_ms": round(connection_time, 2),
                        "response_received": True
                    }
                except asyncio.TimeoutError:
                    return {
                        "healthy": True,
                        "status": "connected_no_response",
                        "connection_time_ms": round((time.time() - start_time) * 1000, 2),
                        "response_received": False
                    }
                    
        except Exception as e:
            return {
                "healthy": False,
                "status": "connection_error",
                "error": str(e)
            }
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check if memory usage is concerning
            memory_warning = memory.percent > 80
            disk_warning = disk.percent > 85
            
            return {
                "healthy": not (memory_warning or disk_warning),
                "status": "ok" if not (memory_warning or disk_warning) else "warning",
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / 1024 / 1024 / 1024, 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
                "warnings": {
                    "high_memory": memory_warning,
                    "high_disk": disk_warning
                }
            }
            
        except ImportError:
            # psutil not available, skip this check
            return {
                "healthy": True,
                "status": "skipped",
                "reason": "psutil not available"
            }
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "error": str(e)
            }
    
    async def _check_error_rates(self) -> Dict[str, Any]:
        """Check error rates from analytics endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/copilotkit/analytics/errors", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        analytics = data.get("analytics", {})
                        
                        recent_errors = analytics.get("recent_errors", 0)
                        total_errors = analytics.get("total_errors", 0)
                        
                        # Consider unhealthy if error rate is too high
                        error_rate_warning = recent_errors > 50  # More than 50 errors in last hour
                        
                        return {
                            "healthy": not error_rate_warning,
                            "status": "ok" if not error_rate_warning else "high_error_rate",
                            "recent_errors": recent_errors,
                            "total_errors": total_errors,
                            "severity_distribution": analytics.get("severity_distribution", {}),
                            "warning": error_rate_warning
                        }
                    else:
                        return {
                            "healthy": False,
                            "status": "analytics_unavailable",
                            "error": f"HTTP {response.status}"
                        }
        except Exception as e:
            return {
                "healthy": True,  # Don't fail health check if analytics are unavailable
                "status": "analytics_error",
                "error": str(e)
            }
    
    async def _check_action_handlers(self) -> Dict[str, Any]:
        """Check if critical action handlers are working"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/copilotkit/actions", timeout=5) as response:
                    if response.status == 200:
                        actions = await response.json()
                        
                        # Check for critical actions
                        critical_actions = [
                            "coordinate_agents",
                            "get_agent_status",
                            "analyze_agent_performance"
                        ]
                        
                        available_actions = [action.get("name") for action in actions]
                        missing_critical = [action for action in critical_actions if action not in available_actions]
                        
                        return {
                            "healthy": len(missing_critical) == 0,
                            "status": "all_critical_available" if len(missing_critical) == 0 else "missing_critical",
                            "total_actions": len(available_actions),
                            "critical_actions_available": len(critical_actions) - len(missing_critical),
                            "missing_critical_actions": missing_critical
                        }
                    else:
                        return {
                            "healthy": False,
                            "status": "actions_unavailable",
                            "error": f"HTTP {response.status}"
                        }
        except Exception as e:
            return {
                "healthy": False,
                "status": "actions_error",
                "error": str(e)
            }

async def main():
    """Main health check execution"""
    checker = HealthChecker()
    results = await checker.run_health_checks()
    
    # Print results for Docker health check
    print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    if results["overall_status"] == "unhealthy":
        sys.exit(1)
    elif results["overall_status"] == "degraded":
        sys.exit(2)  # Warning status
    else:
        sys.exit(0)  # Healthy

if __name__ == "__main__":
    asyncio.run(main())