#!/usr/bin/env python3
"""
AgenticSeek Production Startup Script
Optimized startup for production deployment with proper configuration validation.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_environment():
    """Check required environment variables and configurations."""
    required_env_vars = [
        'ANTHROPIC_API_KEY',
        'OPENAI_API_KEY',
        'SEARXNG_BASE_URL'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    # Check work directory
    work_dir = "/Users/bernhardbudiono/Documents/agenticseek_workspace"
    if not os.path.exists(work_dir):
        print(f"📁 Creating work directory: {work_dir}")
        os.makedirs(work_dir, exist_ok=True)
    
    return True

async def test_core_systems():
    """Test core AgenticSeek systems for production readiness."""
    print("🧪 Testing core systems...")
    
    try:
        # Test memory system
        from sources.openai_multi_agent_memory_system import OpenAIMultiAgentCoordinator
        coordinator = OpenAIMultiAgentCoordinator()
        print("✅ OpenAI Multi-Agent Memory System initialized")
        
        # Test enhanced coordination
        from sources.enhanced_multi_agent_coordinator import EnhancedMultiAgentCoordinator
        enhanced = EnhancedMultiAgentCoordinator()
        print("✅ Enhanced Multi-Agent Coordinator initialized")
        
        # Test router
        from sources.simple_router import SimpleAgentRouter
        router = SimpleAgentRouter([])
        print("✅ Agent Router initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Core system test failed: {str(e)}")
        return False

def main():
    """Main production startup routine."""
    print("🚀 AgenticSeek Production Startup")
    print("="*50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check environment
    if not check_environment():
        print("❌ Environment validation failed")
        sys.exit(1)
    
    # Test core systems
    if not asyncio.run(test_core_systems()):
        print("❌ Core system validation failed")
        sys.exit(1)
    
    print("✅ All systems validated - AgenticSeek is production-ready!")
    print("🎯 You can now start the full API server with: python api.py")

if __name__ == "__main__":
    main()