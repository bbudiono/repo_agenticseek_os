#!/usr/bin/env python3
"""
Quick Validation Test for Apple Silicon LangChain Tools
Tests basic functionality and integration readiness
"""

import asyncio
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def quick_apple_silicon_validation():
    """Quick validation of Apple Silicon LangChain tools"""
    print("🧪 Apple Silicon LangChain Tools - Quick Validation Test")
    print("=" * 60)
    
    test_results = []
    start_time = time.time()
    
    # Test 1: Import and basic initialization
    print("\n🔍 Test 1: Import and Basic Initialization")
    try:
        from sources.langchain_apple_silicon_tools import (
            AppleSiliconEmbeddings,
            AppleSiliconVectorStore,
            AppleSiliconPerformanceMonitor,
            AppleSiliconToolManager,
            OptimizationLevel,
            PerformanceProfile,
            HardwareCapability,
            LANGCHAIN_AVAILABLE
        )
        from sources.apple_silicon_optimization_layer import AppleSiliconOptimizationLayer
        
        print("✅ Successfully imported Apple Silicon LangChain tools")
        print(f"✅ LangChain available: {LANGCHAIN_AVAILABLE}")
        test_results.append(("Import and Initialization", True))
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        test_results.append(("Import and Initialization", False))
        return test_results
    
    # Test 2: Apple Silicon Optimizer
    print("\n🍎 Test 2: Apple Silicon Optimizer")
    try:
        apple_optimizer = AppleSiliconOptimizationLayer()
        print(f"✅ Apple Silicon optimizer initialized")
        print(f"✅ Chip type detected: {apple_optimizer.chip_type}")
        test_results.append(("Apple Silicon Optimizer", True))
        
    except Exception as e:
        print(f"❌ Apple Silicon optimizer failed: {e}")
        test_results.append(("Apple Silicon Optimizer", False))
        return test_results
    
    # Test 3: Embeddings functionality
    print("\n🚀 Test 3: Apple Silicon Embeddings")
    try:
        embeddings = AppleSiliconEmbeddings(
            apple_optimizer=apple_optimizer,
            optimization_level=OptimizationLevel.ENHANCED,
            performance_profile=PerformanceProfile.BALANCED
        )
        
        # Test single embedding
        test_text = "Apple Silicon provides excellent performance for AI workloads"
        embedding = embeddings.embed_query(test_text)
        
        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) == 384, "Embedding should have 384 dimensions"
        
        # Test batch embeddings
        test_docs = [
            "Neural Engine acceleration for machine learning",
            "Metal Performance Shaders for GPU computing",
            "Unified memory architecture benefits"
        ]
        
        batch_embeddings = embeddings.embed_documents(test_docs)
        assert len(batch_embeddings) == len(test_docs), "Should generate embedding for each document"
        
        # Get performance metrics
        perf_metrics = embeddings.get_performance_metrics()
        assert "cache_hit_rate" in perf_metrics, "Should include performance metrics"
        
        print(f"✅ Single embedding: {len(embedding)} dimensions")
        print(f"✅ Batch embeddings: {len(batch_embeddings)} documents")
        print(f"✅ Cache hit rate: {perf_metrics.get('cache_hit_rate', 0):.2%}")
        
        test_results.append(("Apple Silicon Embeddings", True))
        
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        test_results.append(("Apple Silicon Embeddings", False))
    
    # Test 4: Vector Store functionality
    print("\n⚡ Test 4: Apple Silicon Vector Store")
    try:
        vector_store = AppleSiliconVectorStore(
            apple_optimizer=apple_optimizer,
            embeddings=embeddings,
            optimization_level=OptimizationLevel.ENHANCED
        )
        
        # Create test documents
        from langchain.schema import Document
        test_documents = [
            Document(page_content="Apple Silicon M1 chip performance analysis"),
            Document(page_content="Neural Engine AI acceleration capabilities"),
            Document(page_content="Metal GPU programming for machine learning")
        ]
        
        # Add documents
        doc_ids = vector_store.add_documents(test_documents)
        assert len(doc_ids) == len(test_documents), "Should return ID for each document"
        
        # Test similarity search
        query_embedding = embeddings.embed_query("AI performance on Apple Silicon")
        search_results = vector_store._similarity_search_optimized(query_embedding, k=2)
        
        assert len(search_results) <= 2, "Should respect k parameter"
        
        # Get vector store metrics
        vector_metrics = vector_store.get_performance_metrics()
        assert "vector_count" in vector_metrics, "Should include vector metrics"
        
        print(f"✅ Added {len(doc_ids)} documents")
        print(f"✅ Search returned {len(search_results)} results")
        print(f"✅ Vector count: {vector_metrics.get('vector_count', 0)}")
        
        test_results.append(("Apple Silicon Vector Store", True))
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        test_results.append(("Apple Silicon Vector Store", False))
    
    # Test 5: Performance Monitor
    print("\n📊 Test 5: Apple Silicon Performance Monitor")
    try:
        monitor = AppleSiliconPerformanceMonitor(apple_optimizer)
        
        # Get current metrics
        current_metrics = monitor._collect_current_metrics()
        assert current_metrics.chip_type == str(apple_optimizer.chip_type), "Chip type should match"
        
        # Test performance summary
        performance_summary = monitor.get_performance_summary()
        assert "chip_type" in performance_summary, "Should include chip type"
        
        print(f"✅ Performance monitor initialized")
        print(f"✅ Chip type: {current_metrics.chip_type}")
        print(f"✅ CPU utilization: {current_metrics.cpu_utilization:.1f}%")
        print(f"✅ Memory usage: {current_metrics.memory_usage_mb:.1f}MB")
        
        test_results.append(("Apple Silicon Performance Monitor", True))
        
    except Exception as e:
        print(f"❌ Performance monitor test failed: {e}")
        test_results.append(("Apple Silicon Performance Monitor", False))
    
    # Test 6: Tool Manager
    print("\n🔧 Test 6: Apple Silicon Tool Manager")
    try:
        tool_manager = AppleSiliconToolManager(apple_optimizer)
        
        # Test tool retrieval
        retrieved_embeddings = tool_manager.get_embeddings()
        retrieved_vector_store = tool_manager.get_vector_store()
        all_tools = tool_manager.get_all_tools()
        
        assert retrieved_embeddings is not None, "Should retrieve embeddings"
        assert retrieved_vector_store is not None, "Should retrieve vector store"
        
        # Get system summary
        system_summary = tool_manager.get_system_summary()
        assert "chip_type" in system_summary, "Should include system summary"
        
        print(f"✅ Tool manager initialized")
        print(f"✅ Available tools: {len(all_tools)}")
        print(f"✅ System summary generated")
        
        test_results.append(("Apple Silicon Tool Manager", True))
        
    except Exception as e:
        print(f"❌ Tool manager test failed: {e}")
        test_results.append(("Apple Silicon Tool Manager", False))
    
    # Generate test report
    total_time = time.time() - start_time
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "=" * 60)
    print("📋 QUICK VALIDATION RESULTS")
    print("=" * 60)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<40} {status}")
    
    print("-" * 60)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Execution Time: {total_time:.2f}s")
    
    # Readiness assessment
    print(f"\n🚀 READINESS ASSESSMENT")
    print("-" * 30)
    
    if success_rate >= 90:
        print("✅ READY FOR COMPREHENSIVE TESTING")
        print("✅ Core Apple Silicon functionality operational")
        print("✅ Hardware optimization features active")
        print("✅ All major components initialized successfully")
    elif success_rate >= 70:
        print("⚠️  READY FOR BASIC TESTING")
        print("✅ Most core functionality operational")
        print("⚠️  Some optimization features may need attention")
    else:
        print("❌ REQUIRES FIXES BEFORE TESTING")
        print("❌ Critical functionality issues detected")
    
    return test_results

if __name__ == "__main__":
    async def main():
        results = await quick_apple_silicon_validation()
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        if passed == total:
            print("\n🎉 All tests passed! Apple Silicon LangChain Tools validated.")
        else:
            print(f"\n⚠️  {total-passed} test(s) failed. Review implementation.")
        
        return passed == total
    
    asyncio.run(main())