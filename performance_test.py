#!/usr/bin/env python3
"""
Performance test script for the LoL Pick ML API.
Tests the effectiveness of caching and async optimizations.
"""

import asyncio
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

API_BASE_URL = "http://localhost:8100"

def test_prediction_request() -> Dict[str, Any]:
    """Test a single prediction request."""
    payload = {
        "ally_ids": [1, 2, 3, 4],
        "enemy_ids": [5, 6, 7, 8],
        "bans": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "role_id": 0,
        "available_champions": list(range(20, 50)),
        "multipliers": {
            "win_prob": 0.4,
            "kda": 0.2,
            "winrate": 0.15,
            "avg_dmg": 0.1,
            "avg_dmg_taken": -0.1,
            "shielded": 0.0,
            "heals": 0.05,
            "cc_time": 0.05
        },
        "version": "test"
    }
    
    start_time = time.time()
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        end_time = time.time()
        
        return {
            "success": True,
            "response_time": (end_time - start_time) * 1000,
            "status_code": response.status_code,
            "predictions_count": len(response.json().get("predictions", []))
        }
    except Exception as e:
        end_time = time.time()
        return {
            "success": False,
            "response_time": (end_time - start_time) * 1000,
            "error": str(e)
        }

def test_concurrent_predictions(num_requests: int = 10) -> Dict[str, Any]:
    """Test concurrent prediction requests."""
    print(f"Testing {num_requests} concurrent prediction requests...")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(test_prediction_request) for _ in range(num_requests)]
        results = [future.result() for future in futures]
    
    end_time = time.time()
    
    # Analyze results
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]
    
    response_times = [r["response_time"] for r in successful_requests]
    
    summary = {
        "total_requests": num_requests,
        "successful_requests": len(successful_requests),
        "failed_requests": len(failed_requests),
        "total_time": (end_time - start_time) * 1000,
        "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
        "min_response_time": min(response_times) if response_times else 0,
        "max_response_time": max(response_times) if response_times else 0,
        "requests_per_second": num_requests / (end_time - start_time) if end_time > start_time else 0
    }
    
    return summary

def test_model_caching() -> Dict[str, Any]:
    """Test model caching effectiveness."""
    print("Testing model caching effectiveness...")
    
    # First request (cold cache)
    print("  Testing cold cache (first request)...")
    cold_result = test_prediction_request()
    
    # Wait a moment
    time.sleep(0.5)
    
    # Second request (warm cache)
    print("  Testing warm cache (second request)...")
    warm_result = test_prediction_request()
    
    # Third request (should be even faster)
    print("  Testing warm cache (third request)...")
    warm_result_2 = test_prediction_request()
    
    return {
        "cold_cache_time": cold_result["response_time"],
        "warm_cache_time": warm_result["response_time"],
        "warm_cache_time_2": warm_result_2["response_time"],
        "cache_improvement": (
            (cold_result["response_time"] - warm_result["response_time"]) / 
            cold_result["response_time"] * 100
        ) if cold_result["response_time"] > 0 else 0,
        "consistency": abs(warm_result["response_time"] - warm_result_2["response_time"])
    }

def test_api_health() -> Dict[str, Any]:
    """Test API health and get performance stats."""
    try:
        # Test health endpoint
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        health_response.raise_for_status()
        health_data = health_response.json()
        
        # Test performance stats endpoint
        perf_response = requests.get(f"{API_BASE_URL}/performance/stats", timeout=5)
        perf_response.raise_for_status()
        perf_data = perf_response.json()
        
        # Test model cache info
        cache_response = requests.get(f"{API_BASE_URL}/models/cache", timeout=5)
        cache_response.raise_for_status()
        cache_data = cache_response.json()
        
        return {
            "health_status": health_data.get("status"),
            "cached_models": health_data.get("cached_models", 0),
            "performance_stats": perf_data,
            "cache_info": cache_data
        }
    except Exception as e:
        return {
            "error": str(e),
            "health_status": "unknown"
        }

def main():
    """Run all performance tests."""
    print("=" * 60)
    print("LoL Pick ML API Performance Test Suite")
    print("=" * 60)
    
    # Test API health first
    print("\n1. Testing API Health...")
    health_result = test_api_health()
    if health_result.get("health_status") == "ok":
        print("   ✓ API is healthy")
        print(f"   ✓ Cached models: {health_result.get('cached_models', 0)}")
    else:
        print("   ✗ API health check failed")
        print(f"   Error: {health_result.get('error', 'Unknown error')}")
        return
    
    # Test model caching
    print("\n2. Testing Model Caching...")
    cache_result = test_model_caching()
    print(f"   Cold cache time: {cache_result['cold_cache_time']:.2f}ms")
    print(f"   Warm cache time: {cache_result['warm_cache_time']:.2f}ms")
    print(f"   Cache improvement: {cache_result['cache_improvement']:.1f}%")
    print(f"   Consistency: {cache_result['consistency']:.2f}ms variance")
    
    # Test concurrent requests
    print("\n3. Testing Concurrent Requests...")
    concurrent_result = test_concurrent_predictions(10)
    print(f"   Total requests: {concurrent_result['total_requests']}")
    print(f"   Successful: {concurrent_result['successful_requests']}")
    print(f"   Failed: {concurrent_result['failed_requests']}")
    print(f"   Average response time: {concurrent_result['avg_response_time']:.2f}ms")
    print(f"   Requests per second: {concurrent_result['requests_per_second']:.2f}")
    
    # Test higher concurrency
    print("\n4. Testing Higher Concurrency...")
    high_concurrent_result = test_concurrent_predictions(25)
    print(f"   Total requests: {high_concurrent_result['total_requests']}")
    print(f"   Successful: {high_concurrent_result['successful_requests']}")
    print(f"   Failed: {high_concurrent_result['failed_requests']}")
    print(f"   Average response time: {high_concurrent_result['avg_response_time']:.2f}ms")
    print(f"   Requests per second: {high_concurrent_result['requests_per_second']:.2f}")
    
    # Final performance stats
    print("\n5. Final Performance Stats...")
    final_stats = test_api_health()
    if "performance_stats" in final_stats:
        perf_stats = final_stats["performance_stats"]
        print(f"   Total requests processed: {perf_stats.get('request_metrics', {}).get('total_requests', 0)}")
        print(f"   Average prediction time: {perf_stats.get('request_metrics', {}).get('avg_prediction_time', 0):.2f}ms")
        print(f"   Vector pool available: {perf_stats.get('vector_pool', {}).get('available_vectors', 0)}")
        print(f"   Cached models: {perf_stats.get('model_cache', {}).get('cached_models_count', 0)}")
    
    print("\n" + "=" * 60)
    print("Performance test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
