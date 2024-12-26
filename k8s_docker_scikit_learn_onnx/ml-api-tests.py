import requests
from concurrent import futures
import time

BASE_URL = "http://localhost:3000"

def test_endpoint(payload, expected_status=200, test_name=""):
    """Helper function to test the API endpoint"""
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        duration = time.time() - start_time
        
        result = {
            "test_name": test_name,
            "status_code": response.status_code,
            "expected_status": expected_status,
            "duration": round(duration, 3),
            "payload": payload,
            "response": response.json() if response.text else None,
            "passed": response.status_code == expected_status
        }
        
        print(f"\n{test_name}")
        print(f"Status: {'✅' if result['passed'] else '❌'}")
        print(f"Duration: {result['duration']}s")
        print(f"Response: {result['response']}")
        return result
        
    except Exception as e:
        print(f"\n{test_name}: ❌ - Exception: {str(e)}")
        return {
            "test_name": test_name,
            "error": str(e),
            "passed": False
        }

# 1. Valid Test Cases
valid_tests = [
    {
        "name": "Basic valid input",
        "payload": {
            "features": [[1.2, 0.5, 3.4, 2.0]]
        }
    },
    {
        "name": "Multiple samples",
        "payload": {
            "features": [
                [1.2, 0.5, 3.4, 2.0],
                [2.3, 1.1, 4.1, 1.5],
                [0.8, 1.7, 2.9, 3.2]
            ]
        }
    },
    {
        "name": "Integer values",
        "payload": {
            "features": [[1, 2, 3, 4]]
        }
    },
]

# 2. Invalid Test Cases
invalid_tests = [
    {
        "name": "Empty features array",
        "payload": {"features": []},
        "expected_status": 400
    },
    {
        "name": "Wrong number of features",
        "payload": {"features": [[1.0, 2.0, 3.0]]},
        "expected_status": 400
    },
    {
        "name": "Missing features key",
        "payload": {"wrong_key": [[1.0, 2.0, 3.0, 4.0]]},
        "expected_status": 400
    },
    {
        "name": "Null values",
        "payload": {"features": [[1.0, None, 3.0, 4.0]]},
        "expected_status": 400
    },
    {
        "name": "String values",
        "payload": {"features": [[1.0, "2.0", 3.0, 4.0]]},
        "expected_status": 400
    },
    {
        "name": "Invalid JSON",
        "payload": "invalid json",
        "expected_status": 400
    },
]

# 3. Load Testing Function
def load_test(num_requests=100, num_workers=10):
    print(f"\nStarting load test with {num_requests} requests ({num_workers} concurrent)")
    payload = {"features": [[1.2, 0.5, 3.4, 2.0]]}
    
    start_time = time.time()
    results = []
    
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_url = [
            executor.submit(requests.post, f"{BASE_URL}/predict", json=payload)
            for _ in range(num_requests)
        ]
        
        for future in futures.as_completed(future_to_url):
            try:
                response = future.result()
                results.append({
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                })
            except Exception as e:
                results.append({
                    "error": str(e)
                })
    
    total_time = time.time() - start_time
    successful_requests = sum(1 for r in results if "status_code" in r and r["status_code"] == 200)
    avg_response_time = sum(r["response_time"] for r in results if "response_time" in r) / len(results)
    
    print(f"\nLoad Test Results:")
    print(f"Total Time: {round(total_time, 2)}s")
    print(f"Successful Requests: {successful_requests}/{num_requests}")
    print(f"Average Response Time: {round(avg_response_time, 3)}s")
    print(f"Requests per Second: {round(num_requests/total_time, 2)}")
    
    return {
        "total_time": total_time,
        "successful_requests": successful_requests,
        "total_requests": num_requests,
        "avg_response_time": avg_response_time,
        "requests_per_second": num_requests/total_time
    }

def run_all_tests():
    print("Starting API Tests...")
    
    # Run valid tests
    print("\n=== Valid Test Cases ===")
    valid_results = [
        test_endpoint(test["payload"], 200, test["name"])
        for test in valid_tests
    ]
    
    # Run invalid tests
    print("\n=== Invalid Test Cases ===")
    invalid_results = [
        test_endpoint(test["payload"], test["expected_status"], test["name"])
        for test in invalid_tests
    ]
    
    # Run load tests
    print("\n=== Load Tests ===")
    load_test_results = load_test(10000, 10)
    
    # Summary
    total_tests = len(valid_results) + len(invalid_results)
    passed_tests = sum(1 for r in valid_results + invalid_results if r.get("passed", False))
    
    print("\n=== Test Summary ===")
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Failed Tests: {total_tests - passed_tests}")
    
    return {
        "valid_results": valid_results,
        "invalid_results": invalid_results,
        "load_test_results": load_test_results,
        "total_tests": total_tests,
        "passed_tests": passed_tests
    }


if __name__ == "__main__":
    run_all_tests()