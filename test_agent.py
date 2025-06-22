import json
import jsonlines
import time
import os
from typing import List, Dict, Any
from tool_agent import run_advanced_agent

def load_test_queries(file_path: str = "sample_test_queries.jsonl") -> List[Dict]:
    """Load test queries from JSONL file"""
    queries = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))
        print(f"Loaded {len(queries)} test queries from {file_path}")
    except FileNotFoundError:
        print(f"Error loading test queries from {file_path}: File not found")
    except Exception as e:
        print(f"Error loading test queries from {file_path}: {e}")
    return queries

def run_batch_testing(queries: List[Dict[str, Any]], output_file: str, 
                     start_idx: int = 0, end_idx: int = None) -> List[Dict[str, Any]]:
    """
    Run batch testing on a list of queries
    
    Args:
        queries: List of test queries
        output_file: Path to save results
        start_idx: Starting index for testing
        end_idx: Ending index for testing (exclusive)
        
    Returns:
        List of results in standardized format
    """
    if end_idx is None:
        end_idx = len(queries)
    
    results = []
    total_queries = end_idx - start_idx
    
    print(f"Starting batch testing on {total_queries} queries (indices {start_idx}-{end_idx-1})")
    
    for i in range(start_idx, end_idx):
        if i >= len(queries):
            break
            
        query_data = queries[i]
        query_text = query_data.get('Question', query_data.get('query', ''))
        
        print(f"\n--- Processing Query {i+1}/{total_queries} ---")
        print(f"Query: {query_text[:100]}...")
        
        try:
            # Run the advanced agent in test mode
            result = run_advanced_agent(query_text, test_mode=True)
            
            # Add query index for tracking
            result['query_index'] = i
            result['query_text'] = query_text
            
            results.append(result)
            
            print(f"✅ Completed in {result['exe_time']:.2f}s")
            print(f"🛣️  Route: {result['route']}")
            print(f"🎯 Confidence: {result['confidence']:.1%}")
            
        except Exception as e:
            print(f"❌ Error processing query {i}: {e}")
            # Add error result
            error_result = {
                "query": query_text,
                "result": "exe error",
                "route": "error",
                "exe_time": 0.0,
                "confidence": 0.0,
                "steps_executed": 0,
                "successful_steps": 0,
                "error": str(e),
                "tool_executions": [],
                "query_index": i,
                "query_text": query_text
            }
            results.append(error_result)
        
        # Save results after each query (in case of interruption)
        save_results(results, output_file)
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.1)
    
    return results

def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    Save results to a JSONL file
    
    Args:
        results: List of result dictionaries
        output_file: Path to save the results
    """
    try:
        with jsonlines.open(output_file, 'w') as writer:
            for result in results:
                writer.write(result)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def create_evaluation_summary(results: List[Dict[str, Any]], 
                            expected_routes: List[str] = None) -> Dict[str, Any]:
    """
    Create a summary of evaluation results
    
    Args:
        results: List of results from testing
        expected_routes: List of expected routes for comparison
        
    Returns:
        Summary dictionary with evaluation metrics
    """
    total_queries = len(results)
    successful_executions = sum(1 for r in results if r.get('error') is None)
    execution_errors = sum(1 for r in results if r.get('result') == 'exe error')
    
    # Calculate average metrics
    avg_execution_time = sum(r.get('exe_time', 0) for r in results) / total_queries if total_queries > 0 else 0
    avg_confidence = sum(r.get('confidence', 0) for r in results) / total_queries if total_queries > 0 else 0
    avg_steps = sum(r.get('steps_executed', 0) for r in results) / total_queries if total_queries > 0 else 0
    
    # Route analysis
    route_counts = {}
    for r in results:
        route = r.get('route', 'unknown')
        route_counts[route] = route_counts.get(route, 0) + 1
    
    # Compare with expected routes if provided
    route_accuracy = 0
    if expected_routes and len(expected_routes) == len(results):
        correct_routes = sum(1 for i, r in enumerate(results) 
                           if r.get('route') == expected_routes[i])
        route_accuracy = correct_routes / len(results) if results else 0
    
    summary = {
        "total_queries": total_queries,
        "successful_executions": successful_executions,
        "execution_errors": execution_errors,
        "success_rate": successful_executions / total_queries if total_queries > 0 else 0,
        "avg_execution_time": avg_execution_time,
        "avg_confidence": avg_confidence,
        "avg_steps": avg_steps,
        "route_counts": route_counts,
        "route_accuracy": route_accuracy,
        "error_breakdown": {
            "execution_errors": execution_errors,
            "other_errors": total_queries - successful_executions - execution_errors
        }
    }
    
    return summary

def main():
    """
    Main function for running tests
    """
    # Configuration
    test_file = "sample_test_queries.jsonl"  # Path to your test queries
    output_dir = "results_agent"
    target_version = "agent"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test queries
    print(f"Loading test queries from {test_file}")
    queries = load_test_queries(test_file)
    
    if not queries:
        print("No test queries found. Please check the file path.")
        return
    
    print(f"Loaded {len(queries)} test queries")
    
    # Test configuration
    start_idx = 0
    end_idx = min(10, len(queries))  # Test first 10 queries by default
    
    # Run batch testing
    output_file = f"{output_dir}/00{start_idx//10}.jsonl"
    results = run_batch_testing(queries, output_file, start_idx, end_idx)
    
    # Create evaluation summary
    summary = create_evaluation_summary(results)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Successful Executions: {summary['successful_executions']}")
    print(f"Execution Errors: {summary['execution_errors']}")
    print(f"Success Rate: {summary['success_rate']:.2%}")
    print(f"Average Execution Time: {summary['avg_execution_time']:.2f}s")
    print(f"Average Confidence: {summary['avg_confidence']:.2%}")
    print(f"Average Steps: {summary['avg_steps']:.1f}")
    print(f"Route Accuracy: {summary['route_accuracy']:.2%}")
    
    print("\nRoute Distribution:")
    for route, count in summary['route_counts'].items():
        percentage = count / summary['total_queries'] * 100
        print(f"  {route}: {count} ({percentage:.1f}%)")
    
    # Save summary
    summary_file = f"{output_dir}/evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation summary saved to {summary_file}")

if __name__ == "__main__":
    main() 