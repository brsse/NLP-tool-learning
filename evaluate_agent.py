import json
import re
import math
import jsonlines
from typing import List, Dict, Any
from tool_agent import run_advanced_agent

def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from JSONL file"""
    results = []
    try:
        with jsonlines.open(file_path, 'r') as reader:
            for line in reader:
                results.append(line)
    except Exception as e:
        print(f"Error loading results from {file_path}: {e}")
    return results

def load_expected_answers(file_path: str) -> List[Dict[str, Any]]:
    """Load expected answers from JSONL file"""
    expected = []
    try:
        with jsonlines.open(file_path, 'r') as reader:
            for line in reader:
                expected.append(line)
    except Exception as e:
        print(f"Error loading expected answers from {file_path}: {e}")
    return expected

def load_expected_routes(file_path: str) -> List[str]:
    """Load expected routes from text file"""
    routes = []
    try:
        with open(file_path, 'r') as f:
            routes = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading expected routes from {file_path}: {e}")
    return routes

def evaluate_results(answer_list: List[Dict], expect_list: List[Dict], 
                    combination_list: List[str]) -> tuple:
    """
    Evaluate results using the same logic as the original testing framework
    
    Returns:
        tuple: (acc_list, detail_list, wrong_case_list)
    """
    acc_list = []
    wrong_case_list = []
    detail_list = [[0] * 5 for _ in range(len(answer_list))]
    
    result_type_list = [
        'AC: all correct', 
        'PnB: answer correct but parse not the best', 
        'PaE: parse error leads to answer wrong', 
        'PrE: process error(parse correct but answer wrong and not exe error)', 
        'EE: exe error'
    ]
    
    for i in range(min(len(answer_list), len(expect_list))):
        # Check if answer matches expected
        if answer_list[i]['result'] == expect_list[i]['Answer']:
            acc_list.append(1)
            # Check if route matches expected
            if answer_list[i]['route'] == combination_list[i]:
                detail_list[i][0] += 1  # AC: all correct
            else:
                detail_list[i][1] += 1  # PnB: answer correct but parse not the best
        else:
            acc_list.append(0)
            # Check for execution error
            if answer_list[i]['result'] == "exe error":
                detail_list[i][4] += 1  # EE: exe error
            elif answer_list[i]['route'] == combination_list[i]:
                detail_list[i][3] += 1  # PrE: process error
            else:
                detail_list[i][2] += 1  # PaE: parse error
            
            # Add to wrong cases for debugging
            wrong_case_list.append({
                'index': i,
                'query': answer_list[i].get('query', ''),
                'expected_answer': expect_list[i]['Answer'],
                'actual_answer': answer_list[i]['result'],
                'expected_route': combination_list[i],
                'actual_route': answer_list[i]['route'],
                'error_type': result_type_list[detail_list[i].index(1)]
            })
    
    return acc_list, detail_list, wrong_case_list

def calculate_variance(data: List[float]) -> float:
    """Calculate standard deviation of data"""
    data = [d for d in data if d != 0]
    n = len(data)
    if n == 0:
        return 0.00
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    return math.sqrt(variance)

def main():
    """Main evaluation function"""
    # Configuration
    target = 'advanced_agent'
    results_dir = f'results_{target}'
    expected_file = 'sample_test_queries.jsonl'
    combinations_file = 'expected_routes.txt'
    
    # Load expected answers and routes
    print(f"Loading expected answers from {expected_file}")
    expect_list = load_expected_answers(expected_file)
    
    print(f"Loading expected routes from {combinations_file}")
    combination_list = load_expected_routes(combinations_file)
    
    # Process each result file
    acc_list = []
    all_detail_list = []
    all_wrong_cases = []
    
    # For this example, we'll process one result file
    # In practice, you'd loop through multiple files like in the original code
    result_file = f"{results_dir}/000.jsonl"
    
    print(f"Loading results from {result_file}")
    answer_list = load_results(result_file)
    
    if not answer_list:
        print("No results found. Please run the testing first.")
        return
    
    # Evaluate results
    print("Evaluating results...")
    acc_list, detail_list, wrong_cases = evaluate_results(
        answer_list, expect_list, combination_list
    )
    
    all_detail_list.extend(detail_list)
    all_wrong_cases.extend(wrong_cases)
    
    # Calculate accuracy
    if acc_list:
        accuracy = sum(acc_list) / len(acc_list) * 100
        print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Calculate detailed metrics
    if all_detail_list:
        total_cases = len(all_detail_list)
        percentage_list = [[0] * 5 for _ in range(total_cases)]
        
        for j in range(total_cases):
            each = all_detail_list[j]
            for i in range(5):
                if sum(each) > 0:
                    percentage_list[j][i] = 100 * each[i] / sum(each)
        
        # Calculate means for each error type
        pnb = [percentage_list[i][1] for i in range(total_cases)]
        sw = [percentage_list[i][2] for i in range(total_cases)]
        pw = [percentage_list[i][3] for i in range(total_cases)]
        ee = [percentage_list[i][4] for i in range(total_cases)]
        em = [percentage_list[i][0] for i in range(total_cases)]
        
        pnb_mean = sum(pnb) / len(pnb) if pnb else 0
        sw_mean = sum(sw) / len(sw) if sw else 0
        pw_mean = sum(pw) / len(pw) if pw else 0
        ee_mean = sum(ee) / len(ee) if ee else 0
        em_mean = sum(em) / len(em) if em else 0
        
        # Calculate variances
        pnb_v = calculate_variance(pnb)
        sw_v = calculate_variance(sw)
        pw_v = calculate_variance(pw)
        ee_v = calculate_variance(ee)
        em_v = calculate_variance(em)
        
        # Print results in the same format as original
        print(f'\n{pnb_mean:.2f}±{pnb_v:.2f} & {sw_mean:.2f}±{sw_v:.2f} & {pw_mean:.2f}±{pw_v:.2f} & {ee_mean:.2f}±{ee_v:.2f} & {em_mean:.2f}±{em_v:.2f} & {accuracy:.2f}')
        
        # Print detailed breakdown
        print("\nDetailed Breakdown:")
        print(f"AC (All Correct): {em_mean:.2f}±{em_v:.2f}%")
        print(f"PnB (Parse not Best): {pnb_mean:.2f}±{pnb_v:.2f}%")
        print(f"PaE (Parse Error): {sw_mean:.2f}±{sw_v:.2f}%")
        print(f"PrE (Process Error): {pw_mean:.2f}±{pw_v:.2f}%")
        print(f"EE (Execution Error): {ee_mean:.2f}±{ee_v:.2f}%")
    
    # Save wrong cases for analysis
    if all_wrong_cases:
        wrong_cases_file = f"{results_dir}/wrong_cases.json"
        with open(wrong_cases_file, 'w') as f:
            json.dump(all_wrong_cases, f, indent=2)
        print(f"\nWrong cases saved to {wrong_cases_file}")
        
        # Print some examples
        print("\nExample wrong cases:")
        for i, case in enumerate(all_wrong_cases[:3]):
            print(f"Case {i+1}:")
            print(f"  Query: {case['query'][:100]}...")
            print(f"  Expected: {case['expected_answer'][:100]}...")
            print(f"  Actual: {case['actual_answer'][:100]}...")
            print(f"  Error Type: {case['error_type']}")
            print()

if __name__ == "__main__":
    main() 