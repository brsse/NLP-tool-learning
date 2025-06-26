import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

from qa_test import QATestSuite
from qa import COMPREHENSIVE_QA_DATASET

from prompts import (
    ROUTE_SELECTION_PROMPT,
    ROUTE_SELECTION_PROMPT_SINGLE,
    ROUTE_SELECTION_PROMPT_FEWSHOT,
)

def save_ablation_results(results: Dict[str, Any], test_name: str):
    """Save ablation results to separate JSON files"""
    os.makedirs("ablation_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ablation_results/{test_name}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Saved {test_name} results to {filename}")

def print_comparison_results(results: Dict[str, Any], test_name: str):
    """Print comparison results in a clean format"""
    print(f"\n🏆 {test_name.upper()} RESULTS")
    print("=" * 50)
    
    for comparison_key, data in results.get('comparison_results', {}).items():
        if isinstance(data, dict) and 'winner' in data:
            print(f"\n📊 {comparison_key}")
            for approach, metrics in data.items():
                if isinstance(metrics, dict) and 'avg_quality' in metrics:
                    print(f"   {approach}: {metrics['avg_quality']:.3f} quality, {metrics['avg_time']:.2f}s")
            if 'winner' in data:
                print(f"   🏆 Winner: {data['winner']}")

# Test 1: Tool Learning vs Direct vs Simple
def test_tool_vs_direct_vs_simple(models: List[str] = None, sample_size: int = 3) -> Dict[str, Any]:
    """Compare three approaches: Tool Learning, Direct Search, Simple Baseline"""
    qa_suite = QATestSuite()
    
    if not qa_suite.system_available:
        return {'error': 'System not available'}
    
    if models is None:
        models = ['llama3.2', 'deepseek-r1', 'mistral:7b-instruct']
    
    test_cases = COMPREHENSIVE_QA_DATASET[:sample_size]
    results = {
        'test_name': 'Tool Learning vs Direct vs Simple',
        'timestamp': datetime.now().isoformat(),
        'models_tested': models,
        'sample_size': sample_size,
        'comparison_results': {}
    }
    
    print(f"🧪 Testing Tool Learning vs Direct vs Simple")
    print(f"📊 {len(test_cases)} cases with {len(models)} models")
    
    for model in models:
        print(f"\n🤖 {model}...")
        
        # Test all three approaches
        tool_quality, tool_time = test_tool_learning_approach(qa_suite, model, test_cases)
        direct_quality, direct_time = test_direct_search_approach(qa_suite, model, test_cases)
        simple_quality, simple_time = test_simple_baseline_approach(qa_suite, test_cases)
        
        # Find winner
        approaches = {
            'tool_learning': tool_quality,
            'direct_search': direct_quality,
            'simple_baseline': simple_quality
        }
        winner = max(approaches.items(), key=lambda x: x[1])[0]
        
        results['comparison_results'][model] = {
            'tool_learning': {'avg_quality': tool_quality, 'avg_time': tool_time},
            'direct_search': {'avg_quality': direct_quality, 'avg_time': direct_time},
            'simple_baseline': {'avg_quality': simple_quality, 'avg_time': simple_time},
            'winner': winner
        }
    
    save_ablation_results(results, 'tool_vs_direct_vs_simple')
    print_comparison_results(results, 'Tool vs Direct vs Simple')
    return results

# Test 2: Route Ablation (Single vs Multi-route)
def test_route_ablation(models: List[str] = None, sample_size: int = 3) -> Dict[str, Any]:
    """Compare single route vs multi-route effectiveness"""
    qa_suite = QATestSuite()
    
    if not qa_suite.system_available:
        return {'error': 'System not available'}
    
    if models is None:
        models = ['llama3.2', 'deepseek-r1', 'mistral:7b-instruct']
    
    test_cases = COMPREHENSIVE_QA_DATASET[:sample_size]
    results = {
        'test_name': 'Single Route vs Multi-Route',
        'timestamp': datetime.now().isoformat(),
        'models_tested': models,
        'sample_size': sample_size,
        'comparison_results': {}
    }
    
    print(f"🧪 Testing Single Route vs Multi-Route")
    print(f"📊 {len(test_cases)} cases with {len(models)} models")
    
    for model in models:
        print(f"\n🤖 {model}...")
        
        single_quality, single_time = test_single_route_approach(qa_suite, model, test_cases)
        multi_quality, multi_time = test_multi_route_approach(qa_suite, model, test_cases)
        
        winner = 'multi_route' if multi_quality > single_quality else 'single_route'
        
        results['comparison_results'][model] = {
            'single_route': {'avg_quality': single_quality, 'avg_time': single_time},
            'multi_route': {'avg_quality': multi_quality, 'avg_time': multi_time},
            'quality_improvement': multi_quality - single_quality,
            'winner': winner
        }
    
    save_ablation_results(results, 'route_ablation')
    print_comparison_results(results, 'Route Ablation')
    return results

# Test 3: Static vs Dynamic Data
def test_static_vs_dynamic(models: List[str] = None, sample_size: int = 3) -> Dict[str, Any]:
    """Compare static dataset vs dynamic arXiv API"""
    qa_suite = QATestSuite()
    
    if not qa_suite.system_available:
        return {'error': 'System not available'}
    
    if models is None:
        models = ['llama3.2', 'deepseek-r1', 'mistral:7b-instruct']
    
    test_cases = COMPREHENSIVE_QA_DATASET[:sample_size]
    results = {
        'test_name': 'Static vs Dynamic Data',
        'timestamp': datetime.now().isoformat(),
        'models_tested': models,
        'sample_size': sample_size,
        'comparison_results': {}
    }
    
    print(f"🧪 Testing Static vs Dynamic Data")
    print(f"📊 {len(test_cases)} cases with {len(models)} models")
    
    for model in models:
        print(f"\n🤖 {model}...")
        
        static_quality, static_time = test_static_data_approach(qa_suite, model, test_cases)
        dynamic_quality, dynamic_time = test_dynamic_data_approach(qa_suite, model, test_cases)
        
        winner = 'dynamic' if dynamic_quality > static_quality else 'static'
        
        results['comparison_results'][model] = {
            'static_data': {'avg_quality': static_quality, 'avg_time': static_time},
            'dynamic_data': {'avg_quality': dynamic_quality, 'avg_time': dynamic_time},
            'quality_improvement': dynamic_quality - static_quality,
            'winner': winner
        }
    
    save_ablation_results(results, 'static_vs_dynamic')
    print_comparison_results(results, 'Static vs Dynamic')
    return results

# Test 4: Cross-Model Consistency
def test_cross_model_consistency(models: List[str] = None, sample_size: int = 3) -> Dict[str, Any]:
    """Test consistency across different models"""
    qa_suite = QATestSuite()
    
    if not qa_suite.system_available:
        return {'error': 'System not available'}
    
    if models is None:
        models = ['llama3.2', 'deepseek-r1', 'mistral:7b-instruct']
    
    test_cases = COMPREHENSIVE_QA_DATASET[:sample_size]
    results = {
        'test_name': 'Cross-Model Consistency',
        'timestamp': datetime.now().isoformat(),
        'models_tested': models,
        'sample_size': sample_size,
        'model_results': {},
        'consistency_metrics': {}
    }
    
    print(f"🧪 Testing Cross-Model Consistency")
    print(f"📊 {len(test_cases)} cases with {len(models)} models")
    
    model_responses = {}
    
    for model in models:
        print(f"\n🤖 {model}...")
        model_quality, model_time, responses = test_model_consistency(qa_suite, model, test_cases)
        
        results['model_results'][model] = {
            'avg_quality': model_quality,
            'avg_time': model_time,
            'responses': responses
        }
        model_responses[model] = responses
    
    # Calculate consistency metrics
    if len(models) >= 2:
        consistency_score = calculate_response_consistency(model_responses)
        results['consistency_metrics'] = {
            'overall_consistency': consistency_score,
            'most_consistent_model': max(models, key=lambda m: results['model_results'][m]['avg_quality']),
            'quality_variance': calculate_quality_variance(results['model_results'])
        }
    
    save_ablation_results(results, 'cross_model_consistency')
    print_cross_model_results(results)
    return results

# Test 5: Prompt Engineering 
def test_prompt_ablation(models: List[str] = None, sample_size: int = 3) -> Dict[str, Any]:
    """Ablation test for different route selection prompts"""
    qa_suite = QATestSuite()

    if not qa_suite.system_available:
        return {'error': 'System not available'}
    
    if models is None:
        models = ['deepseek-r1']

    prompt_variants = {
        "baseline": ROUTE_SELECTION_PROMPT,
        "single": ROUTE_SELECTION_PROMPT_SINGLE,
        "fewshot": ROUTE_SELECTION_PROMPT_FEWSHOT,
    }
    
    test_cases = COMPREHENSIVE_QA_DATASET[:sample_size]
    results = {
        'test_name': 'Prompt Engineering Ablation',
        'timestamp': datetime.now().isoformat(),
        'models_tested': models,
        'sample_size': sample_size,
        'prompt_variants': list(prompt_variants.keys()),
        'prompt_results': {}
    }
    
    for model in models:
        print(f"\n🤖 Testing on model: {model}")
        for prompt_name, prompt_template in prompt_variants.items():
            print(f"   📋 Prompt variant: {prompt_name}")
            total_quality = 0
            total_time = 0
            for case in test_cases:
                try:
                    start_time = time.time()
                    
                    tool_system = qa_suite.tool_system_static
                    tool_system.ollama_model = model
                    # Here is the KEY: override the prompt passed to the tool system for routing
                    # (You must implement this if not already supported in your tool system)
                    custom_prompt = prompt_template.format(query=case['query'])
                    routes, confidence, explanation = tool_system.select_route_with_custom_prompt(case['query'], custom_prompt)
                    
                    papers = tool_system.search_papers(case['query'], max_results=5)
                    response = tool_system.generate_response(case['query'], routes, papers)
                    
                    end_time = time.time()
                    
                    quality_scores = qa_suite.quality_evaluator.evaluate_response(
                        case['query'], response, papers, case
                    )
                    
                    # For debugging purposes
                    print(f"Model: {model}, Prompt: {prompt_name}, Query: {case['query']}")
                    print(f"  Routes: {routes}")
                    print(f"  Response: {response[:120]}")
                    print(f"  Quality: {quality_scores['average_quality']}\n")

                    total_quality += quality_scores['average_quality']
                    total_time += (end_time - start_time)
                    
                except Exception as e:
                    print(f"  Prompt ablation error: {e}")
                    total_quality += 0.1
                    total_time += 10.0
            
            avg_quality = total_quality / len(test_cases)
            avg_time = total_time / len(test_cases)
            results['prompt_results'][f"{model}__{prompt_name}"] = {
                'avg_quality': avg_quality,
                'avg_time': avg_time
            }
            print(f"      Quality: {avg_quality:.3f}  Time: {avg_time:.2f}s")
    
    save_ablation_results(results, 'prompt_ablation')
    return results

# Helper functions for different testing approaches
def test_tool_learning_approach(qa_suite: QATestSuite, model: str, test_cases: List[Dict]) -> Tuple[float, float]:
    """Test full tool learning system"""
    total_quality = 0
    total_time = 0
    
    for case in test_cases:
        try:
            start_time = time.time()
            
            tool_system = qa_suite.tool_system_static
            tool_system.ollama_model = model
            
            routes, confidence, explanation = tool_system.select_route(case['query'])
            papers = tool_system.search_papers(case['query'], max_results=5)
            response = tool_system.generate_response(case['query'], routes, papers)
            
            end_time = time.time()
            
            quality_scores = qa_suite.quality_evaluator.evaluate_response(
                case['query'], response, papers, case
            )
            
            total_quality += quality_scores['average_quality']
            total_time += (end_time - start_time)
            
        except Exception as e:
            print(f"  Tool learning error: {e}")
            total_quality += 0.1
            total_time += 10.0
    
    return total_quality / len(test_cases), total_time / len(test_cases)

def test_direct_search_approach(qa_suite: QATestSuite, model: str, test_cases: List[Dict]) -> Tuple[float, float]:
    """Test direct search without route selection"""
    total_quality = 0
    total_time = 0
    
    for case in test_cases:
        try:
            start_time = time.time()
            
            tool_system = qa_suite.tool_system_static
            tool_system.ollama_model = model
            
            # Skip route selection, go directly to search
            papers = tool_system.search_papers(case['query'], max_results=5)
            response = tool_system.generate_response(case['query'], ['search'], papers)
            
            end_time = time.time()
            
            quality_scores = qa_suite.quality_evaluator.evaluate_response(
                case['query'], response, papers, case
            )
            
            total_quality += quality_scores['average_quality']
            total_time += (end_time - start_time)
            
        except Exception as e:
            print(f"  Direct search error: {e}")
            total_quality += 0.1
            total_time += 8.0
    
    return total_quality / len(test_cases), total_time / len(test_cases)

def test_simple_baseline_approach(qa_suite: QATestSuite, test_cases: List[Dict]) -> Tuple[float, float]:
    """Test simple baseline using expected answers"""
    total_quality = 0
    total_time = 0
    
    for case in test_cases:
        try:
            start_time = time.time()
            
            response = case.get('expected_answer', 'No answer available.')
            papers = []
            
            end_time = time.time()
            
            quality_scores = qa_suite.quality_evaluator.evaluate_response(
                case['query'], response, papers, case
            )
            
            total_quality += quality_scores['average_quality']
            total_time += (end_time - start_time)
            
        except Exception as e:
            print(f"  Simple baseline error: {e}")
            total_quality += 0.05
            total_time += 0.001
    
    return total_quality / len(test_cases), total_time / len(test_cases)

def test_single_route_approach(qa_suite: QATestSuite, model: str, test_cases: List[Dict]) -> Tuple[float, float]:
    """Test forcing single route selection"""
    total_quality = 0
    total_time = 0
    
    for case in test_cases:
        try:
            start_time = time.time()
            
            tool_system = qa_suite.tool_system_static
            tool_system.ollama_model = model
            
            # Force single route by taking first expected route
            single_route = case['expected_route'].split(',')[0].strip()
            papers = tool_system.search_papers(case['query'], max_results=5)
            response = tool_system.generate_response(case['query'], [single_route], papers)
            
            end_time = time.time()
            
            quality_scores = qa_suite.quality_evaluator.evaluate_response(
                case['query'], response, papers, case
            )
            
            total_quality += quality_scores['average_quality']
            total_time += (end_time - start_time)
            
        except Exception as e:
            print(f"  Single route error: {e}")
            total_quality += 0.1
            total_time += 8.0
    
    return total_quality / len(test_cases), total_time / len(test_cases)

def test_multi_route_approach(qa_suite: QATestSuite, model: str, test_cases: List[Dict]) -> Tuple[float, float]:
    """Test full multi-route selection"""
    return test_tool_learning_approach(qa_suite, model, test_cases)

def test_static_data_approach(qa_suite: QATestSuite, model: str, test_cases: List[Dict]) -> Tuple[float, float]:
    """Test using static dataset"""
    total_quality = 0
    total_time = 0
    
    for case in test_cases:
        try:
            start_time = time.time()
            
            tool_system = qa_suite.tool_system_static
            tool_system.ollama_model = model
            
            routes, confidence, explanation = tool_system.select_route(case['query'])
            papers = tool_system.search_papers(case['query'], max_results=5)
            response = tool_system.generate_response(case['query'], routes, papers)
            
            end_time = time.time()
            
            quality_scores = qa_suite.quality_evaluator.evaluate_response(
                case['query'], response, papers, case
            )
            
            total_quality += quality_scores['average_quality']
            total_time += (end_time - start_time)
            
        except Exception as e:
            print(f"  Static data error: {e}")
            total_quality += 0.1
            total_time += 10.0
    
    return total_quality / len(test_cases), total_time / len(test_cases)

def test_dynamic_data_approach(qa_suite: QATestSuite, model: str, test_cases: List[Dict]) -> Tuple[float, float]:
    """Test using dynamic arXiv API"""
    total_quality = 0
    total_time = 0
    
    for case in test_cases:
        try:
            start_time = time.time()
            
            tool_system = qa_suite.tool_system_dynamic
            tool_system.ollama_model = model
            
            routes, confidence, explanation = tool_system.select_route(case['query'])
            papers = tool_system.search_papers(case['query'], max_results=5)
            response = tool_system.generate_response(case['query'], routes, papers)
            
            end_time = time.time()
            
            quality_scores = qa_suite.quality_evaluator.evaluate_response(
                case['query'], response, papers, case
            )
            
            total_quality += quality_scores['average_quality']
            total_time += (end_time - start_time)
            
        except Exception as e:
            print(f"  Dynamic data error: {e}")
            total_quality += 0.1
            total_time += 15.0
    
    return total_quality / len(test_cases), total_time / len(test_cases)

def test_model_consistency(qa_suite: QATestSuite, model: str, test_cases: List[Dict]) -> Tuple[float, float, List[str]]:
    """Test model and collect responses for consistency analysis"""
    total_quality = 0
    total_time = 0
    responses = []
    
    for case in test_cases:
        try:
            start_time = time.time()
            
            tool_system = qa_suite.tool_system_static
            tool_system.ollama_model = model
            
            routes, confidence, explanation = tool_system.select_route(case['query'])
            papers = tool_system.search_papers(case['query'], max_results=5)
            response = tool_system.generate_response(case['query'], routes, papers)
            
            end_time = time.time()
            
            quality_scores = qa_suite.quality_evaluator.evaluate_response(
                case['query'], response, papers, case
            )
            
            total_quality += quality_scores['average_quality']
            total_time += (end_time - start_time)
            responses.append(response[:200])  # Store first 200 chars for comparison
            
        except Exception as e:
            print(f"  Model consistency error: {e}")
            total_quality += 0.1
            total_time += 10.0
            responses.append("Error occurred")
    
    return total_quality / len(test_cases), total_time / len(test_cases), responses

def calculate_response_consistency(model_responses: Dict[str, List[str]]) -> float:
    """Calculate consistency score across models"""
    if len(model_responses) < 2:
        return 1.0
    
    models = list(model_responses.keys())
    total_similarity = 0
    comparisons = 0
    
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            responses1 = model_responses[model1]
            responses2 = model_responses[model2]
            
            for r1, r2 in zip(responses1, responses2):
                # Simple word overlap similarity
                words1 = set(r1.lower().split())
                words2 = set(r2.lower().split())
                if words1 or words2:
                    similarity = len(words1 & words2) / len(words1 | words2)
                    total_similarity += similarity
                    comparisons += 1
    
    return total_similarity / comparisons if comparisons > 0 else 0.0

def calculate_quality_variance(model_results: Dict[str, Dict]) -> float:
    """Calculate variance in quality scores across models"""
    qualities = [result['avg_quality'] for result in model_results.values()]
    if len(qualities) < 2:
        return 0.0
    
    mean_quality = sum(qualities) / len(qualities)
    variance = sum((q - mean_quality) ** 2 for q in qualities) / len(qualities)
    return variance

def print_cross_model_results(results: Dict[str, Any]):
    """Print cross-model consistency results"""
    print(f"\n🏆 CROSS-MODEL CONSISTENCY RESULTS")
    print("=" * 50)
    
    for model, data in results['model_results'].items():
        print(f"\n🤖 {model}")
        print(f"   Quality: {data['avg_quality']:.3f}")
        print(f"   Time: {data['avg_time']:.2f}s")
    
    if 'consistency_metrics' in results:
        metrics = results['consistency_metrics']
        print(f"\n📊 Consistency Metrics:")
        print(f"   Overall Consistency: {metrics['overall_consistency']:.3f}")
        print(f"   Quality Variance: {metrics['quality_variance']:.4f}")
        print(f"   Most Consistent: {metrics['most_consistent_model']}")

def main():
    """Run all ablation tests"""
    print("🧪 Comprehensive Ablation Testing Suite")
    print("=" * 45)
    
    # models = ['llama3.2', 'deepseek-r1', 'mistral:7b-instruct']
    models = ['mistral:7b-instruct']

    sample_size = 3
    
    test_map = {
        "tool_vs_direct_vs_simple": test_tool_vs_direct_vs_simple,
        "route_ablation": test_route_ablation,
        "static_vs_dynamic": test_static_vs_dynamic,
        "model_comparison": test_cross_model_consistency,
        "prompt_ablation": test_prompt_ablation,
    }

    if len(sys.argv) > 1:
        # run test specified in the argument
        test_name = sys.argv[1]
        test_func = test_map.get(test_name)
        if test_func:
            print(f"\nRunning test: {test_name}")
            try:
                result = test_func(models, sample_size)
                if 'error' in result:
                    print(f"❌ Test failed: {result['error']}")
                else:
                    print("✅ Test completed successfully!")
            except Exception as e:
                print(f"❌ Test failed with error: {e}")
        else:
            print(f"❌ Unknown test: {test_name}")
            print(f"Available tests: {list(test_map.keys())}")
    else:
        # run all the tests
        print("Running all tests...")
        for name, func in test_map.items():
            print(f"\n{'='*20} {name.replace('_', ' ').title()} {'='*20}")
            try:
                result = func(models, sample_size)
                if 'error' in result:
                    print(f"❌ {name} failed: {result['error']}")
                else:
                    print(f"✅ {name} completed!")
            except Exception as e:
                print(f"❌ {name} failed with error: {e}")
    
    print(f"\n🎉 All ablation tests completed!")
    print(f"📁 Results saved in ablation_results/ directory")

if __name__ == "__main__":
    main() 