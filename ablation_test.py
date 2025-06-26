#!/usr/bin/env python3
"""
ABLATION TESTING FRAMEWORK
Research experiments to validate tool learning effectiveness.

This is separate from qa_test.py which focuses on production monitoring.
Ablation studies include:
1. Tool Learning vs Direct Search vs Simple Search
2. Single Route vs Multi-Route
3. Static vs Dynamic Data
"""

import os
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Any
import logging

# Import modules
from qa_test import QATestSuite
from qa import COMPREHENSIVE_QA_DATASET

logger = logging.getLogger(__name__)

class AblationTestSuite:
    """Simple ablation testing focused on core value proposition"""
    
    def __init__(self):
        self.qa_test_suite = QATestSuite()
    
    def run_tool_learning_vs_baseline(self, models: List[str] = None, sample_size: int = 5) -> Dict[str, Any]:
        """Core ablation study: Tool Learning vs Baseline approaches"""
        if not self.qa_test_suite.system_available:
            return {'error': 'Tool learning system not available'}
        
        if models is None:
            models = ['llama3.2']
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'models_tested': models,
            'sample_size': sample_size,
            'comparison_results': {}
        }
        
        test_cases = COMPREHENSIVE_QA_DATASET[:sample_size]
        
        print(f"🧪 Ablation Study: Tool Learning vs Baseline")
        print(f"Testing {len(test_cases)} cases with {len(models)} models")
        print("=" * 50)
        
        for model in models:
            print(f"\n🤖 Testing {model}...")
            
            # Test approaches
            tool_results = self._test_tool_learning(model, test_cases)
            simple_results = self._test_simple_baseline(model, test_cases)
            
            # Compare
            results['comparison_results'][model] = {
                'tool_learning': tool_results,
                'simple_baseline': simple_results,
                'quality_improvement': tool_results['avg_quality'] - simple_results['avg_quality'],
                'time_overhead': tool_results['avg_time'] - simple_results['avg_time'],
                'winner': 'tool_learning' if tool_results['avg_quality'] > simple_results['avg_quality'] else 'simple_baseline'
            }
        
        self._save_and_print_results(results)
        return results
    
    def _test_tool_learning(self, model: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """Test actual tool learning system"""
        total_quality = 0
        total_time = 0
        successful_tests = 0
        
        for case in test_cases:
            try:
                start_time = time.time()
                
                # Use actual tool learning system
                tool_system = self.qa_test_suite.tool_system_static
                tool_system.ollama_model = model
                
                routes, confidence, explanation = tool_system.select_route(case['query'])
                papers = tool_system.search_papers(case['query'], max_results=5)
                response = tool_system.generate_response(case['query'], routes, papers)
                
                end_time = time.time()
                
                # Evaluate
                quality_scores = self.qa_test_suite.quality_evaluator.evaluate_response(
                    case['query'], response, papers, case
                )
                
                total_quality += quality_scores['average_quality']
                total_time += (end_time - start_time)
                successful_tests += 1
                
            except Exception as e:
                logger.error(f"Tool learning failed: {e}")
                total_quality += 0.1
                total_time += 10.0
        
        return {
            'avg_quality': total_quality / len(test_cases),
            'avg_time': total_time / len(test_cases),
            'success_rate': successful_tests / len(test_cases)
        }
    
    def _test_simple_baseline(self, model: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """Test simple baseline without route selection"""
        total_quality = 0
        total_time = 0
        successful_tests = 0
        
        for case in test_cases:
            try:
                start_time = time.time()
                
                # Simple approach: just use expected answer directly
                response = case.get('expected_answer', 'No answer available.')
                papers = []  # No paper search in simple baseline
                
                end_time = time.time()
                
                # Evaluate
                quality_scores = self.qa_test_suite.quality_evaluator.evaluate_response(
                    case['query'], response, papers, case
                )
                
                total_quality += quality_scores['average_quality']
                total_time += (end_time - start_time)
                successful_tests += 1
                
            except Exception as e:
                logger.error(f"Simple baseline failed: {e}")
                total_quality += 0.05
                total_time += 1.0
        
        return {
            'avg_quality': total_quality / len(test_cases),
            'avg_time': total_time / len(test_cases),
            'success_rate': successful_tests / len(test_cases)
        }
    
    def _save_and_print_results(self, results: Dict[str, Any]):
        """Save and print ablation results"""
        # Save results
        os.makedirs("ablation_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"ablation_results/tool_learning_vs_baseline_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏆 ABLATION STUDY RESULTS")
        print("=" * 60)
        
        for model, data in results['comparison_results'].items():
            print(f"\n🤖 Model: {model}")
            print(f"   Tool Learning: Quality={data['tool_learning']['avg_quality']:.3f}, Time={data['tool_learning']['avg_time']:.2f}s")
            print(f"   Simple Baseline: Quality={data['simple_baseline']['avg_quality']:.3f}, Time={data['simple_baseline']['avg_time']:.2f}s")
            print(f"   🏆 Winner: {data['winner']}")
            
            improvement = data['quality_improvement']
            if improvement > 0:
                percentage = (improvement / data['simple_baseline']['avg_quality']) * 100
                print(f"   📈 Tool Learning improves quality by {improvement:.3f} ({percentage:.1f}%)")
            else:
                print(f"   📉 Tool Learning underperforms by {abs(improvement):.3f}")
        
        print(f"\n💾 Results saved to {filename}")

def main():
    """Run simple ablation study"""
    print("🧪 Simple Ablation Testing Framework")
    print("=" * 40)
    
    ablation_suite = AblationTestSuite()
    
    if not ablation_suite.qa_test_suite.system_available:
        print("❌ Tool learning system not available!")
        return
    
    # Run ablation study
    results = ablation_suite.run_tool_learning_vs_baseline(
        models=['llama3.2'],
        sample_size=3  # Very small for quick testing
    )
    
    if 'error' in results:
        print(f"❌ Ablation failed: {results['error']}")
    else:
        print("✅ Ablation study completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 