"""
CORE QA TESTING FRAMEWORK
Generates comprehensive JSON results containing the 6 core features:
1. Route Performance: Accuracy per route
2. Response Quality: Compare answers with qa.py expected answers
3. Model Analysis: Metrics + time comparison
4. Difficulty Analysis: Performance by easy/medium/hard questions
5. Papers Analysis: Retrieval quantity and quality metrics
6. Summary Metrics: Key performance indicators

This is the PRODUCTION monitoring framework for system performance.
For research ablation studies, see: ablation_test.py
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import logging

# Import our custom modules
from model import ModelManager
from qa import COMPREHENSIVE_QA_DATASET, get_dataset_statistics
import re

logger = logging.getLogger(__name__)

class QualityEvaluator:
    """Response quality evaluation with expected answer comparison"""
    
    def evaluate_response(self, query: str, response: str, papers: List[Dict], test_case: Dict) -> Dict[str, float]:
        """Evaluate response quality across multiple dimensions"""
        scores = {}
        
        scores['keyword_coverage'] = self._evaluate_keyword_coverage(query, response)
        scores['structure_coherence'] = self._evaluate_structure_coherence(response)
        scores['length_optimization'] = self._evaluate_length_optimization(response)
        scores['information_density'] = self._evaluate_information_density(response, papers)
        scores['expected_answer_alignment'] = self._evaluate_expected_answer_alignment(response, test_case)
        
        # Calculate average quality score
        scores['average_quality'] = sum(scores.values()) / len(scores)
        
        return scores
    
    def _evaluate_keyword_coverage(self, query: str, response: str) -> float:
        """Evaluate how well response covers query keywords"""
        query_words = [word.lower() for word in re.findall(r'\b\w+\b', query) if len(word) > 3]
        response_words = [word.lower() for word in re.findall(r'\b\w+\b', response)]
        
        if not query_words:
            return 0.5
        
        coverage = sum(1 for word in query_words if word in response_words) / len(query_words)
        return min(coverage, 1.0)
    
    def _evaluate_structure_coherence(self, response: str) -> float:
        """Evaluate logical structure and flow - enhanced for multi-route analysis"""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        
        # Enhanced introduction detection
        intro_indicators = ['based', 'found', 'according', 'research', 'analysis', 'examining', 'reviewing', 'analyzing']
        has_introduction = any(word in response_lower[:150] for word in intro_indicators)
        
        # Content depth indicators
        has_substantial_content = len(response) > 200
        
        # Multi-route structure indicators
        has_route_organization = any(indicator in response_lower for indicator in [
            'search', 'comparison', 'analysis', 'trend', 'citation', 'author', 'venue'
        ])
        
        # Specific research information
        has_specific_info = (any(char.isdigit() for char in response) or 
                           any(word in response_lower for word in [
                               'paper', 'study', 'research', 'author', 'published', 'year',
                               'method', 'approach', 'finding', 'result', 'impact'
                           ]))
        
        # Enhanced conclusion detection
        conclusion_indicators = ['overall', 'conclusion', 'summary', 'therefore', 'in summary', 'findings suggest']
        has_conclusion = any(word in response_lower[-200:] for word in conclusion_indicators)
        
        # Section organization (for multi-route responses)
        has_sections = response.count('\n\n') > 1 or any(marker in response for marker in ['1.', '2.', '##', '**'])
        
        # Comprehensive analysis indicators
        has_comprehensive_analysis = len([word for word in ['compare', 'analyze', 'examine', 'discuss', 'evaluate'] 
                                        if word in response_lower]) >= 2
        
        structure_elements = [
            has_introduction,
            has_substantial_content, 
            has_route_organization,
            has_specific_info,
            has_conclusion,
            has_sections,
            has_comprehensive_analysis
        ]
        
        structure_score = sum(structure_elements) / len(structure_elements)
        return structure_score
    
    def _evaluate_length_optimization(self, response: str) -> float:
        """Evaluate if response length is appropriate - favor comprehensive responses"""
        length = len(response)
        
        if length < 50:
            return 0.1  # Too short - likely failed
        elif 50 <= length < 150:
            return 0.5  # Brief but acceptable for simple queries
        elif 150 <= length < 400:
            return 0.8  # Good for basic responses
        elif 400 <= length < 1000:
            return 1.0  # Optimal range for comprehensive analysis
        elif 1000 <= length < 2000:
            return 0.95 # Very good - detailed analysis
        elif 2000 <= length < 3000:
            return 0.85 # Good - thorough but getting verbose
        else:
            return 0.6  # Still acceptable for complex multi-route responses
    
    def _evaluate_information_density(self, response: str, papers: List[Dict]) -> float:
        """Evaluate richness of relevant information from papers - enhanced for multi-route analysis"""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        
        # Enhanced research indicators including route-specific terms
        research_indicators = [
            'paper', 'study', 'research', 'author', 'year', 'published',
            'citation', 'impact', 'analysis', 'comparison', 'trend', 'evolution',
            'method', 'approach', 'finding', 'result', 'conclusion', 'abstract'
        ]
        indicator_count = sum(1 for indicator in research_indicators if indicator in response_lower)
        
        # Check for paper citations and content integration
        paper_citations = 0
        content_integration = 0
        if papers:
            for paper in papers[:5]:
                # Check title word matches
                title_words = [word for word in paper.get('title', '').lower().split() if len(word) > 4][:5]
                if any(word in response_lower for word in title_words):
                    paper_citations += 1
                
                # Check author mentions
                authors = paper.get('authors', [])
                if authors and any(author.lower().split()[-1] in response_lower for author in authors[:2]):
                    content_integration += 1
        
        # Multi-route analysis indicators
        route_indicators = [
            'compare', 'comparison', 'versus', 'trend', 'evolution', 'citation',
            'impact', 'related', 'similar', 'venue', 'journal', 'conference'
        ]
        route_analysis_count = sum(1 for indicator in route_indicators if indicator in response_lower)
        
        max_papers = max(len(papers[:5]), 1)
        
        # Enhanced scoring that rewards comprehensive analysis
        density_score = min(
            (indicator_count / 15) * 0.4 +           # Research terminology
            (paper_citations / max_papers) * 0.3 +    # Paper citation integration
            (content_integration / max_papers) * 0.2 + # Author/content mentions
            (route_analysis_count / 8) * 0.1,         # Multi-route analysis terms
            1.0
        )
        
        return density_score
    
    def _evaluate_expected_answer_alignment(self, response: str, test_case: Dict) -> float:
        """Evaluate alignment with expected answer from qa.py"""
        expected_keywords = test_case.get('expected_keywords', [])
        
        if not expected_keywords:
            return 0.5
        
        matched_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in response.lower())
        return matched_keywords / len(expected_keywords)

class QATestSuite:
    """Simplified QA testing with comprehensive JSON output"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.quality_evaluator = QualityEvaluator()
        self.test_cases = COMPREHENSIVE_QA_DATASET
        
        # Import tool learning system
        try:
            from tool_learning import ToolLearningSystem
            self.tool_system_static = ToolLearningSystem(use_static_data=True)
            self.tool_system_dynamic = ToolLearningSystem(use_static_data=False)
            self.system_available = True
        except ImportError:
            logger.warning("Tool learning system not available")
            self.system_available = False
    
    def run_test(self, models: List[str] = None, test_static: bool = True, test_dynamic: bool = True, sample_size: int = None) -> Dict[str, Any]:
        """Run testing and generate JSON results with 8 required features"""
        if not self.system_available:
            return {'error': 'Tool learning system not available'}
        
        if models is None:
            models = self.model_manager.get_installed_models()
            if not models:
                models = [self.model_manager.default_model]
        
        # Use sample for quick testing
        test_cases = self.test_cases[:sample_size] if sample_size else self.test_cases
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Testing {len(test_cases)} test cases with models: {', '.join(models)}")
        
        # Test each model with both static and dynamic data
        for model in models:
            print(f"\n🤖 Testing model: {model}")
            
            if test_static:
                print("   📊 Static data mode...")
                static_results = self._test_single_model(model, use_static=True, test_cases=test_cases)
                self._save_model_results(model, static_results, 'static', timestamp)
            
            if test_dynamic:
                print("   🌐 Dynamic data mode...")
                dynamic_results = self._test_single_model(model, use_static=False, test_cases=test_cases)
                self._save_model_results(model, dynamic_results, 'dynamic', timestamp)
        
        return {'status': 'completed', 'timestamp': timestamp, 'models_tested': models}
    
    def _test_single_model(self, model: str, use_static: bool, test_cases: List[Dict]) -> Dict[str, Any]:
        """Test a single model and generate comprehensive results with 8 features"""
        tool_system = self.tool_system_static if use_static else self.tool_system_dynamic
        data_mode = "static" if use_static else "dynamic"
        
        # Initialize tracking
        route_stats = {}
        difficulty_stats = {}
        papers_found_total = 0
        total_start_time = time.time()
        test_results = []
        errors = []
        
        for i, test_case in enumerate(test_cases):
            try:
                result = self._test_single_case(test_case, model, tool_system)
                test_results.append(result)
                
                # Update statistics
                route = test_case['expected_route']
                difficulty = test_case['difficulty']
                
                # Initialize tracking dictionaries
                if route not in route_stats:
                    route_stats[route] = {'correct': 0, 'total': 0, 'quality_sum': 0.0, 'papers_sum': 0}
                if difficulty not in difficulty_stats:
                    difficulty_stats[difficulty] = {'correct': 0, 'total': 0, 'quality_sum': 0.0, 'papers_sum': 0}
                
                # Update counters
                route_stats[route]['total'] += 1
                difficulty_stats[difficulty]['total'] += 1
                
                # Handle multi-route correctness
                if result['route_correct']:
                    route_stats[route]['correct'] += 1
                    difficulty_stats[difficulty]['correct'] += 1
                elif result.get('route_partial', False):
                    # Give partial credit for multi-route where expected route is included
                    route_stats[route]['correct'] += 0.5
                    difficulty_stats[difficulty]['correct'] += 0.5
                
                quality_score = result['quality_scores']['average_quality']
                papers_found = result['papers_found']
                
                route_stats[route]['quality_sum'] += quality_score
                route_stats[route]['papers_sum'] += papers_found
                difficulty_stats[difficulty]['quality_sum'] += quality_score
                difficulty_stats[difficulty]['papers_sum'] += papers_found
                
                papers_found_total += papers_found
                
            except Exception as e:
                logger.error(f"Test case {i} failed: {e}")
                errors.append({'test_case_id': i, 'error': str(e)})
        
        total_time = time.time() - total_start_time
        total_cases = len(test_cases)
        
        # Calculate comprehensive results with 6 CORE TESTING FEATURES
        results = {
            # Feature 1: Route Performance - Multi-route accuracy analysis
            'route_performance': {
                route: {
                    'accuracy': stats['correct'] / stats['total'],
                    'avg_quality': stats['quality_sum'] / stats['total'],
                    'avg_papers_found': stats['papers_sum'] / stats['total'],
                    'total_cases': stats['total'],
                    'avg_quality_score': stats['quality_sum'] / stats['total']
                } for route, stats in route_stats.items()
            },
            
            # Feature 2: Response Quality - qa.py expected answer comparison
            'response_quality': {
                'overall_avg_quality': sum(r['quality_scores']['average_quality'] for r in test_results) / len(test_results),
                'keyword_coverage_avg': sum(r['quality_scores']['keyword_coverage'] for r in test_results) / len(test_results),
                'expected_answer_alignment_avg': sum(r['quality_scores']['expected_answer_alignment'] for r in test_results) / len(test_results),
                'structure_coherence_avg': sum(r['quality_scores']['structure_coherence'] for r in test_results) / len(test_results),
                'qa_comparison_metrics': {
                    'similarity_scores': [r.get('qa_similarity', 0) for r in test_results],
                    'keyword_match_scores': [r.get('keyword_match', 0) for r in test_results],
                    'expected_answer_coverage': sum(1 for r in test_results if r.get('qa_similarity', 0) > 0.5) / len(test_results)
                }
            },
            
            # Feature 3: Model Analysis - Quality + time benchmarking
            'model_analysis': {
                'model_name': model,
                'data_mode': data_mode,
                'total_test_cases': total_cases,
                'overall_accuracy': sum(r['correct'] for r in route_stats.values()) / total_cases,
                'performance_metrics': {
                    'avg_response_time': total_time / total_cases,
                    'total_time': total_time,
                    'fastest_response': min([r['response_time'] for r in test_results], default=0),
                    'slowest_response': max([r['response_time'] for r in test_results], default=0)
                },
                'quality_benchmarks': {
                    'min_quality': min([r['quality_scores']['average_quality'] for r in test_results], default=0),
                    'max_quality': max([r['quality_scores']['average_quality'] for r in test_results], default=0),
                    'avg_quality': sum([r['quality_scores']['average_quality'] for r in test_results]) / len(test_results) if test_results else 0
                },
                'error_analysis': {
                    'error_rate': len(errors) / total_cases,
                    'error_details': errors
                }
            },
            
            # Feature 4: Difficulty Analysis - Easy/medium/hard breakdown
            'difficulty_analysis': {
                difficulty: {
                    'accuracy': stats['correct'] / stats['total'],
                    'avg_quality': stats['quality_sum'] / stats['total'],
                    'avg_papers_found': stats['papers_sum'] / stats['total'],
                    'total_cases': stats['total'],
                    'performance_trend': 'expected' if difficulty == 'easy' and stats['correct']/stats['total'] > 0.8 else 'review_needed'
                } for difficulty, stats in difficulty_stats.items()
            },
            
            # Feature 5: Papers Analysis + Relevance - Quantity + quality metrics
            'papers_analysis': {
                'retrieval_metrics': {
                    'total_papers_found': papers_found_total,
                    'avg_papers_per_query': papers_found_total / total_cases,
                    'max_papers_found': max([r['papers_found'] for r in test_results], default=0),
                    'min_papers_found': min([r['papers_found'] for r in test_results], default=0),
                    'zero_results_rate': sum(1 for r in test_results if r['papers_found'] == 0) / len(test_results)
                },
                'relevance_quality': {
                    'papers_by_route': {route: stats['papers_sum'] / stats['total'] for route, stats in route_stats.items()},
                    'papers_by_difficulty': {difficulty: stats['papers_sum'] / stats['total'] for difficulty, stats in difficulty_stats.items()},
                    'expected_vs_actual': {
                        'meeting_expectations': sum(1 for r in test_results 
                                                  if r['papers_found'] >= r.get('expected_papers_min', 1)) / len(test_results),
                        'exceeding_expectations': sum(1 for r in test_results 
                                                    if r['papers_found'] > r.get('expected_papers_min', 1) * 2) / len(test_results)
                    }
                }
            },
            
            # Feature 6: Summary Metrics - Key performance indicators
            'summary_metrics': {
                'sample_size': total_cases,
                'overall_success_rate': sum(1 for r in test_results if r['quality_scores']['average_quality'] > 0.5) / len(test_results),
                'avg_response_time': total_time / total_cases,
                'avg_papers_per_query': papers_found_total / total_cases,
                'error_rate': len(errors) / total_cases,
                'best_performing_route': max(route_stats.items(), key=lambda x: x[1]['correct'] / x[1]['total'])[0] if route_stats else 'none',
                'most_challenging_difficulty': min(difficulty_stats.items(), key=lambda x: x[1]['correct'] / x[1]['total'])[0] if difficulty_stats else 'none'
            }
        }
        
        return results
    
    def _test_single_case(self, test_case: Dict, model: str, tool_system) -> Dict[str, Any]:
        """Test a single case with multi-route support"""
        query = test_case['query']
        expected_route = test_case['expected_route']
        
        start_time = time.time()
        
        tool_system.ollama_model = model
        
        # Multi-route selection
        routes, confidence, explanation = tool_system.select_route(query)
        
        # Handle both single and multi-route evaluation
        if isinstance(routes, list):
            selected_routes = routes
        else:
            selected_routes = [routes]
        
        # Route correctness - check if expected route is in selected routes
        route_correct = expected_route in selected_routes
        route_partial = len([r for r in selected_routes if r == expected_route]) > 0
        
        # Paper search
        papers = tool_system.search_papers(query, max_results=10)
        papers_found = len(papers)
        
        # Response generation with multi-route
        response = tool_system.generate_response(query, selected_routes, papers)
        
        # Quality evaluation
        quality_scores = self.quality_evaluator.evaluate_response(query, response, papers, test_case)
        
        # Calculate qa.py comparison metrics
        from qa import calculate_expected_answer_similarity, calculate_keyword_match_score
        
        expected_answer = test_case.get('expected_answer', '')
        expected_keywords = test_case.get('expected_keywords', [])
        
        qa_similarity = calculate_expected_answer_similarity(response, expected_answer)
        keyword_match = calculate_keyword_match_score(response, expected_keywords)
        
        end_time = time.time()
        
        return {
            'query': query,
            'expected_route': expected_route,
            'selected_routes': selected_routes,
            'route_correct': route_correct,
            'route_partial': route_partial,
            'route_count': len(selected_routes),
            'confidence': confidence,
            'papers_found': papers_found,
            'response_length': len(response),
            'response_time': end_time - start_time,
            'quality_scores': quality_scores,
            'qa_similarity': qa_similarity,
            'keyword_match': keyword_match,
            'expected_keywords_found': sum(1 for kw in expected_keywords if kw.lower() in response.lower()),
            'multi_route_info': {
                'routes_selected': len(selected_routes),
                'primary_route_correct': expected_route == selected_routes[0] if selected_routes else False,
                'all_routes': selected_routes
            }
        }
    

    
    def _save_model_results(self, model: str, results: Dict, data_mode: str, timestamp: str):
        """Save results to model/[model_name]/ directory"""
        model_dir = f"model/{model}"
        os.makedirs(model_dir, exist_ok=True)
        
        filename = f"qa_test_results_{data_mode}_{timestamp}.json"
        filepath = os.path.join(model_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"   💾 Saved {data_mode} results to {filepath}")

def main():
    """Run QA testing with sample size for verification"""
    print("🚀 QA Testing System - Sample Run")
    print("=" * 50)
    
    test_suite = QATestSuite()
    
    if not test_suite.system_available:
        print("❌ Tool learning system not available!")
        return
    
    # Run sample test with first 5 test cases to verify functionality
    results = test_suite.run_test(
        models=None,  # Use all available models
        test_static=True,
        test_dynamic=True,
        sample_size=5  # Just 5 test cases for quick verification
    )
    
    if 'error' in results:
        print(f"❌ Testing failed: {results['error']}")
    else:
        print("\n✅ Sample testing completed successfully!")
        print(f"📁 Results saved to model/[model_name]/ directories")
        print("🔍 Each JSON file contains all 6 core features")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 