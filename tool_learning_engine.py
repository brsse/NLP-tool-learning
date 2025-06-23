#!/usr/bin/env python3
"""
Enhanced Tool Learning Engine for Research Paper Queries

Improved engine with better pattern matching and confidence scoring
for accurate route selection and ablation testing.
"""

import re
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime

class ToolLearningEngine:
    """
    Enhanced tool learning engine with improved accuracy for routing research paper queries
    """
    
    def __init__(self):
        # Enhanced routes with better patterns and weights
        self.routes = {
            'searchPapers': {
                'description': 'Search for research papers',
                'patterns': [
                    (r'(search|find|look for|get).*(paper|article|publication|research)', 3),
                    (r'(paper|article|research).*(about|on)(?!\s+related)', 3),  # Not "related"
                    (r'(recent|latest|new).*(paper|research|publication)', 2),
                    (r'(literature|papers).*(review|survey)', 2),
                    (r'^(papers?|articles?|research)(?!\s+(by|from|of|related))', 2),  # Exclude author/related queries
                ]
            },
            'getAuthorInfo': {
                'description': 'Get information about authors',
                'patterns': [
                    (r'(author|researcher|scientist).*(info|detail|profile|background)', 4),
                    (r'(who is|about|tell me about)\s+[A-Z][\w\s]+', 5),  # "who is Name" pattern
                    (r'(who is|about|tell me about).*(author|researcher|scientist)', 4),
                    (r'(papers?|research|work|publications?).*by\s+[A-Z][\w\s]+', 5),  # "papers...by Name" (higher weight)
                    (r'by\s+[A-Z][\w\s]+.*(papers?|research|work|publications?)', 5),  # "by Name...papers" (higher weight)
                    (r'(publications? of|works? of|research of)', 3),
                    (r'(author|researcher).*profile', 3),
                    (r'.*by\s+[A-Z][\w\s]+', 4),  # Any query mentioning "by Name"
                    (r'(Dr\.|Prof\.|Professor)\s+[A-Z][\w\s]+', 4),  # "Dr. Name" or "Prof. Name"
                ]
            },
            'getCitations': {
                'description': 'Get citation information',
                'patterns': [
                    (r'(citation|cite).*(count|number|analysis)', 4),
                    (r'(how many|number of).*(citation|cite)', 4),
                    (r'(impact factor|h-index|impact).*(paper|article|journal)', 3),
                    (r'(cited|citing).*(paper|work)', 3),
                    (r'(bibliometric|citation).*(analysis|study)', 3),
                ]
            },
            'getRelatedPapers': {
                'description': 'Find related research',
                'patterns': [
                    (r'(papers?|research|work|studies?).*related.*to', 6),  # "papers related to..." (higher weight)
                    (r'related.*(paper|work|research|studies?)', 5),
                    (r'(similar|comparable).*(paper|work|research|studies?)', 4),
                    (r'(connection|link|relationship).*(research|paper)', 3),
                    (r'(based on|building on|following|extending)', 3),
                    (r'(same topic|similar topic|related field)', 3),
                    (r'(referenced|citing|cited).*(work|paper)', 2),
                ]
            },
            'comparePapers': {
                'description': 'Compare multiple papers or approaches',
                'patterns': [
                    (r'(compare|comparison|versus|vs\.?|against)', 4),
                    (r'(difference|differ|contrast).*(between|among)', 4),
                    (r'(better|best|superior|inferior).*(approach|method|technique)', 3),
                    (r'(advantage|disadvantage|pros|cons)', 3),
                    (r'(which is better|what.*difference)', 3),
                ]
            },
            'trendAnalysis': {
                'description': 'Analyze research trends',
                'patterns': [
                    (r'trends?\s+in\s+[\w\s]+', 5),  # "trends in computer vision"
                    (r'(trend|evolution|development|progress).*(field|area|research)', 4),
                    (r'(over time|temporal|historical|longitudinal)', 3),
                    (r'(advancement|growth|change).*(field|discipline)', 3),
                    (r'(emerging|future|upcoming).*(research|trend)', 3),
                    (r'(timeline|chronological|historical)', 2),
                    (r'evolution\s+of\s+[\w\s]+', 4),  # "evolution of NLP"
                ]
            },
            'journalAnalysis': {
                'description': 'Analyze journals and venues',
                'patterns': [
                    (r'analysis\s+of\s+[\w\s]*journal', 5),  # "analysis of Nature journal"
                    (r'(journal|venue|conference).*(impact|ranking|quality)', 4),
                    (r'(publication|journal).*(metric|analysis|evaluation)', 3),
                    (r'(impact factor|h5.index|ranking)', 4),
                    (r'(best|top|leading).*(journal|venue|conference)', 3),
                    (r'(where to publish|publication venue)', 3),
                    (r'(Nature|Science|IEEE|ACM|PNAS).*journal', 4),  # Specific journal names
                ]
            }
        }
        
        self.query_count = 0
        self.route_history = []
        
        # Keywords that boost/reduce confidence for specific routes
        self.boost_keywords = {
            'searchPapers': ['find', 'search', 'papers', 'articles', 'literature'],
            'getAuthorInfo': ['author', 'researcher', 'scientist', 'profile', 'who'],
            'getCitations': ['citation', 'cite', 'impact', 'h-index', 'cited'],
            'getRelatedPapers': ['related', 'similar', 'connected', 'linked'],
            'comparePapers': ['compare', 'versus', 'vs', 'difference', 'better'],
            'trendAnalysis': ['trend', 'evolution', 'development', 'over time'],
            'journalAnalysis': ['journal', 'venue', 'conference', 'publish']
        }
    
    def select_route(self, query: str) -> Tuple[str, float, str]:
        """
        Select the best route for a given query with improved accuracy
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (route_name, confidence, explanation)
        """
        self.query_count += 1
        query_lower = query.lower().strip()
        
        # Calculate weighted scores for each route
        route_scores = {}
        
        for route_name, route_info in self.routes.items():
            score = 0
            matched_patterns = []
            pattern_details = []
            
            # Pattern matching with weights
            for pattern, weight in route_info['patterns']:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                if matches:
                    score += weight * len(matches)
                    matched_patterns.append(pattern)
                    pattern_details.append(f"{len(matches)} match(es) for '{pattern}' (weight: {weight})")
            
            # Keyword boosting
            keyword_boost = 0
            for keyword in self.boost_keywords.get(route_name, []):
                if keyword.lower() in query_lower:
                    keyword_boost += 1
            
            score += keyword_boost * 0.5
            
            if score > 0:
                route_scores[route_name] = {
                    'score': score,
                    'patterns': matched_patterns,
                    'details': pattern_details,
                    'keyword_boost': keyword_boost
                }
        
        # Select best route and calculate confidence
        if not route_scores:
            # Fallback logic with better defaults
            if any(word in query_lower for word in ['author', 'researcher', 'who is']):
                best_route = 'getAuthorInfo'
                confidence = 0.4
                explanation = "Fallback: detected author-related keywords"
            elif any(word in query_lower for word in ['citation', 'impact', 'cited']):
                best_route = 'getCitations'
                confidence = 0.4
                explanation = "Fallback: detected citation-related keywords"
            elif any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
                best_route = 'comparePapers'
                confidence = 0.4
                explanation = "Fallback: detected comparison keywords"
            else:
                best_route = 'searchPapers'
                confidence = 0.3
                explanation = "Fallback: no specific patterns matched, defaulting to paper search"
        else:
            # Get route with highest score
            best_route_data = max(route_scores.items(), key=lambda x: x[1]['score'])
            best_route = best_route_data[0]
            best_score = best_route_data[1]['score']
            
            # Calculate confidence based on score strength and pattern specificity
            max_possible_score = sum(weight for _, weight in self.routes[best_route]['patterns'])
            pattern_confidence = min(0.9, best_score / max_possible_score)
            
            # Adjust confidence based on competing routes
            competing_scores = [data['score'] for route, data in route_scores.items() if route != best_route]
            if competing_scores:
                max_competing = max(competing_scores)
                dominance = best_score / (best_score + max_competing)
                confidence = pattern_confidence * dominance
            else:
                confidence = pattern_confidence
            
            # Ensure minimum confidence for strong matches
            if best_score >= 4:
                confidence = max(confidence, 0.7)
            elif best_score >= 2:
                confidence = max(confidence, 0.5)
            
            # Create explanation
            num_patterns = len(route_scores[best_route]['patterns'])
            explanation = f"Matched {num_patterns} pattern(s) with score {best_score:.1f} for {self.routes[best_route]['description'].lower()}"
            
            if route_scores[best_route]['keyword_boost'] > 0:
                explanation += f" (keyword boost: +{route_scores[best_route]['keyword_boost']})"
        
        # Record this decision for learning
        self.route_history.append({
            'query': query,
            'route': best_route,
            'confidence': confidence,
            'score_details': route_scores,
            'timestamp': datetime.now().isoformat()
        })
        
        return best_route, confidence, explanation
    
    def get_route_scores(self, query: str) -> Dict[str, Any]:
        """Get scoring details for all routes for a query (for debugging/analysis)"""
        query_lower = query.lower().strip()
        route_scores = {}
        
        for route_name, route_info in self.routes.items():
            score = 0
            matched_patterns = []
            
            # Pattern matching with weights
            for pattern, weight in route_info['patterns']:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                if matches:
                    score += weight * len(matches)
                    matched_patterns.append(pattern)
            
            # Keyword boosting
            keyword_boost = 0
            for keyword in self.boost_keywords.get(route_name, []):
                if keyword.lower() in query_lower:
                    keyword_boost += 1
            
            score += keyword_boost * 0.5
            
            route_scores[route_name] = {
                'score': score,
                'patterns_matched': len(matched_patterns),
                'keyword_boost': keyword_boost,
                'total_score': score
            }
        
        return route_scores
    
    def get_available_routes(self) -> List[str]:
        """Get list of available routes"""
        return list(self.routes.keys())
    
    def get_route_description(self, route: str) -> str:
        """Get description for a specific route"""
        return self.routes.get(route, {}).get('description', 'Unknown route')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about route usage"""
        if not self.route_history:
            return {'total_queries': 0, 'route_distribution': {}}
        
        route_counts = {}
        confidence_by_route = {}
        total_confidence = 0
        
        for entry in self.route_history:
            route = entry['route']
            confidence = entry['confidence']
            
            route_counts[route] = route_counts.get(route, 0) + 1
            if route not in confidence_by_route:
                confidence_by_route[route] = []
            confidence_by_route[route].append(confidence)
            total_confidence += confidence
        
        # Calculate average confidence per route
        avg_confidence_by_route = {}
        for route, confidences in confidence_by_route.items():
            avg_confidence_by_route[route] = sum(confidences) / len(confidences)
        
        return {
            'total_queries': len(self.route_history),
            'route_distribution': route_counts,
            'average_confidence': total_confidence / len(self.route_history),
            'confidence_by_route': avg_confidence_by_route,
            'most_confident_route': max(avg_confidence_by_route.items(), key=lambda x: x[1]) if avg_confidence_by_route else None
        }
    
    def test_accuracy(self, test_cases: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Test the accuracy of route selection with detailed metrics
        
        Args:
            test_cases: List of (query, expected_route) tuples
            
        Returns:
            Dictionary with detailed accuracy metrics
        """
        if not test_cases:
            return {'accuracy': 0.0, 'total_cases': 0}
        
        correct = 0
        results = []
        confidence_correct = []
        confidence_incorrect = []
        
        for query, expected_route in test_cases:
            predicted_route, confidence, explanation = self.select_route(query)
            is_correct = predicted_route == expected_route
            
            if is_correct:
                correct += 1
                confidence_correct.append(confidence)
            else:
                confidence_incorrect.append(confidence)
            
            results.append({
                'query': query,
                'expected': expected_route,
                'predicted': predicted_route,
                'confidence': confidence,
                'correct': is_correct,
                'explanation': explanation
            })
        
        accuracy = correct / len(test_cases)
        
        return {
            'accuracy': accuracy,
            'total_cases': len(test_cases),
            'correct_predictions': correct,
            'incorrect_predictions': len(test_cases) - correct,
            'avg_confidence_correct': sum(confidence_correct) / len(confidence_correct) if confidence_correct else 0,
            'avg_confidence_incorrect': sum(confidence_incorrect) / len(confidence_incorrect) if confidence_incorrect else 0,
            'detailed_results': results
        }
    
    def export_results(self, test_results: Dict[str, Any], filename: str = None) -> str:
        """Export test results for ablation studies"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tool_learning_results_{timestamp}.json"
        
        export_data = {
            'system_info': {
                'system_type': 'tool_learning',
                'total_routes': len(self.routes),
                'route_names': list(self.routes.keys())
            },
            'test_results': test_results,
            'route_statistics': self.get_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename

def main():
    """Enhanced test of the tool learning engine"""
    engine = ToolLearningEngine()
    
    # Test queries with expected routes
    test_cases = [
        ("find papers about machine learning", "searchPapers"),
        ("search for research on deep learning", "searchPapers"),
        ("who is Geoffrey Hinton", "getAuthorInfo"),
        ("papers by Yann LeCun", "getAuthorInfo"),
        ("citation count for transformer paper", "getCitations"),
        ("impact factor of Nature journal", "getCitations"),
        ("compare BERT and GPT models", "comparePapers"),
        ("difference between CNN and RNN", "comparePapers"),
        ("trends in AI research", "trendAnalysis"),
        ("evolution of natural language processing", "trendAnalysis"),
        ("Nature journal ranking", "journalAnalysis"),
        ("where to publish ML papers", "journalAnalysis"),
        ("papers related to computer vision", "getRelatedPapers"),
        ("similar work to transformer architecture", "getRelatedPapers")
    ]
    
    print("🤖 Enhanced Tool Learning Engine Test")
    print("=" * 50)
    
    # Test individual queries
    print("\n🧠 Individual Query Testing")
    print("-" * 40)
    
    for query, expected in test_cases[:5]:  # Test first 5
        route, confidence, explanation = engine.select_route(query)
        status = "✅" if route == expected else "❌"
        print(f"\n{status} Query: {query}")
        print(f"   Expected: {expected}")
        print(f"   Predicted: {route} (confidence: {confidence:.3f})")
        print(f"   Explanation: {explanation}")
    
    # Run full accuracy test
    print(f"\n📊 Full Accuracy Test ({len(test_cases)} cases)")
    print("-" * 40)
    
    results = engine.test_accuracy(test_cases)
    
    print(f"Overall Accuracy: {results['accuracy']:.1%} ({results['correct_predictions']}/{results['total_cases']})")
    print(f"Avg Confidence (Correct): {results['avg_confidence_correct']:.3f}")
    print(f"Avg Confidence (Incorrect): {results['avg_confidence_incorrect']:.3f}")
    
    # Show statistics
    stats = engine.get_statistics()
    print(f"\n📈 System Statistics:")
    print(f"Total queries processed: {stats['total_queries']}")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    print(f"Route distribution: {stats['route_distribution']}")
    
    if stats['most_confident_route']:
        route, conf = stats['most_confident_route']
        print(f"Most confident route: {route} ({conf:.3f})")
    
    # Export results for ablation studies
    export_file = engine.export_results(results)
    print(f"\n💾 Results exported to: {export_file}")
    print("Ready for ablation testing with other models!")

if __name__ == "__main__":
    main() 