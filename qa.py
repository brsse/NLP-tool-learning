"""
Simple QA Dataset for Tool Learning System
Realistic expected answers based on actual arxiv dataset content.
"""

from typing import Dict, List, Any

# ============================================================================
# SIMPLE QA DATASET 
# ============================================================================

COMPREHENSIVE_QA_DATASET = [
    # searchPapers Route
    {
        'id': 1, 
        'query': 'find papers about machine learning optimization', 
        'expected_route': 'searchPapers', 
        'difficulty': 'easy', 
        'expected_answer': 'Found optimization papers including "Lecture Notes: Optimization for Machine Learning" by Elad Hazan (2019) and "A Survey of Optimization Methods from a Machine Learning Perspective" by Shiliang Sun et al.',
        'expected_papers_min': 3,
        'expected_keywords': ['optimization', 'machine learning', 'Elad Hazan', 'gradient']
    },
    {
        'id': 2, 
        'query': 'search for adversarial machine learning papers', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Found adversarial ML papers including "Adversarial Attacks and Defenses in Machine Learning-Powered Networks: A Contemporary Survey" by Yulong Wang et al. (2023) which covers modern adversarial attacks and defense techniques in deep learning networks.',
        'expected_papers_min': 2,
        'expected_keywords': ['adversarial', 'machine learning', 'attacks', 'defense']
    },
    {
        'id': 3, 
        'query': 'healthcare machine learning applications', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Found healthcare ML papers including "Machine Learning for Clinical Predictive Analytics" by Wei-Hung Weng (2019) and "Probabilistic Machine Learning for Healthcare" by Chen et al.',
        'expected_papers_min': 2,
        'expected_keywords': ['healthcare', 'clinical', 'Wei-Hung Weng', 'medical']
    },
    {
        'id': 4, 
        'query': 'automated machine learning AutoML', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Found AutoML papers including "Techniques for Automated Machine Learning" by Chen et al. and "AutoCompete: A Framework for Machine Learning Competition" by Thakur et al.',
        'expected_papers_min': 2,
        'expected_keywords': ['automated', 'AutoML', 'machine learning', 'techniques']
    },
    {
        'id': 5, 
        'query': 'quantum machine learning methods', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'expected_answer': 'Found quantum ML papers including "A comprehensive review of Quantum Machine Learning: from NISQ to Fault Tolerance" by Wang et al. and "Challenges and Opportunities in Quantum Machine Learning" by Cerezo et al.',
        'expected_papers_min': 2,
        'expected_keywords': ['quantum', 'machine learning', 'NISQ', 'quantum computing']
    },

    # getAuthorInfo Route  
    {
        'id': 6, 
        'query': 'who is Elad Hazan', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'expected_answer': 'Elad Hazan is a professor of Computer Science at Princeton University and author of "Lecture Notes: Optimization for Machine Learning" (2019). His research focuses on machine learning and mathematical optimization.',
        'expected_papers_min': 1,
        'expected_keywords': ['Elad Hazan', 'optimization', 'Princeton', 'machine learning']
    },
    {
        'id': 7, 
        'query': 'research by Wei-Hung Weng', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'expected_answer': 'Wei-Hung Weng authored "Machine Learning for Clinical Predictive Analytics" and "Representation Learning for Electronic Health Records". His research focuses on clinical machine learning applications.',
        'expected_papers_min': 1,
        'expected_keywords': ['Wei-Hung Weng', 'clinical', 'healthcare', 'machine learning']
    },
    {
        'id': 8, 
        'query': 'papers by Xiaojin Zhu', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'expected_answer': 'Xiaojin Zhu authored "An Optimal Control View of Adversarial Machine Learning" (2018), exploring adversarial ML through optimal control and reinforcement learning perspectives.',
        'expected_papers_min': 1,
        'expected_keywords': ['Xiaojin Zhu', 'adversarial', 'optimal control', 'reinforcement learning']
    },

    # getCitations Route
    {
        'id': 9, 
        'query': 'citation analysis for machine learning papers', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'expected_answer': 'Citation analysis shows high impact papers include optimization surveys, healthcare applications, and quantum ML reviews with significant academic influence.',
        'expected_papers_min': 1,
        'expected_keywords': ['citation', 'analysis', 'impact', 'machine learning']
    },
    {
        'id': 10, 
        'query': 'impact of automated machine learning research', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'expected_answer': 'AutoML research has significant impact with frameworks like AutoCompete and comprehensive surveys driving adoption in industry and academia.',
        'expected_papers_min': 1,
        'expected_keywords': ['impact', 'automated machine learning', 'AutoML', 'research']
    },

    # getRelatedPapers Route
    {
        'id': 11, 
        'query': 'papers related to machine learning optimization', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Related optimization papers include surveys on optimization methods, Bayesian optimization guides, and gradient-based learning techniques for ML.',
        'expected_papers_min': 3,
        'expected_keywords': ['related', 'optimization', 'machine learning', 'gradient']
    },
    {
        'id': 12, 
        'query': 'similar research to healthcare ML', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Related healthcare research includes clinical predictive analytics, EHR representation learning, and probabilistic ML for medical applications.',
        'expected_papers_min': 2,
        'expected_keywords': ['similar', 'healthcare', 'clinical', 'medical ML']
    },

    # comparePapers Route
    {
        'id': 13, 
        'query': 'compare supervised and unsupervised learning', 
        'expected_route': 'comparePapers', 
        'difficulty': 'easy', 
        'expected_answer': 'Supervised learning uses labeled data for prediction while unsupervised learning finds patterns in unlabeled data. Different applications and evaluation methods.',
        'expected_papers_min': 2,
        'expected_keywords': ['labeled', 'supervised', 'unsupervised', 'learning']
    },
    {
        'id': 14, 
        'query': 'classical ML versus deep learning approaches', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Classical ML uses handcrafted features while deep learning learns representations automatically. Deep learning excels with large data, classical ML better for small datasets.',
        'expected_papers_min': 2,
        'expected_keywords': ['classical ML', 'deep learning', 'comparison', 'features']
    },

    # trendAnalysis Route
    {
        'id': 15, 
        'query': 'trends in machine learning research', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'expected_answer': 'Current trends include quantum ML, automated ML, healthcare applications, and interpretable AI. Growing focus on safety and ethical considerations.',
        'expected_papers_min': 3,
        'expected_keywords': ['trends', 'machine learning', 'quantum', 'automated']
    },
    {
        'id': 16, 
        'query': 'evolution of optimization algorithms', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'expected_answer': 'Optimization evolved from simple gradient descent to advanced methods like Adam, Bayesian optimization, and quantum-inspired algorithms for ML.',
        'expected_papers_min': 2,
        'expected_keywords': ['evolution', 'optimization', 'algorithms', 'gradient descent']
    },

    # journalAnalysis Route
    {
        'id': 17, 
        'query': 'best venues for machine learning research', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'medium', 
        'expected_answer': 'Top ML venues include journals like JMLR and TPAMI, and conferences like ICML, ICLR, NeurIPS, and other specialized conferences for domains like healthcare AI and quantum computing.',
        'expected_papers_min': 1,
        'expected_keywords': ['journals', 'JMLR', 'ICML', 'conferences']
    },
    {
        'id': 18, 
        'query': 'publication patterns in AI research', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'medium', 
        'expected_answer': 'AI research shows increasing publication volume with arXiv preprints becoming standard. Growing specialization in subfields like quantum ML and healthcare.',
        'expected_papers_min': 1,
        'expected_keywords': ['publication', 'patterns', 'AI research', 'arXiv']
    },

    # Multi-Route Cases
    {
        'id': 19, 
        'query': 'find papers by Elad Hazan on optimization', 
        'expected_route': 'searchPapers, getAuthorInfo', 
        'difficulty': 'medium', 
        'expected_answer': 'Elad Hazan authored "Lecture Notes: Optimization for Machine Learning" (2019). The paper covers optimization fundamentals for ML derived from Princeton University courses and tutorials.',
        'expected_papers_min': 1,
        'expected_keywords': ['Elad Hazan', 'optimization', 'Princeton', 'lecture notes']
    },
    {
        'id': 20, 
        'query': 'compare quantum ML approaches and analyze trends', 
        'expected_route': 'comparePapers, trendAnalysis', 
        'difficulty': 'hard', 
        'expected_answer': 'Quantum ML includes NISQ and fault-tolerant approaches. Trends show growing interest in quantum advantages for specific ML tasks, with challenges in near-term implementations.',
        'expected_papers_min': 2,
        'expected_keywords': ['quantum ML', 'NISQ', 'fault-tolerant', 'trends']
    }
]

# ============================================================================
# SIMPLE UTILITY FUNCTIONS
# ============================================================================

def get_dataset_statistics() -> Dict[str, Any]:
    """Get basic statistics about the QA dataset"""
    total_cases = len(COMPREHENSIVE_QA_DATASET)
    
    route_counts = {}
    difficulty_counts = {}
    
    for case in COMPREHENSIVE_QA_DATASET:
        # Count routes (handle multi-route)
        routes = case['expected_route'].split(', ')
        for route in routes:
            route_counts[route.strip()] = route_counts.get(route.strip(), 0) + 1
        
        # Count difficulties
        difficulty_counts[case['difficulty']] = difficulty_counts.get(case['difficulty'], 0) + 1
    
    return {
        'total_cases': total_cases,
        'route_distribution': route_counts,
        'difficulty_distribution': difficulty_counts,
        'multi_route_cases': len([c for c in COMPREHENSIVE_QA_DATASET if ',' in c['expected_route']])
    }

def calculate_expected_answer_similarity(actual_response: str, expected_answer: str) -> float:
    """Simple similarity between actual and expected answers"""
    if not actual_response or not expected_answer:
        return 0.0
    
    actual_words = set(actual_response.lower().split())
    expected_words = set(expected_answer.lower().split())
    
    if not expected_words:
        return 0.0
    
    intersection = len(actual_words.intersection(expected_words))
    union = len(actual_words.union(expected_words))
    
    return intersection / union if union > 0 else 0.0

def calculate_keyword_match_score(response: str, expected_keywords: List[str]) -> float:
    """Calculate how well response matches expected keywords"""
    if not expected_keywords:
        return 1.0
    
    response_lower = response.lower()
    matched_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
    
    return matched_keywords / len(expected_keywords)

def main():
    """Print simple dataset statistics"""
    print("📊 Simple QA Dataset for Tool Learning")
    print("=" * 40)
    
    stats = get_dataset_statistics()
    print(f"Total test cases: {stats['total_cases']}")
    print(f"Multi-route cases: {stats['multi_route_cases']}")
    print()
    
    print("Routes covered:")
    for route, count in sorted(stats['route_distribution'].items()):
        print(f"  {route}: {count}")
    print()
    
    print("Difficulty levels:")
    for difficulty, count in sorted(stats['difficulty_distribution'].items()):
        print(f"  {difficulty}: {count}")
    print()
    
    print("✅ Simple, realistic dataset ready for testing!")

if __name__ == "__main__":
    main() 