#!/usr/bin/env python3
"""
Comprehensive QA Testing for Tool Learning Chatbot

Single unified testing file covering all routes, edge cases, and hidden test scenarios.
Tests tool learning accuracy, route selection, and response quality.
"""

import json
from typing import Dict, List, Any, Tuple
from datetime import datetime
from tool_learning_engine import ToolLearningEngine

class ComprehensiveQATests:
    """Unified Q&A testing suite for tool learning evaluation"""
    
    def __init__(self):
        self.engine = ToolLearningEngine()
        self.test_results = []
        
        # Comprehensive test cases covering all routes and edge cases
        self.test_cases = [
            # ===== searchPapers Route (23 cases) =====
            {
                "id": "search_001",
                "query": "find papers about machine learning",
                "expected_route": "searchPapers",
                "difficulty": "easy",
                "category": "basic_search",
                "sources": ["arXiv"],
                "expected_answer": "I found several influential papers in machine learning..."
            },
            {
                "id": "search_002", 
                "query": "search for research on deep learning",
                "expected_route": "searchPapers",
                "difficulty": "easy",
                "category": "basic_search",
                "sources": ["arXiv"],
                "expected_answer": "Deep learning research includes foundational works..."
            },
            {
                "id": "search_003",
                "query": "get papers on natural language processing",
                "expected_route": "searchPapers", 
                "difficulty": "easy",
                "category": "basic_search",
                "sources": ["arXiv"],
                "expected_answer": "Key NLP papers include transformer architectures..."
            },
            {
                "id": "search_004",
                "query": "look for computer vision articles",
                "expected_route": "searchPapers",
                "difficulty": "easy", 
                "category": "basic_search",
                "sources": ["arXiv"],
                "expected_answer": "Important computer vision papers include CNNs and object detection..."
            },
            {
                "id": "search_005",
                "query": "recent papers on transformers",
                "expected_route": "searchPapers",
                "difficulty": "medium",
                "category": "temporal_search",
                "sources": ["arXiv"],
                "expected_answer": "Recent transformer research builds on attention mechanisms..."
            },
            {
                "id": "search_006",
                "query": "literature review on neural networks", 
                "expected_route": "searchPapers",
                "difficulty": "medium",
                "category": "literature_review",
                "sources": ["arXiv"],
                "expected_answer": "Neural network literature spans multiple decades..."
            },
            {
                "id": "search_007",
                "query": "papers about attention mechanisms",
                "expected_route": "searchPapers",
                "difficulty": "medium",
                "category": "technical_search", 
                "sources": ["arXiv"],
                "expected_answer": "Attention mechanisms revolutionized sequence modeling..."
            },
            {
                "id": "search_008",
                "query": "find research on object detection",
                "expected_route": "searchPapers",
                "difficulty": "medium",
                "category": "technical_search",
                "sources": ["arXiv"],
                "expected_answer": "Object detection evolved from R-CNN to YOLO architectures..."
            },
            {
                "id": "search_009",
                "query": "search for pre-training papers",
                "expected_route": "searchPapers",
                "difficulty": "medium",
                "category": "technical_search",
                "sources": ["arXiv"],
                "expected_answer": "Pre-training methods like BERT and GPT changed NLP..."
            },
            {
                "id": "search_010",
                "query": "medical AI research papers",
                "expected_route": "searchPapers",
                "difficulty": "medium",
                "category": "domain_search",
                "sources": ["PubMed"],
                "expected_answer": "Medical AI research focuses on diagnostic and treatment applications..."
            },
            {
                "id": "search_011",
                "query": "biological machine learning papers",
                "expected_route": "searchPapers",
                "difficulty": "medium",
                "category": "domain_search",
                "sources": ["bioRxiv"],
                "expected_answer": "Biological ML applications include protein folding and genomics..."
            },
            {
                "id": "search_012",
                "query": "chemistry AI publications", 
                "expected_route": "searchPapers",
                "difficulty": "medium",
                "category": "domain_search",
                "sources": ["chemRxiv"],
                "expected_answer": "Chemistry AI focuses on drug discovery and material design..."
            },
            {
                "id": "search_013",
                "query": "COVID-19 detection papers",
                "expected_route": "searchPapers",
                "difficulty": "medium", 
                "category": "temporal_search",
                "sources": ["medRxiv"],
                "expected_answer": "COVID-19 detection research exploded during the pandemic..."
            },
            {
                "id": "search_014",
                "query": "protein folding prediction research",
                "expected_route": "searchPapers",
                "difficulty": "hard",
                "category": "specialized_search",
                "sources": ["bioRxiv"],
                "expected_answer": "Protein folding prediction was revolutionized by AlphaFold..."
            },
            {
                "id": "search_015",
                "query": "catalyst design machine learning",
                "expected_route": "searchPapers",
                "difficulty": "hard",
                "category": "specialized_search", 
                "sources": ["chemRxiv"],
                "expected_answer": "ML for catalyst design optimizes chemical reactions..."
            },
            {
                "id": "search_016",
                "query": "single-cell RNA sequencing analysis",
                "expected_route": "searchPapers",
                "difficulty": "hard",
                "category": "specialized_search",
                "sources": ["bioRxiv"],
                "expected_answer": "scRNA-seq analysis reveals cellular heterogeneity..."
            },
            {
                "id": "search_017",
                "query": "precision medicine AI applications",
                "expected_route": "searchPapers",
                "difficulty": "hard",
                "category": "specialized_search",
                "sources": ["medRxiv"],
                "expected_answer": "Precision medicine uses AI for personalized treatment..."
            },
            {
                "id": "search_018",
                "query": "quantum chemistry machine learning",
                "expected_route": "searchPapers",
                "difficulty": "hard",
                "category": "specialized_search",
                "sources": ["chemRxiv"],
                "expected_answer": "Quantum chemistry ML predicts molecular properties..."
            },
            {
                "id": "search_019",
                "query": "drug discovery AI pipelines", 
                "expected_route": "searchPapers",
                "difficulty": "hard",
                "category": "specialized_search",
                "sources": ["chemRxiv"],
                "expected_answer": "AI drug discovery accelerates pharmaceutical development..."
            },
            {
                "id": "search_020",
                "query": "automated cell analysis computer vision",
                "expected_route": "searchPapers",
                "difficulty": "hard",
                "category": "specialized_search",
                "sources": ["bioRxiv"],
                "expected_answer": "Automated cell analysis uses computer vision for biological research..."
            },
            {
                "id": "search_021",
                "query": "machine learning",
                "expected_route": "searchPapers",
                "difficulty": "easy",
                "category": "ambiguous",
                "sources": ["arXiv"],
                "expected_answer": "Machine learning encompasses various algorithms and techniques..."
            },
            {
                "id": "search_022",
                "query": "BERT",
                "expected_route": "searchPapers", 
                "difficulty": "medium",
                "category": "ambiguous",
                "sources": ["arXiv"],
                "expected_answer": "BERT is a bidirectional transformer for language understanding..."
            },
            {
                "id": "search_023",
                "query": "papers published in Nature",
                "expected_route": "searchPapers",
                "difficulty": "medium",
                "category": "venue_search",
                "sources": ["PubMed"],
                "expected_answer": "Nature publishes high-impact research across sciences..."
            },

            # ===== getAuthorInfo Route (15 cases) =====
            {
                "id": "author_001",
                "query": "papers by Yoshua Bengio",
                "expected_route": "getAuthorInfo",
                "difficulty": "easy",
                "category": "famous_author",
                "sources": ["arXiv"],
                "expected_answer": "Yoshua Bengio is a Turing Award winner and deep learning pioneer..."
            },
            {
                "id": "author_002",
                "query": "who is Geoffrey Hinton",
                "expected_route": "getAuthorInfo",
                "difficulty": "easy", 
                "category": "famous_author",
                "sources": ["arXiv"],
                "expected_answer": "Geoffrey Hinton is the 'godfather of deep learning'..."
            },
            {
                "id": "author_003",
                "query": "research by Ian Goodfellow",
                "expected_route": "getAuthorInfo", 
                "difficulty": "easy",
                "category": "famous_author",
                "sources": ["arXiv"],
                "expected_answer": "Ian Goodfellow invented GANs and contributed to deep learning..."
            },
            {
                "id": "author_004",
                "query": "author profile for Ashish Vaswani",
                "expected_route": "getAuthorInfo",
                "difficulty": "medium",
                "category": "specific_author",
                "sources": ["arXiv"],
                "expected_answer": "Ashish Vaswani led the team that created the Transformer..."
            },
            {
                "id": "author_005",
                "query": "publications of Yann LeCun",
                "expected_route": "getAuthorInfo",
                "difficulty": "easy",
                "category": "famous_author", 
                "sources": ["arXiv"],
                "expected_answer": "Yann LeCun pioneered CNNs and is a Turing Award winner..."
            },
            {
                "id": "author_006",
                "query": "tell me about Jacob Devlin",
                "expected_route": "getAuthorInfo",
                "difficulty": "medium",
                "category": "specific_author",
                "sources": ["arXiv"],
                "expected_answer": "Jacob Devlin is the lead author of BERT..."
            },
            {
                "id": "author_007",
                "query": "researcher Andrew Ng background",
                "expected_route": "getAuthorInfo",
                "difficulty": "medium",
                "category": "famous_author",
                "sources": ["arXiv"],
                "expected_answer": "Andrew Ng is a leading AI educator and researcher..."
            },
            {
                "id": "author_008",
                "query": "works by Kaiming He",
                "expected_route": "getAuthorInfo",
                "difficulty": "medium",
                "category": "specific_author",
                "sources": ["arXiv"], 
                "expected_answer": "Kaiming He created ResNet and other important CV architectures..."
            },
            {
                "id": "author_009",
                "query": "author information for Tom Brown",
                "expected_route": "getAuthorInfo",
                "difficulty": "medium",
                "category": "specific_author",
                "sources": ["arXiv"],
                "expected_answer": "Tom Brown led the GPT-3 development at OpenAI..."
            },
            {
                "id": "author_010",
                "query": "scientist Joseph Redmon profile",
                "expected_route": "getAuthorInfo",
                "difficulty": "medium",
                "category": "specific_author",
                "sources": ["arXiv"],
                "expected_answer": "Joseph Redmon created the YOLO object detection system..."
            },
            {
                "id": "author_011",
                "query": "Dr. Jennifer Liu publications",
                "expected_route": "getAuthorInfo",
                "difficulty": "hard",
                "category": "medical_author",
                "sources": ["medRxiv"],
                "expected_answer": "Dr. Jennifer Liu published COVID-19 diagnostic imaging research..."
            },
            {
                "id": "author_012",
                "query": "Prof. Elena Rodriguez research",
                "expected_route": "getAuthorInfo",
                "difficulty": "hard",
                "category": "bio_author",
                "sources": ["bioRxiv"],
                "expected_answer": "Dr. Elena Rodriguez works on neural networks for protein folding..."
            },
            {
                "id": "author_013",
                "query": "Dr. Rebecca Taylor chemistry papers",
                "expected_route": "getAuthorInfo",
                "difficulty": "hard",
                "category": "chem_author", 
                "sources": ["chemRxiv"],
                "expected_answer": "Dr. Rebecca Taylor researches ML for catalyst design..."
            },
            {
                "id": "author_014",
                "query": "researcher Maria Gonzalez NLP",
                "expected_route": "getAuthorInfo",
                "difficulty": "hard",
                "category": "specific_author",
                "sources": ["PubMed"],
                "expected_answer": "Dr. Maria Gonzalez works on clinical NLP for mental health..."
            },
            {
                "id": "author_015",
                "query": "Prof. Anna Kowalski single-cell research",
                "expected_route": "getAuthorInfo",
                "difficulty": "hard",
                "category": "bio_author",
                "sources": ["bioRxiv"],
                "expected_answer": "Dr. Anna Kowalski develops deep learning for scRNA-seq analysis..."
            },

            # ===== getCitations Route (10 cases) =====
            {
                "id": "cite_001",
                "query": "citation count for BERT paper",
                "expected_route": "getCitations",
                "difficulty": "easy",
                "category": "basic_citation",
                "sources": ["arXiv"],
                "expected_answer": "The BERT paper has accumulated over 45,000 citations..."
            },
            {
                "id": "cite_002",
                "query": "how many citations does ResNet have",
                "expected_route": "getCitations",
                "difficulty": "easy",
                "category": "basic_citation",
                "sources": ["arXiv"],
                "expected_answer": "ResNet has over 89,000 citations..."
            },
            {
                "id": "cite_003",
                "query": "impact factor of transformer paper",
                "expected_route": "getCitations",
                "difficulty": "medium",
                "category": "impact_analysis",
                "sources": ["arXiv"],
                "expected_answer": "The Transformer paper has 70,245 citations and tremendous impact..."
            },
            {
                "id": "cite_004",
                "query": "cited count for GPT-3 publication",
                "expected_route": "getCitations",
                "difficulty": "medium", 
                "category": "recent_citation",
                "sources": ["arXiv"],
                "expected_answer": "GPT-3 has over 25,000 citations despite being recent..."
            },
            {
                "id": "cite_005",
                "query": "h-index analysis for deep learning papers",
                "expected_route": "getCitations",
                "difficulty": "hard",
                "category": "metric_analysis",
                "sources": ["arXiv"],
                "expected_answer": "Deep learning papers show high h-index values..."
            },
            {
                "id": "cite_006",
                "query": "bibliometric study of YOLO paper",
                "expected_route": "getCitations",
                "difficulty": "medium",
                "category": "bibliometric",
                "sources": ["arXiv"],
                "expected_answer": "YOLO has influenced object detection research significantly..."
            },
            {
                "id": "cite_007",
                "query": "citation analysis for RoBERTa",
                "expected_route": "getCitations",
                "difficulty": "medium",
                "category": "citation_analysis",
                "sources": ["arXiv"],
                "expected_answer": "RoBERTa improved BERT and gained substantial citations..."
            },
            {
                "id": "cite_008",
                "query": "COVID-19 paper citation impact",
                "expected_route": "getCitations",
                "difficulty": "medium",
                "category": "domain_citation",
                "sources": ["medRxiv"],
                "expected_answer": "COVID-19 diagnostic papers show rapid citation accumulation..."
            },
            {
                "id": "cite_009",
                "query": "protein folding paper citations",
                "expected_route": "getCitations",
                "difficulty": "medium",
                "category": "domain_citation",
                "sources": ["bioRxiv"],
                "expected_answer": "Protein folding AI papers gain citations in biological research..."
            },
            {
                "id": "cite_010",
                "query": "catalyst design paper impact",
                "expected_route": "getCitations",
                "difficulty": "medium",
                "category": "domain_citation",
                "sources": ["chemRxiv"],
                "expected_answer": "Catalyst design ML papers impact chemical engineering..."
            },

            # ===== comparePapers Route (8 cases) =====
            {
                "id": "comp_001",
                "query": "compare BERT and GPT models",
                "expected_route": "comparePapers",
                "difficulty": "medium",
                "category": "model_comparison",
                "sources": ["arXiv"],
                "expected_answer": "BERT uses bidirectional encoding while GPT uses autoregressive generation..."
            },
            {
                "id": "comp_002",
                "query": "difference between ResNet and YOLO",
                "expected_route": "comparePapers",
                "difficulty": "medium",
                "category": "architecture_comparison",
                "sources": ["arXiv"],
                "expected_answer": "ResNet focuses on classification while YOLO does object detection..."
            },
            {
                "id": "comp_003",
                "query": "BERT versus RoBERTa comparison",
                "expected_route": "comparePapers",
                "difficulty": "medium",
                "category": "model_comparison",
                "sources": ["arXiv"],
                "expected_answer": "RoBERTa improved BERT's training procedure and performance..."
            },
            {
                "id": "comp_004",
                "query": "transformer vs CNN architectures",
                "expected_route": "comparePapers",
                "difficulty": "hard",
                "category": "architecture_comparison",
                "sources": ["arXiv"],
                "expected_answer": "Transformers use attention while CNNs use convolutions..."
            },
            {
                "id": "comp_005",
                "query": "COVID-19 detection methods comparison",
                "expected_route": "comparePapers",
                "difficulty": "hard",
                "category": "method_comparison",
                "sources": ["medRxiv"],
                "expected_answer": "COVID-19 detection methods vary in accuracy and speed..."
            },
            {
                "id": "comp_006",
                "query": "protein folding vs drug discovery AI",
                "expected_route": "comparePapers",
                "difficulty": "hard",
                "category": "application_comparison",
                "sources": ["bioRxiv"],
                "expected_answer": "Protein folding and drug discovery use different AI approaches..."
            },
            {
                "id": "comp_007",
                "query": "catalyst design vs synthesis prediction",
                "expected_route": "comparePapers",
                "difficulty": "hard",
                "category": "method_comparison",
                "sources": ["chemRxiv"],
                "expected_answer": "Catalyst design and synthesis prediction serve different purposes..."
            },
            {
                "id": "comp_008",
                "query": "what's better GPT-3 or BERT",
                "expected_route": "comparePapers",
                "difficulty": "medium",
                "category": "model_comparison",
                "sources": ["arXiv"],
                "expected_answer": "GPT-3 and BERT serve different purposes in NLP..."
            },

            # ===== trendAnalysis Route (6 cases) =====
            {
                "id": "trend_001",
                "query": "trends in computer vision",
                "expected_route": "trendAnalysis",
                "difficulty": "medium",
                "category": "field_trends",
                "sources": ["arXiv"],
                "expected_answer": "Computer vision evolved from CNNs to transformers and attention..."
            },
            {
                "id": "trend_002",
                "query": "evolution of language models",
                "expected_route": "trendAnalysis",
                "difficulty": "medium",
                "category": "model_evolution",
                "sources": ["arXiv"],
                "expected_answer": "Language models progressed from RNNs to transformers to LLMs..."
            },
            {
                "id": "trend_003",
                "query": "development in medical AI over time",
                "expected_route": "trendAnalysis",
                "difficulty": "hard",
                "category": "domain_trends",
                "sources": ["medRxiv"],
                "expected_answer": "Medical AI evolved from basic pattern recognition to deep learning..."
            },
            {
                "id": "trend_004",
                "query": "historical progress in biological AI",
                "expected_route": "trendAnalysis",
                "difficulty": "hard",
                "category": "domain_trends",
                "sources": ["bioRxiv"],
                "expected_answer": "Biological AI advanced from simple models to protein folding..."
            },
            {
                "id": "trend_005",
                "query": "chemistry AI evolution timeline",
                "expected_route": "trendAnalysis",
                "difficulty": "hard",
                "category": "domain_trends",
                "sources": ["chemRxiv"],
                "expected_answer": "Chemistry AI progressed from molecular descriptors to deep learning..."
            },
            {
                "id": "trend_006",
                "query": "development in deep learning over time",
                "expected_route": "trendAnalysis",
                "difficulty": "medium",
                "category": "field_trends",
                "sources": ["arXiv"],
                "expected_answer": "Deep learning evolved from MLPs to CNNs to transformers..."
            },

            # ===== journalAnalysis Route (3 cases) =====
            {
                "id": "journal_001",
                "query": "Nature journal impact factor",
                "expected_route": "journalAnalysis",
                "difficulty": "easy",
                "category": "journal_metrics",
                "sources": ["PubMed"],
                "expected_answer": "Nature has one of the highest impact factors in science..."
            },
            {
                "id": "journal_002",
                "query": "best venues for machine learning papers",
                "expected_route": "journalAnalysis",
                "difficulty": "medium",
                "category": "venue_recommendation",
                "sources": ["arXiv"],
                "expected_answer": "Top ML venues include NeurIPS, ICML, and ICLR..."
            },
            {
                "id": "journal_003",
                "query": "medical journal ranking for AI research",
                "expected_route": "journalAnalysis",
                "difficulty": "hard",
                "category": "domain_venues",
                "sources": ["PubMed"],
                "expected_answer": "Medical AI research appears in Nature Medicine, NEJM AI..."
            },

            # ===== getRelatedPapers Route (5 cases) =====
            {
                "id": "related_001",
                "query": "papers related to transformers",
                "expected_route": "getRelatedPapers",
                "difficulty": "medium",
                "category": "related_search",
                "sources": ["arXiv"],
                "expected_answer": "Papers related to transformers include attention mechanisms..."
            },
            {
                "id": "related_002",
                "query": "similar research to protein folding",
                "expected_route": "getRelatedPapers",
                "difficulty": "hard",
                "category": "related_search",
                "sources": ["bioRxiv"],
                "expected_answer": "Related protein research includes structural prediction..."
            },
            {
                "id": "related_003",
                "query": "connected work to COVID-19 detection",
                "expected_route": "getRelatedPapers",
                "difficulty": "hard",
                "category": "related_search",
                "sources": ["medRxiv"],
                "expected_answer": "Related work includes medical imaging and diagnostic AI..."
            },
            {
                "id": "related_004",
                "query": "similar papers to catalyst design",
                "expected_route": "getRelatedPapers",
                "difficulty": "hard",
                "category": "related_search",
                "sources": ["chemRxiv"],
                "expected_answer": "Related chemistry research includes reaction prediction..."
            },
            {
                "id": "related_005",
                "query": "papers building on BERT architecture",
                "expected_route": "getRelatedPapers",
                "difficulty": "medium",
                "category": "related_search",
                "sources": ["arXiv"],
                "expected_answer": "BERT-based models include RoBERTa, DeBERTa, and others..."
            },

            # ===== Multi-Route Test Cases (4 cases) =====
            {
                "id": "multi_001",
                "query": "find related transformer papers, analyze their citations, and compare approaches",
                "expected_route": "getRelatedPapers",
                "difficulty": "hard",
                "category": "multi_step",
                "sources": ["arXiv"],
                "multi_route": ["getRelatedPapers", "getCitations", "comparePapers"],
                "expected_answer": "Related transformer papers show varying citation patterns and different architectural approaches..."
            },
            {
                "id": "multi_002",
                "query": "similar protein folding papers and their impact compared to AlphaFold",
                "expected_route": "getRelatedPapers",
                "difficulty": "hard",
                "category": "multi_step",
                "sources": ["bioRxiv"],
                "multi_route": ["getRelatedPapers", "getCitations", "comparePapers"],
                "expected_answer": "Protein folding research shows AlphaFold's dominant impact over related approaches..."
            },
            {
                "id": "multi_003",
                "query": "find BERT papers, get author details, and analyze citation trends",
                "expected_route": "searchPapers",
                "difficulty": "hard",
                "category": "multi_step",
                "sources": ["arXiv"],
                "multi_route": ["searchPapers", "getAuthorInfo", "getCitations"],
                "expected_answer": "BERT research involves key authors like Jacob Devlin with strong citation trends..."
            },
            {
                "id": "multi_004",
                "query": "analyze AI trends, find recent papers, and compare with earlier work",
                "expected_route": "trendAnalysis",
                "difficulty": "hard",
                "category": "multi_step",
                "sources": ["arXiv"],
                "multi_route": ["trendAnalysis", "searchPapers", "comparePapers"],
                "expected_answer": "AI trends show evolution from basic models to sophisticated architectures..."
            }
        ]
        
    def get_test_cases(self) -> List[Dict[str, Any]]:
        """Get all test cases"""
        return self.test_cases
    
    def get_multi_route_cases(self) -> List[Dict[str, Any]]:
        """Get multi-route test cases"""
        return [case for case in self.test_cases if 'multi_route' in case]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get test suite statistics"""
        route_distribution = {}
        difficulty_distribution = {}
        category_distribution = {}
        source_distribution = {}
        
        for case in self.test_cases:
            route = case['expected_route']
            difficulty = case['difficulty']
            category = case['category']
            sources = case['sources']
            
            route_distribution[route] = route_distribution.get(route, 0) + 1
            difficulty_distribution[difficulty] = difficulty_distribution.get(difficulty, 0) + 1
            category_distribution[category] = category_distribution.get(category, 0) + 1
            
            for source in sources:
                source_distribution[source] = source_distribution.get(source, 0) + 1
        
        return {
            'total_cases': len(self.test_cases),
            'route_distribution': route_distribution,
            'difficulty_distribution': difficulty_distribution,
            'category_distribution': category_distribution,
            'source_distribution': source_distribution,
            'multi_route_cases': len(self.get_multi_route_cases())
        }
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive test cases and evaluate performance"""
        print("🧪 Running Comprehensive QA Tests")
        print("=" * 50)
        
        test_cases = self.get_test_cases()
        total_tests = len(test_cases)
        correct_routes = 0
        route_accuracy_by_type = {}
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i:2d}/{total_tests}] Testing: {test_case['query'][:50]}...")
            
            # Test route selection
            predicted_route, confidence, explanation = self.engine.select_route(test_case['query'])
            expected_route = test_case['expected_route']
            
            is_correct = predicted_route == expected_route
            if is_correct:
                correct_routes += 1
            
            # Track accuracy by route type
            if expected_route not in route_accuracy_by_type:
                route_accuracy_by_type[expected_route] = {'correct': 0, 'total': 0}
            route_accuracy_by_type[expected_route]['total'] += 1
            if is_correct:
                route_accuracy_by_type[expected_route]['correct'] += 1
            
            # Store result
            result = {
                'test_id': test_case['id'],
                'query': test_case['query'],
                'expected_route': expected_route,
                'predicted_route': predicted_route,
                'confidence': confidence,
                'correct': is_correct,
                'explanation': explanation,
                'expected_answer': test_case['expected_answer'],
                'difficulty': test_case['difficulty'],
                'category': test_case['category'],
                'sources': test_case['sources'],
                'multi_route': test_case.get('multi_route', None)
            }
            self.test_results.append(result)
            
            # Print result
            status = "✅" if is_correct else "❌"
            difficulty_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(test_case['difficulty'], "⚪")
            print(f"    {status} {difficulty_icon} Expected: {expected_route}, Got: {predicted_route} (conf: {confidence:.3f})")
            if not is_correct:
                print(f"    ⚠️  Explanation: {explanation}")
            if test_case.get('multi_route'):
                print(f"    🔗 Multi-route: {' -> '.join(test_case['multi_route'])}")
        
        # Calculate overall metrics
        overall_accuracy = correct_routes / total_tests
        
        print(f"\n📊 Overall Results:")
        print(f"Total Tests: {total_tests}")
        print(f"Correct Routes: {correct_routes}")
        print(f"Overall Accuracy: {overall_accuracy:.1%}")
        
        print(f"\n📈 Accuracy by Route Type:")
        for route, stats in route_accuracy_by_type.items():
            route_acc = stats['correct'] / stats['total']
            print(f"  {route:15s}: {stats['correct']:2d}/{stats['total']:2d} ({route_acc:.1%})")
        
        # Show multi-route test results
        multi_route_cases = [r for r in self.test_results if r['multi_route']]
        if multi_route_cases:
            print(f"\n🔗 Multi-Route Test Results:")
            multi_correct = len([r for r in multi_route_cases if r['correct']])
            multi_accuracy = multi_correct / len(multi_route_cases)
            print(f"  Multi-Route Accuracy: {multi_correct}/{len(multi_route_cases)} ({multi_accuracy:.1%})")
            
            for result in multi_route_cases:
                status = "✅" if result['correct'] else "❌"
                print(f"  {status} {result['query'][:40]}...")
                print(f"     Expected: {result['expected_route']}, Got: {result['predicted_route']}")
        
        return {
            'total_tests': total_tests,
            'correct_routes': correct_routes,
            'overall_accuracy': overall_accuracy,
            'route_accuracy': route_accuracy_by_type,
            'multi_route_accuracy': multi_accuracy if multi_route_cases else 0,
            'detailed_results': self.test_results
        }
    
    def export_results(self, filename: str = None) -> str:
        """Export all test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"qa_test_results_{timestamp}.json"
        
        if not self.test_results:
            print("No test results to export. Run tests first.")
            return ""
        
        export_data = {
            'metadata': {
                'test_timestamp': datetime.now().isoformat(),
                'total_test_cases': len(self.get_test_cases()),
                'engine_routes': self.engine.get_available_routes(),
                'test_type': 'comprehensive_qa_evaluation',
                'qa_statistics': self.get_statistics()
            },
            'overall_results': {
                'total_tests': len(self.test_results),
                'correct_routes': len([r for r in self.test_results if r['correct']]),
                'accuracy': len([r for r in self.test_results if r['correct']]) / len(self.test_results)
            },
            'detailed_results': self.test_results,
            'engine_statistics': self.engine.get_statistics()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results exported to: {filename}")
        return filename

def main():
    """Run comprehensive QA testing"""
    print("🤖 Comprehensive Tool Learning QA Test Suite")
    print("=" * 50)
    
    qa_tests = ComprehensiveQATests()
    
    # Show test statistics
    stats = qa_tests.get_statistics()
    print(f"📊 Test Suite Statistics:")
    print(f"  Total test cases: {stats['total_cases']}")
    print(f"  Route distribution: {stats['route_distribution']}")
    print(f"  Difficulty levels: {stats['difficulty_distribution']}")
    print(f"  Multi-route cases: {stats['multi_route_cases']}")
    
    # Run comprehensive tests
    test_results = qa_tests.run_comprehensive_tests()
    
    # Export results
    export_file = qa_tests.export_results()
    
    print(f"\n🎯 Final Summary:")
    print(f"  Tests Run: {test_results['total_tests']}")
    print(f"  Accuracy: {test_results['overall_accuracy']:.1%}")
    print(f"  Results: {export_file}")
    
    if test_results['overall_accuracy'] >= 0.8:
        print("🎉 Great performance! System is ready for deployment.")
    elif test_results['overall_accuracy'] >= 0.6:
        print("⚠️  Moderate performance. Consider improving route patterns.")
    else:
        print("❌ Poor performance. Significant improvements needed.")

if __name__ == "__main__":
    main() 