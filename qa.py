#!/usr/bin/env python3
"""
QA Dataset for Tool Learning System

Test cases covering all 7 routes with multi-route support.
Based on actual static arXiv dataset content.
"""

from typing import Dict, List, Any

# ============================================================================
# QA DATASET BASED ON ACTUAL STATIC DATA
# ============================================================================

COMPREHENSIVE_QA_DATASET = [
    # searchPapers Route (20 cases)
    {
        'id': 1, 
        'query': 'find papers about machine learning optimization', 
        'expected_route': 'searchPapers', 
        'difficulty': 'easy', 
        'category': 'basic_search',
        'expected_answer': 'Should find papers on ML optimization methods including gradient descent, Bayesian optimization, and learning algorithms.',
        'expected_papers_min': 3,
        'expected_keywords': ['machine learning', 'optimization', 'gradient', 'learning']
    },
    {
        'id': 2, 
        'query': 'search for adversarial machine learning papers', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'specific_topic',
        'expected_answer': 'Should find papers on adversarial ML, attacks, and defense mechanisms.',
        'expected_papers_min': 2,
        'expected_keywords': ['adversarial', 'machine learning', 'attack', 'control']
    },
    {
        'id': 3, 
        'query': 'find BERT papers and applications', 
        'expected_route': 'searchPapers', 
        'difficulty': 'easy', 
        'category': 'specific_model',
        'expected_answer': 'Should find BERT-related papers including BERT variants, applications, and improvements.',
        'expected_papers_min': 5,
        'expected_keywords': ['BERT', 'bidirectional', 'encoder', 'transformer']
    },
    {
        'id': 4, 
        'query': 'healthcare machine learning applications', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'domain_application',
        'expected_answer': 'Should find papers on ML applications in healthcare, clinical prediction, and medical AI.',
        'expected_papers_min': 2,
        'expected_keywords': ['healthcare', 'clinical', 'medical', 'machine learning']
    },
    {
        'id': 5, 
        'query': 'tool learning and agent systems', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'agent_systems',
        'expected_answer': 'Should find papers on tool learning, agent-based systems, and tool use.',
        'expected_papers_min': 3,
        'expected_keywords': ['tool learning', 'agent', 'tool use', 'reasoning']
    },
    {
        'id': 6, 
        'query': 'retrieval augmented generation RAG systems', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'rag_systems',
        'expected_answer': 'Should find papers on RAG systems, retrieval-augmented generation, and knowledge integration.',
        'expected_papers_min': 3,
        'expected_keywords': ['retrieval', 'augmented', 'generation', 'RAG']
    },
    {
        'id': 7, 
        'query': 'hallucination detection in language models', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'category': 'factuality',
        'expected_answer': 'Should find papers on hallucination detection, factuality verification, and truthfulness.',
        'expected_papers_min': 3,
        'expected_keywords': ['hallucination', 'detection', 'factuality', 'truthfulness']
    },
    {
        'id': 8, 
        'query': 'federated learning and distributed ML', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'distributed_learning',
        'expected_answer': 'Should find papers on federated learning, distributed machine learning, and privacy-preserving ML.',
        'expected_papers_min': 2,
        'expected_keywords': ['federated', 'distributed', 'machine learning', 'privacy']
    },
    {
        'id': 9, 
        'query': 'automated machine learning AutoML', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'automl',
        'expected_answer': 'Should find papers on automated machine learning, AutoML techniques, and automated optimization.',
        'expected_papers_min': 2,
        'expected_keywords': ['automated', 'AutoML', 'machine learning', 'optimization']
    },
    {
        'id': 10, 
        'query': 'quantum machine learning methods', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'category': 'quantum_ml',
        'expected_answer': 'Should find papers on quantum machine learning, quantum algorithms, and quantum computing for ML.',
        'expected_papers_min': 2,
        'expected_keywords': ['quantum', 'machine learning', 'quantum computing', 'algorithm']
    },
    {
        'id': 11, 
        'query': 'machine learning interpretability and explainability', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'interpretability',
        'expected_answer': 'Should find papers on ML interpretability, explainable AI, and model transparency.',
        'expected_papers_min': 2,
        'expected_keywords': ['interpretability', 'explainable', 'transparency', 'machine learning']
    },
    {
        'id': 12, 
        'query': 'deep learning for natural language processing', 
        'expected_route': 'searchPapers', 
        'difficulty': 'easy', 
        'category': 'deep_learning_nlp',
        'expected_answer': 'Should find papers on deep learning applications in NLP, language models, and text processing.',
        'expected_papers_min': 3,
        'expected_keywords': ['deep learning', 'natural language', 'NLP', 'language model']
    },
    {
        'id': 13, 
        'query': 'reinforcement learning and Q-learning', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'reinforcement_learning',
        'expected_answer': 'Should find papers on reinforcement learning algorithms, Q-learning, and RL applications.',
        'expected_papers_min': 1,
        'expected_keywords': ['reinforcement learning', 'Q-learning', 'RL', 'learning']
    },
    {
        'id': 14, 
        'query': 'meta-learning and learning to learn', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'category': 'meta_learning',
        'expected_answer': 'Should find papers on meta-learning, learning to learn, and few-shot learning.',
        'expected_papers_min': 1,
        'expected_keywords': ['meta-learning', 'learning to learn', 'few-shot', 'transfer']
    },
    {
        'id': 15, 
        'query': 'computer vision and machine learning', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'computer_vision',
        'expected_answer': 'Should find papers on computer vision applications of ML, image processing, and visual recognition.',
        'expected_papers_min': 1,
        'expected_keywords': ['computer vision', 'machine learning', 'image', 'visual']
    },
    {
        'id': 16, 
        'query': 'time series machine learning', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'time_series',
        'expected_answer': 'Should find papers on time series analysis with ML, temporal modeling, and forecasting.',
        'expected_papers_min': 1,
        'expected_keywords': ['time series', 'temporal', 'machine learning', 'forecasting']
    },
    {
        'id': 17, 
        'query': 'privacy preserving machine learning', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'category': 'privacy',
        'expected_answer': 'Should find papers on privacy-preserving ML, differential privacy, and secure learning.',
        'expected_papers_min': 1,
        'expected_keywords': ['privacy', 'preserving', 'machine learning', 'secure']
    },
    {
        'id': 18, 
        'query': 'transfer learning techniques', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'transfer_learning',
        'expected_answer': 'Should find papers on transfer learning methods, domain adaptation, and knowledge transfer.',
        'expected_papers_min': 1,
        'expected_keywords': ['transfer learning', 'domain adaptation', 'knowledge transfer', 'learning']
    },
    {
        'id': 19, 
        'query': 'machine learning for scientific computing', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'scientific_ml',
        'expected_answer': 'Should find papers on ML applications in scientific computing, physics-informed ML, and scientific discovery.',
        'expected_papers_min': 1,
        'expected_keywords': ['scientific', 'computing', 'machine learning', 'physics']
    },
    {
        'id': 20, 
        'query': 'unsupervised learning and clustering', 
        'expected_route': 'searchPapers', 
        'difficulty': 'easy', 
        'category': 'unsupervised',
        'expected_answer': 'Should find papers on unsupervised learning methods, clustering algorithms, and dimensionality reduction.',
        'expected_papers_min': 1,
        'expected_keywords': ['unsupervised learning', 'clustering', 'dimensionality', 'learning']
    },

    # getAuthorInfo Route (8 cases)
    {
        'id': 21, 
        'query': 'who is Elad Hazan', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'category': 'author_info',
        'expected_answer': 'Should provide information about Elad Hazan, author of optimization for machine learning papers.',
        'expected_papers_min': 0,
        'expected_keywords': ['Elad Hazan', 'optimization', 'machine learning', 'Princeton']
    },
    {
        'id': 22, 
        'query': 'research by Wei-Hung Weng', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'category': 'author_info',
        'expected_answer': 'Should provide information about Wei-Hung Weng, known for clinical ML and healthcare applications.',
        'expected_papers_min': 0,
        'expected_keywords': ['Wei-Hung Weng', 'clinical', 'healthcare', 'machine learning']
    },
    {
        'id': 23, 
        'query': 'tell me about Ian Goodfellow', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'easy', 
        'category': 'famous_author',
        'expected_answer': 'Should provide information about Ian Goodfellow, inventor of GANs and deep learning researcher.',
        'expected_papers_min': 0,
        'expected_keywords': ['Ian Goodfellow', 'GAN', 'deep learning', 'generative']
    },
    {
        'id': 24, 
        'query': 'papers by Amnon Shashua', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'category': 'author_info',
        'expected_answer': 'Should provide information about Amnon Shashua, author of machine learning class notes.',
        'expected_papers_min': 0,
        'expected_keywords': ['Amnon Shashua', 'machine learning', 'statistical inference', 'education']
    },
    {
        'id': 25, 
        'query': 'research profile of Xiaojin Zhu', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'category': 'author_info',
        'expected_answer': 'Should provide information about Xiaojin Zhu, known for adversarial ML and optimal control.',
        'expected_papers_min': 0,
        'expected_keywords': ['Xiaojin Zhu', 'adversarial', 'optimal control', 'machine learning']
    },
    {
        'id': 26, 
        'query': 'who is Yoshua Bengio', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'easy', 
        'category': 'famous_author',
        'expected_answer': 'Should provide information about Yoshua Bengio, deep learning pioneer and researcher.',
        'expected_papers_min': 0,
        'expected_keywords': ['Yoshua Bengio', 'deep learning', 'neural networks', 'AI']
    },
    {
        'id': 27, 
        'query': 'publications by Ayaz Akram', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'hard', 
        'category': 'specific_author',
        'expected_answer': 'Should provide information about Ayaz Akram, researcher in ML and computer architecture.',
        'expected_papers_min': 0,
        'expected_keywords': ['Ayaz Akram', 'computer architecture', 'machine learning', 'tribes']
    },
    {
        'id': 28, 
        'query': 'research background of Michail Schlesinger', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'hard', 
        'category': 'specific_author',
        'expected_answer': 'Should provide information about Michail Schlesinger, known for minimax deviation strategies.',
        'expected_papers_min': 0,
        'expected_keywords': ['Michail Schlesinger', 'minimax', 'deviation', 'learning samples']
    },

    # getCitations Route (6 cases)
    {
        'id': 29, 
        'query': 'citation analysis for machine learning papers', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'category': 'citation_analysis',
        'expected_answer': 'Should analyze citations and impact of machine learning research papers.',
        'expected_papers_min': 1,
        'expected_keywords': ['citation', 'analysis', 'machine learning', 'impact']
    },
    {
        'id': 30, 
        'query': 'impact factor of optimization research', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'category': 'citation_analysis',
        'expected_answer': 'Should discuss impact and citation metrics for optimization research.',
        'expected_papers_min': 0,
        'expected_keywords': ['impact factor', 'optimization', 'research', 'citation']
    },
    {
        'id': 31, 
        'query': 'bibliometric study of adversarial ML', 
        'expected_route': 'getCitations', 
        'difficulty': 'hard', 
        'category': 'bibliometric',
        'expected_answer': 'Should provide bibliometric analysis of adversarial machine learning research.',
        'expected_papers_min': 0,
        'expected_keywords': ['bibliometric', 'adversarial', 'machine learning', 'study']
    },
    {
        'id': 32, 
        'query': 'citation count for BERT papers', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'category': 'model_citations',
        'expected_answer': 'Should provide citation information and impact analysis for BERT-related papers.',
        'expected_papers_min': 1,
        'expected_keywords': ['citation count', 'BERT', 'papers', 'impact']
    },
    {
        'id': 33, 
        'query': 'research impact of federated learning', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'category': 'field_impact',
        'expected_answer': 'Should analyze the research impact and citation patterns of federated learning.',
        'expected_papers_min': 0,
        'expected_keywords': ['research impact', 'federated learning', 'citation', 'patterns']
    },
    {
        'id': 34, 
        'query': 'h-index analysis for ML researchers', 
        'expected_route': 'getCitations', 
        'difficulty': 'hard', 
        'category': 'researcher_metrics',
        'expected_answer': 'Should discuss h-index and other citation metrics for machine learning researchers.',
        'expected_papers_min': 0,
        'expected_keywords': ['h-index', 'ML researchers', 'citation metrics', 'analysis']
    },

    # getRelatedPapers Route (8 cases)
    {
        'id': 35, 
        'query': 'papers related to machine learning optimization', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'related_search',
        'expected_answer': 'Should find papers related to ML optimization, gradient methods, and learning algorithms.',
        'expected_papers_min': 3,
        'expected_keywords': ['related papers', 'optimization', 'machine learning', 'gradient']
    },
    {
        'id': 36, 
        'query': 'similar research to BERT models', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'model_related',
        'expected_answer': 'Should find research similar to BERT, including other transformer models and language models.',
        'expected_papers_min': 3,
        'expected_keywords': ['similar research', 'BERT', 'transformer', 'language model']
    },
    {
        'id': 37, 
        'query': 'connected work to adversarial machine learning', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'security_related',
        'expected_answer': 'Should find work connected to adversarial ML, robustness, and security.',
        'expected_papers_min': 2,
        'expected_keywords': ['connected work', 'adversarial', 'machine learning', 'robustness']
    },
    {
        'id': 38, 
        'query': 'related studies on healthcare ML', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'domain_related',
        'expected_answer': 'Should find studies related to healthcare applications of machine learning.',
        'expected_papers_min': 2,
        'expected_keywords': ['related studies', 'healthcare', 'ML', 'clinical']
    },
    {
        'id': 39, 
        'query': 'papers similar to tool learning research', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'agent_related',
        'expected_answer': 'Should find papers similar to tool learning, including agent systems and reasoning.',
        'expected_papers_min': 2,
        'expected_keywords': ['similar papers', 'tool learning', 'agent', 'reasoning']
    },
    {
        'id': 40, 
        'query': 'related work on federated learning', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'distributed_related',
        'expected_answer': 'Should find work related to federated learning and distributed machine learning.',
        'expected_papers_min': 2,
        'expected_keywords': ['related work', 'federated learning', 'distributed', 'privacy']
    },
    {
        'id': 41, 
        'query': 'connected research to quantum ML', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'hard', 
        'category': 'quantum_related',
        'expected_answer': 'Should find research connected to quantum machine learning and quantum computing.',
        'expected_papers_min': 1,
        'expected_keywords': ['connected research', 'quantum ML', 'quantum computing', 'algorithm']
    },
    {
        'id': 42, 
        'query': 'similar papers to interpretable ML', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'interpretability_related',
        'expected_answer': 'Should find papers similar to interpretable ML, explainable AI, and transparency research.',
        'expected_papers_min': 2,
        'expected_keywords': ['similar papers', 'interpretable', 'explainable', 'transparency']
    },

    # comparePapers Route (8 cases)
    {
        'id': 43, 
        'query': 'compare supervised and unsupervised learning', 
        'expected_route': 'comparePapers', 
        'difficulty': 'easy', 
        'category': 'learning_types',
        'expected_answer': 'Should compare different learning paradigms, their methods, and applications.',
        'expected_papers_min': 2,
        'expected_keywords': ['compare', 'supervised', 'unsupervised', 'learning']
    },
    {
        'id': 44, 
        'query': 'difference between optimization methods', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'category': 'optimization_comparison',
        'expected_answer': 'Should compare different optimization approaches in machine learning.',
        'expected_papers_min': 2,
        'expected_keywords': ['difference', 'optimization methods', 'gradient', 'algorithm']
    },
    {
        'id': 45, 
        'query': 'BERT versus other language models', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'category': 'model_comparison',
        'expected_answer': 'Should compare BERT with other language models and their capabilities.',
        'expected_papers_min': 2,
        'expected_keywords': ['BERT versus', 'language models', 'comparison', 'transformer']
    },
    {
        'id': 46, 
        'query': 'compare centralized vs federated learning', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'category': 'architecture_comparison',
        'expected_answer': 'Should compare centralized and federated learning approaches.',
        'expected_papers_min': 2,
        'expected_keywords': ['compare', 'centralized', 'federated learning', 'distributed']
    },
    {
        'id': 47, 
        'query': 'classical ML versus deep learning approaches', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'category': 'paradigm_comparison',
        'expected_answer': 'Should compare classical machine learning with deep learning methods.',
        'expected_papers_min': 2,
        'expected_keywords': ['classical ML', 'deep learning', 'approaches', 'comparison']
    },
    {
        'id': 48, 
        'query': 'automated ML vs manual ML approaches', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'category': 'automation_comparison',
        'expected_answer': 'Should compare automated machine learning with manual ML development.',
        'expected_papers_min': 1,
        'expected_keywords': ['automated ML', 'manual ML', 'approaches', 'AutoML']
    },
    {
        'id': 49, 
        'query': 'compare privacy-preserving ML techniques', 
        'expected_route': 'comparePapers', 
        'difficulty': 'hard', 
        'category': 'privacy_comparison',
        'expected_answer': 'Should compare different privacy-preserving machine learning techniques.',
        'expected_papers_min': 1,
        'expected_keywords': ['compare', 'privacy-preserving', 'ML techniques', 'privacy']
    },
    {
        'id': 50, 
        'query': 'quantum ML versus classical ML', 
        'expected_route': 'comparePapers', 
        'difficulty': 'hard', 
        'category': 'quantum_comparison',
        'expected_answer': 'Should compare quantum machine learning with classical machine learning approaches.',
        'expected_papers_min': 1,
        'expected_keywords': ['quantum ML', 'classical ML', 'comparison', 'quantum computing']
    },

    # trendAnalysis Route (8 cases)
    {
        'id': 51, 
        'query': 'trends in machine learning research', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'category': 'field_trends',
        'expected_answer': 'Should analyze trends and evolution in machine learning research over time.',
        'expected_papers_min': 3,
        'expected_keywords': ['trends', 'machine learning', 'research', 'evolution']
    },
    {
        'id': 52, 
        'query': 'evolution of optimization algorithms', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'category': 'algorithm_trends',
        'expected_answer': 'Should analyze the evolution and trends in optimization algorithms for ML.',
        'expected_papers_min': 2,
        'expected_keywords': ['evolution', 'optimization algorithms', 'trends', 'development']
    },
    {
        'id': 53, 
        'query': 'progress in BERT and transformer models', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'category': 'model_trends',
        'expected_answer': 'Should analyze progress and trends in BERT and transformer model development.',
        'expected_papers_min': 3,
        'expected_keywords': ['progress', 'BERT', 'transformer models', 'trends']
    },
    {
        'id': 54, 
        'query': 'trends in adversarial machine learning', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'category': 'security_trends',
        'expected_answer': 'Should analyze trends in adversarial ML research and security developments.',
        'expected_papers_min': 2,
        'expected_keywords': ['trends', 'adversarial', 'machine learning', 'security']
    },
    {
        'id': 55, 
        'query': 'emerging patterns in healthcare AI', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'category': 'domain_trends',
        'expected_answer': 'Should analyze emerging patterns and trends in healthcare AI applications.',
        'expected_papers_min': 2,
        'expected_keywords': ['emerging patterns', 'healthcare AI', 'trends', 'medical']
    },
    {
        'id': 56, 
        'query': 'development timeline of federated learning', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'category': 'technology_timeline',
        'expected_answer': 'Should analyze the development timeline and trends in federated learning.',
        'expected_papers_min': 1,
        'expected_keywords': ['development timeline', 'federated learning', 'trends', 'evolution']
    },
    {
        'id': 57, 
        'query': 'trends in quantum machine learning', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'hard', 
        'category': 'quantum_trends',
        'expected_answer': 'Should analyze trends and development in quantum machine learning research.',
        'expected_papers_min': 1,
        'expected_keywords': ['trends', 'quantum machine learning', 'quantum computing', 'development']
    },
    {
        'id': 58, 
        'query': 'research evolution in interpretable AI', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'category': 'interpretability_trends',
        'expected_answer': 'Should analyze research evolution and trends in interpretable AI and explainability.',
        'expected_papers_min': 2,
        'expected_keywords': ['research evolution', 'interpretable AI', 'explainability', 'trends']
    },

    # journalAnalysis Route (6 cases)
    {
        'id': 59, 
        'query': 'best venues for machine learning research', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'medium', 
        'category': 'venue_analysis',
        'expected_answer': 'Should analyze publication venues, conferences, and journals for ML research.',
        'expected_papers_min': 1,
        'expected_keywords': ['best venues', 'machine learning research', 'conferences', 'journals']
    },
    {
        'id': 60, 
        'query': 'top conferences for AI papers', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'medium', 
        'category': 'conference_analysis',
        'expected_answer': 'Should identify and analyze top conferences for AI and ML research publication.',
        'expected_papers_min': 1,
        'expected_keywords': ['top conferences', 'AI papers', 'publication', 'venues']
    },
    {
        'id': 61, 
        'query': 'publication patterns in optimization research', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'medium', 
        'category': 'publication_patterns',
        'expected_answer': 'Should analyze publication patterns and venues for optimization research.',
        'expected_papers_min': 1,
        'expected_keywords': ['publication patterns', 'optimization research', 'venues', 'journals']
    },
    {
        'id': 62, 
        'query': 'journal rankings for deep learning', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'medium', 
        'category': 'journal_rankings',
        'expected_answer': 'Should analyze journal rankings and impact for deep learning research.',
        'expected_papers_min': 1,
        'expected_keywords': ['journal rankings', 'deep learning', 'impact', 'research']
    },
    {
        'id': 63, 
        'query': 'venue analysis for healthcare AI research', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'hard', 
        'category': 'domain_venues',
        'expected_answer': 'Should analyze publication venues specifically for healthcare AI research.',
        'expected_papers_min': 1,
        'expected_keywords': ['venue analysis', 'healthcare AI', 'research', 'medical journals']
    },
    {
        'id': 64, 
        'query': 'conference trends in NLP research', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'medium', 
        'category': 'field_venues',
        'expected_answer': 'Should analyze conference trends and venues for NLP research publication.',
        'expected_papers_min': 1,
        'expected_keywords': ['conference trends', 'NLP research', 'venues', 'natural language']
    },

    # Multi-Route Cases (10 cases)
    {
        'id': 65, 
        'query': 'find papers by Elad Hazan on optimization', 
        'expected_route': 'searchPapers, getAuthorInfo', 
        'difficulty': 'medium', 
        'category': 'multi_route',
        'expected_answer': 'Should find optimization papers and provide author information about Elad Hazan.',
        'expected_papers_min': 1,
        'expected_keywords': ['Elad Hazan', 'optimization', 'papers', 'author info']
    },
    {
        'id': 66, 
        'query': 'compare BERT models and analyze their citations', 
        'expected_route': 'comparePapers, getCitations', 
        'difficulty': 'hard', 
        'category': 'multi_route',
        'expected_answer': 'Should compare different BERT models and analyze their citation impact.',
        'expected_papers_min': 2,
        'expected_keywords': ['compare', 'BERT models', 'citations', 'analysis']
    },
    {
        'id': 67, 
        'query': 'search for federated learning papers and find related work', 
        'expected_route': 'searchPapers, getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'multi_route',
        'expected_answer': 'Should find federated learning papers and identify related research.',
        'expected_papers_min': 2,
        'expected_keywords': ['federated learning', 'papers', 'related work', 'distributed']
    },
    {
        'id': 68, 
        'query': 'analyze trends in machine learning and best publication venues', 
        'expected_route': 'trendAnalysis, journalAnalysis', 
        'difficulty': 'hard', 
        'category': 'multi_route',
        'expected_answer': 'Should analyze ML trends and identify top publication venues.',
        'expected_papers_min': 2,
        'expected_keywords': ['trends', 'machine learning', 'publication venues', 'analysis']
    },
    {
        'id': 69, 
        'query': 'who is Ian Goodfellow and what are his key papers', 
        'expected_route': 'getAuthorInfo, searchPapers', 
        'difficulty': 'medium', 
        'category': 'multi_route',
        'expected_answer': 'Should provide author information and find key papers by Ian Goodfellow.',
        'expected_papers_min': 1,
        'expected_keywords': ['Ian Goodfellow', 'author info', 'key papers', 'research']
    },
    {
        'id': 70, 
        'query': 'find adversarial ML papers and compare different approaches', 
        'expected_route': 'searchPapers, comparePapers', 
        'difficulty': 'hard', 
        'category': 'multi_route',
        'expected_answer': 'Should find adversarial ML papers and compare different adversarial approaches.',
        'expected_papers_min': 2,
        'expected_keywords': ['adversarial ML', 'papers', 'compare', 'approaches']
    },
    {
        'id': 71, 
        'query': 'analyze healthcare AI trends and related research', 
        'expected_route': 'trendAnalysis, getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'multi_route',
        'expected_answer': 'Should analyze healthcare AI trends and find related research.',
        'expected_papers_min': 2,
        'expected_keywords': ['healthcare AI', 'trends', 'related research', 'medical']
    },
    {
        'id': 72, 
        'query': 'compare optimization methods and analyze their impact', 
        'expected_route': 'comparePapers, getCitations', 
        'difficulty': 'hard', 
        'category': 'multi_route',
        'expected_answer': 'Should compare optimization methods and analyze their research impact.',
        'expected_papers_min': 2,
        'expected_keywords': ['optimization methods', 'compare', 'impact', 'citations']
    },
    {
        'id': 73, 
        'query': 'research by Wei-Hung Weng and similar healthcare papers', 
        'expected_route': 'getAuthorInfo, getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'multi_route',
        'expected_answer': 'Should provide author information and find similar healthcare research.',
        'expected_papers_min': 1,
        'expected_keywords': ['Wei-Hung Weng', 'healthcare papers', 'author info', 'clinical']
    },
    {
        'id': 74, 
        'query': 'quantum ML research trends and publication venues', 
        'expected_route': 'trendAnalysis, journalAnalysis', 
        'difficulty': 'hard', 
        'category': 'multi_route',
        'expected_answer': 'Should analyze quantum ML trends and identify publication venues.',
        'expected_papers_min': 1,
        'expected_keywords': ['quantum ML', 'trends', 'publication venues', 'research']
    }
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_dataset_statistics() -> Dict[str, Any]:
    """Get comprehensive statistics about the QA dataset"""
    total_cases = len(COMPREHENSIVE_QA_DATASET)
    
    route_counts = {}
    difficulty_counts = {}
    category_counts = {}
    
    for case in COMPREHENSIVE_QA_DATASET:
        # Count routes (handle multi-route)
        routes = case['expected_route'].split(', ')
        for route in routes:
            route_counts[route.strip()] = route_counts.get(route.strip(), 0) + 1
        
        # Count difficulties
        diff = case['difficulty']
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        # Count categories
        cat = case['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    return {
        'total_cases': total_cases,
        'route_distribution': route_counts,
        'difficulty_distribution': difficulty_counts,
        'category_distribution': category_counts,
        'multi_route_cases': len([c for c in COMPREHENSIVE_QA_DATASET if ',' in c['expected_route']])
    }

def get_cases_by_route(route: str) -> List[Dict[str, Any]]:
    """Get all test cases for a specific route"""
    return [case for case in COMPREHENSIVE_QA_DATASET if route in case['expected_route']]

def get_cases_by_difficulty(difficulty: str) -> List[Dict[str, Any]]:
    """Get all test cases for a specific difficulty"""
    return [case for case in COMPREHENSIVE_QA_DATASET if case['difficulty'] == difficulty]

def get_expected_answer(test_id: int) -> Dict[str, Any]:
    """Get expected answer for a specific test case"""
    for case in COMPREHENSIVE_QA_DATASET:
        if case['id'] == test_id:
            return case
    return {}

def validate_test_case(case: Dict[str, Any]) -> bool:
    """Validate that a test case has all required fields"""
    required_fields = ['id', 'query', 'expected_route', 'difficulty', 'category', 
                      'expected_answer', 'expected_papers_min', 'expected_keywords']
    return all(field in case for field in required_fields)

def get_multi_route_cases() -> List[Dict[str, Any]]:
    """Get all multi-route test cases"""
    return [case for case in COMPREHENSIVE_QA_DATASET if ',' in case['expected_route']]

def main():
    """Print dataset statistics"""
    print("QA Dataset Statistics:")
    print("=" * 50)
    
    stats = get_dataset_statistics()
    print(f"Total test cases: {stats['total_cases']}")
    print(f"Multi-route cases: {stats['multi_route_cases']}")
    print()
    
    print("Route Distribution:")
    for route, count in sorted(stats['route_distribution'].items()):
        print(f"  {route}: {count}")
    print()
    
    print("Difficulty Distribution:")
    for difficulty, count in sorted(stats['difficulty_distribution'].items()):
        print(f"  {difficulty}: {count}")
    print()
    
    print("Top Categories:")
    sorted_categories = sorted(stats['category_distribution'].items(), 
                              key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories[:10]:
        print(f"  {category}: {count}")

if __name__ == "__main__":
    main() 