#!/usr/bin/env python3
"""
Comprehensive QA Dataset for Tool Learning System

78 test cases covering all 7 routes with multiple difficulty levels.
Includes expected answers for ablation testing and comparison.
"""

from typing import Dict, List, Any

# ============================================================================
# COMPREHENSIVE TEST CASES (78 Questions) WITH EXPECTED ANSWERS
# ============================================================================

COMPREHENSIVE_QA_DATASET = [
    # searchPapers Route (25 cases)
    {
        'id': 1, 
        'query': 'find papers about machine learning', 
        'expected_route': 'searchPapers', 
        'difficulty': 'easy', 
        'category': 'basic_search',
        'expected_answer': 'Should find papers related to machine learning algorithms, optimization, and learning theory. Expected topics include supervised learning, unsupervised learning, optimization methods, and machine learning foundations.',
        'expected_papers_min': 3,
        'expected_keywords': ['machine learning', 'algorithm', 'optimization', 'learning']
    },
    {
        'id': 2, 
        'query': 'search for deep learning research', 
        'expected_route': 'searchPapers', 
        'difficulty': 'easy', 
        'category': 'basic_search',
        'expected_answer': 'Should find papers on deep learning, neural networks, architectures, and training methods. Expected topics include neural network architectures, deep learning methods, and training techniques.',
        'expected_papers_min': 3,
        'expected_keywords': ['deep learning', 'neural network', 'architecture', 'training']
    },
    {
        'id': 3, 
        'query': 'get papers on artificial intelligence', 
        'expected_route': 'searchPapers', 
        'difficulty': 'easy', 
        'category': 'basic_search',
        'expected_answer': 'Should find general AI papers covering various AI topics including machine learning, deep learning, and AI applications.',
        'expected_papers_min': 2,
        'expected_keywords': ['artificial intelligence', 'AI', 'machine learning', 'intelligent']
    },
    {
        'id': 4, 
        'query': 'lookup neural network papers', 
        'expected_route': 'searchPapers', 
        'difficulty': 'easy', 
        'category': 'basic_search',
        'expected_answer': 'Should find papers specifically about neural networks, their architectures, and applications.',
        'expected_papers_min': 3,
        'expected_keywords': ['neural network', 'network', 'neural', 'architecture']
    },
    {
        'id': 5, 
        'query': 'find research on natural language processing', 
        'expected_route': 'searchPapers', 
        'difficulty': 'easy', 
        'category': 'basic_search',
        'expected_answer': 'Should find NLP-related papers, potentially including BERT and language model research.',
        'expected_papers_min': 2,
        'expected_keywords': ['natural language', 'NLP', 'language', 'text']
    },
    {
        'id': 6, 
        'query': 'transformer architecture papers', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'specific_topic',
        'expected_answer': 'Should find papers about transformer architectures or mathematical transformations (depending on dataset content).',
        'expected_papers_min': 2,
        'expected_keywords': ['transformer', 'transform', 'architecture']
    },
    {
        'id': 7, 
        'query': 'BERT model research papers', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'specific_topic',
        'expected_answer': 'Should find papers specifically about BERT models, variants, and applications. Expected papers include BERT-DRE, BERT-JAM, ExtremeBERT.',
        'expected_papers_min': 3,
        'expected_keywords': ['BERT', 'bidirectional', 'encoder', 'language model']
    },
    {
        'id': 8, 
        'query': 'retrieval augmented generation studies', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'specific_topic',
        'expected_answer': 'Should find papers about RAG systems and retrieval-augmented generation methods.',
        'expected_papers_min': 2,
        'expected_keywords': ['retrieval', 'augmented', 'generation', 'RAG']
    },
    {
        'id': 9, 
        'query': 'hallucination detection in language models', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'category': 'specific_topic',
        'expected_answer': 'Should find papers about hallucination detection, factuality verification, and truthfulness in language models.',
        'expected_papers_min': 2,
        'expected_keywords': ['hallucination', 'detection', 'factuality', 'truthfulness']
    },
    {
        'id': 10, 
        'query': 'tool learning and agent-based systems', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'category': 'specific_topic',
        'expected_answer': 'Should find papers about tool learning, agent systems, and tool use in AI.',
        'expected_papers_min': 2,
        'expected_keywords': ['tool learning', 'agent', 'tool use', 'reasoning']
    },
    {
        'id': 11, 
        'query': 'attention mechanisms in neural networks', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'specific_topic',
        'expected_answer': 'Should find papers about attention mechanisms, self-attention, and attention-based models.',
        'expected_papers_min': 2,
        'expected_keywords': ['attention', 'mechanism', 'self-attention', 'neural']
    },
    {
        'id': 12, 
        'query': 'convolutional neural networks for computer vision', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'specific_topic',
        'expected_answer': 'Should find papers about CNNs and computer vision applications.',
        'expected_papers_min': 1,
        'expected_keywords': ['convolutional', 'CNN', 'computer vision', 'neural network']
    },
    {
        'id': 13, 
        'query': 'reinforcement learning algorithms', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'specific_topic',
        'expected_answer': 'Should find papers about reinforcement learning methods and algorithms.',
        'expected_papers_min': 1,
        'expected_keywords': ['reinforcement learning', 'RL', 'algorithm', 'learning']
    },
    {
        'id': 14, 
        'query': 'papers on gradient descent optimization', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'category': 'technical',
        'expected_answer': 'Should find papers about optimization methods, gradient descent, and optimization algorithms.',
        'expected_papers_min': 2,
        'expected_keywords': ['optimization', 'gradient', 'descent', 'algorithm']
    },
    {
        'id': 15, 
        'query': 'research on transfer learning techniques', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'technical',
        'expected_answer': 'Should find papers about transfer learning methods and techniques.',
        'expected_papers_min': 1,
        'expected_keywords': ['transfer learning', 'transfer', 'learning', 'technique']
    },
    {
        'id': 16, 
        'query': 'find studies on adversarial examples', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'category': 'technical',
        'expected_answer': 'Should find papers about adversarial examples, adversarial attacks, and robustness.',
        'expected_papers_min': 1,
        'expected_keywords': ['adversarial', 'examples', 'attack', 'robustness']
    },
    {
        'id': 17, 
        'query': 'papers about model interpretability and explainability', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'category': 'technical',
        'expected_answer': 'Should find papers about explainable AI, model interpretability, and transparency.',
        'expected_papers_min': 1,
        'expected_keywords': ['interpretability', 'explainability', 'explainable', 'transparency']
    },
    {
        'id': 18, 
        'query': 'research on federated learning systems', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'category': 'technical',
        'expected_answer': 'Should find papers about federated learning, distributed learning, and privacy-preserving ML.',
        'expected_papers_min': 1,
        'expected_keywords': ['federated learning', 'federated', 'distributed', 'privacy']
    },
    {
        'id': 19, 
        'query': 'find papers on few-shot learning', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'technical',
        'expected_answer': 'Should find papers about few-shot learning methods and techniques.',
        'expected_papers_min': 1,
        'expected_keywords': ['few-shot', 'few shot', 'learning', 'shot']
    },
    {
        'id': 20, 
        'query': 'zero-shot learning and generalization', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'category': 'technical',
        'expected_answer': 'Should find papers about zero-shot learning and generalization methods.',
        'expected_papers_min': 1,
        'expected_keywords': ['zero-shot', 'zero shot', 'generalization', 'learning']
    },
    {
        'id': 21, 
        'query': 'machine learning for healthcare applications', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'domain',
        'expected_answer': 'Should find papers about ML applications in healthcare and medical domains.',
        'expected_papers_min': 1,
        'expected_keywords': ['healthcare', 'medical', 'machine learning', 'application']
    },
    {
        'id': 22, 
        'query': 'AI in autonomous driving systems', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'domain',
        'expected_answer': 'Should find papers about AI applications in autonomous vehicles and driving.',
        'expected_papers_min': 1,
        'expected_keywords': ['autonomous', 'driving', 'vehicle', 'AI']
    },
    {
        'id': 23, 
        'query': 'natural language processing for finance', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'domain',
        'expected_answer': 'Should find papers about NLP applications in financial domains.',
        'expected_papers_min': 1,
        'expected_keywords': ['finance', 'financial', 'NLP', 'language']
    },
    {
        'id': 24, 
        'query': 'computer vision for robotics', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'category': 'domain',
        'expected_answer': 'Should find papers about computer vision applications in robotics.',
        'expected_papers_min': 1,
        'expected_keywords': ['computer vision', 'robotics', 'robot', 'vision']
    },
    {
        'id': 25, 
        'query': 'deep learning for drug discovery', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'category': 'domain',
        'expected_answer': 'Should find papers about deep learning applications in drug discovery and pharmaceutical research.',
        'expected_papers_min': 1,
        'expected_keywords': ['drug discovery', 'pharmaceutical', 'deep learning', 'drug']
    },

    # getAuthorInfo Route (12 cases)
    {
        'id': 26, 
        'query': 'who is Geoffrey Hinton', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'easy', 
        'category': 'famous_author',
        'expected_answer': 'Should provide information about Geoffrey Hinton, a renowned AI researcher known for deep learning and neural networks. Often called the "Godfather of AI".',
        'expected_papers_min': 0,
        'expected_keywords': ['Geoffrey Hinton', 'deep learning', 'neural networks', 'AI researcher']
    },
    {
        'id': 27, 
        'query': 'tell me about Yann LeCun', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'easy', 
        'category': 'famous_author',
        'expected_answer': 'Should provide information about Yann LeCun, known for convolutional neural networks and computer vision research.',
        'expected_papers_min': 0,
        'expected_keywords': ['Yann LeCun', 'convolutional', 'computer vision', 'CNN']
    },
    {
        'id': 28, 
        'query': 'research by Yoshua Bengio', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'easy', 
        'category': 'famous_author',
        'expected_answer': 'Should provide information about Yoshua Bengio, known for deep learning research and RNNs.',
        'expected_papers_min': 0,
        'expected_keywords': ['Yoshua Bengio', 'deep learning', 'RNN', 'researcher']
    },
    {
        'id': 29, 
        'query': 'who is Andrew Ng and what are his contributions', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'easy', 
        'category': 'famous_author',
        'expected_answer': 'Should provide information about Andrew Ng, known for machine learning education, Coursera, and AI applications.',
        'expected_papers_min': 0,
        'expected_keywords': ['Andrew Ng', 'machine learning', 'education', 'AI']
    },
    {
        'id': 30, 
        'query': 'papers by Ian Goodfellow', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'category': 'famous_author',
        'expected_answer': 'Should provide information about Ian Goodfellow, inventor of GANs (Generative Adversarial Networks).',
        'expected_papers_min': 0,
        'expected_keywords': ['Ian Goodfellow', 'GAN', 'generative', 'adversarial']
    },
    {
        'id': 31, 
        'query': 'publications of Fei-Fei Li', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'category': 'famous_author',
        'expected_answer': 'Should provide information about Fei-Fei Li, known for computer vision and ImageNet.',
        'expected_papers_min': 0,
        'expected_keywords': ['Fei-Fei Li', 'computer vision', 'ImageNet', 'vision']
    },
    {
        'id': 32, 
        'query': 'research background of Ashish Vaswani', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'category': 'specific_author',
        'expected_answer': 'Should provide information about Ashish Vaswani, co-author of the transformer architecture paper.',
        'expected_papers_min': 0,
        'expected_keywords': ['Ashish Vaswani', 'transformer', 'attention', 'Google']
    },
    {
        'id': 33, 
        'query': 'who is Sebastian Ruder', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'category': 'specific_author',
        'expected_answer': 'Should provide information about Sebastian Ruder, known for NLP research and transfer learning.',
        'expected_papers_min': 0,
        'expected_keywords': ['Sebastian Ruder', 'NLP', 'transfer learning', 'researcher']
    },
    {
        'id': 34, 
        'query': 'papers and contributions by Ilya Sutskever', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'category': 'specific_author',
        'expected_answer': 'Should provide information about Ilya Sutskever, co-founder of OpenAI and sequence-to-sequence learning.',
        'expected_papers_min': 0,
        'expected_keywords': ['Ilya Sutskever', 'OpenAI', 'sequence', 'learning']
    },
    {
        'id': 35, 
        'query': 'research profile of Andrej Karpathy', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'category': 'specific_author',
        'expected_answer': 'Should provide information about Andrej Karpathy, known for computer vision and neural networks.',
        'expected_papers_min': 0,
        'expected_keywords': ['Andrej Karpathy', 'computer vision', 'neural networks', 'Tesla']
    },
    {
        'id': 36, 
        'query': 'author information for Jacob Devlin', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'hard', 
        'category': 'specific_author',
        'expected_answer': 'Should provide information about Jacob Devlin, lead author of BERT paper at Google.',
        'expected_papers_min': 0,
        'expected_keywords': ['Jacob Devlin', 'BERT', 'Google', 'NLP']
    },
    {
        'id': 37, 
        'query': 'publications by Alec Radford', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'hard', 
        'category': 'specific_author',
        'expected_answer': 'Should provide information about Alec Radford, lead author of GPT papers at OpenAI.',
        'expected_papers_min': 0,
        'expected_keywords': ['Alec Radford', 'GPT', 'OpenAI', 'language model']
    },

    # getCitations Route (10 cases) - Note: Limited real citation data
    {
        'id': 38, 
        'query': 'citation count for BERT paper', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'category': 'basic_citation',
        'expected_answer': 'Should provide citation information for BERT papers, though specific counts may not be available in static dataset.',
        'expected_papers_min': 1,
        'expected_keywords': ['citation', 'BERT', 'impact', 'count']
    },
    {
        'id': 39, 
        'query': 'how many citations does the transformer paper have', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'category': 'basic_citation',
        'expected_answer': 'Should provide citation information for transformer-related papers.',
        'expected_papers_min': 1,
        'expected_keywords': ['citation', 'transformer', 'paper', 'count']
    },
    {
        'id': 40, 
        'query': 'impact factor of GPT research', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'category': 'basic_citation',
        'expected_answer': 'Should discuss the impact and influence of GPT research.',
        'expected_papers_min': 0,
        'expected_keywords': ['impact', 'GPT', 'influence', 'research']
    },
    {
        'id': 41, 
        'query': 'citation analysis for ResNet paper', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'category': 'basic_citation',
        'expected_answer': 'Should provide analysis of ResNet paper citations and impact.',
        'expected_papers_min': 0,
        'expected_keywords': ['citation', 'ResNet', 'analysis', 'impact']
    },
    {
        'id': 42, 
        'query': 'bibliometric study of attention mechanisms', 
        'expected_route': 'getCitations', 
        'difficulty': 'hard', 
        'category': 'impact_analysis',
        'expected_answer': 'Should provide bibliometric analysis of attention mechanism research.',
        'expected_papers_min': 0,
        'expected_keywords': ['bibliometric', 'attention', 'study', 'analysis']
    },
    {
        'id': 43, 
        'query': 'h-index for language model research', 
        'expected_route': 'getCitations', 
        'difficulty': 'hard', 
        'category': 'impact_analysis',
        'expected_answer': 'Should discuss h-index metrics for language model research.',
        'expected_papers_min': 0,
        'expected_keywords': ['h-index', 'language model', 'metric', 'research']
    },
    {
        'id': 44, 
        'query': 'paper impact assessment for deep learning', 
        'expected_route': 'getCitations', 
        'difficulty': 'hard', 
        'category': 'impact_analysis',
        'expected_answer': 'Should assess the impact of deep learning papers and research.',
        'expected_papers_min': 1,
        'expected_keywords': ['impact', 'assessment', 'deep learning', 'paper']
    },
    {
        'id': 45, 
        'query': 'citation metrics for computer vision papers', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'category': 'domain_citation',
        'expected_answer': 'Should provide citation metrics for computer vision research.',
        'expected_papers_min': 0,
        'expected_keywords': ['citation', 'metrics', 'computer vision', 'papers']
    },
    {
        'id': 46, 
        'query': 'impact analysis of NLP research papers', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'category': 'domain_citation',
        'expected_answer': 'Should analyze the impact of NLP research papers.',
        'expected_papers_min': 1,
        'expected_keywords': ['impact', 'analysis', 'NLP', 'research']
    },
    {
        'id': 47, 
        'query': 'citation patterns in machine learning', 
        'expected_route': 'getCitations', 
        'difficulty': 'hard', 
        'category': 'domain_citation',
        'expected_answer': 'Should analyze citation patterns and trends in machine learning research.',
        'expected_papers_min': 1,
        'expected_keywords': ['citation', 'patterns', 'machine learning', 'trends']
    },

    # getRelatedPapers Route (8 cases)
    {
        'id': 48, 
        'query': 'papers related to transformer architectures', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'related_search',
        'expected_answer': 'Should find papers related to transformer architectures and attention mechanisms.',
        'expected_papers_min': 2,
        'expected_keywords': ['related', 'transformer', 'architecture', 'attention']
    },
    {
        'id': 49, 
        'query': 'similar research to BERT', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'related_search',
        'expected_answer': 'Should find research similar to BERT, including other language models and encoder architectures.',
        'expected_papers_min': 2,
        'expected_keywords': ['similar', 'BERT', 'language model', 'encoder']
    },
    {
        'id': 50, 
        'query': 'connected work to hallucination detection', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'hard', 
        'category': 'related_search',
        'expected_answer': 'Should find work connected to hallucination detection and factuality verification.',
        'expected_papers_min': 1,
        'expected_keywords': ['connected', 'hallucination', 'detection', 'factuality']
    },
    {
        'id': 51, 
        'query': 'related papers on retrieval systems', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'related_search',
        'expected_answer': 'Should find papers related to retrieval systems and information retrieval.',
        'expected_papers_min': 1,
        'expected_keywords': ['related', 'retrieval', 'systems', 'information']
    },
    {
        'id': 52, 
        'query': 'papers building on attention mechanisms', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'related_search',
        'expected_answer': 'Should find papers that build on or extend attention mechanisms.',
        'expected_papers_min': 1,
        'expected_keywords': ['building', 'attention', 'mechanisms', 'extend']
    },
    {
        'id': 53, 
        'query': 'similar work to tool learning', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'hard', 
        'category': 'related_search',
        'expected_answer': 'Should find work similar to tool learning and agent-based systems.',
        'expected_papers_min': 1,
        'expected_keywords': ['similar', 'tool learning', 'agent', 'systems']
    },
    {
        'id': 54, 
        'query': 'research related to memory systems', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'category': 'related_search',
        'expected_answer': 'Should find research related to memory systems and memory-augmented models.',
        'expected_papers_min': 1,
        'expected_keywords': ['related', 'memory', 'systems', 'augmented']
    },
    {
        'id': 55, 
        'query': 'connected research on graph neural networks', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'hard', 
        'category': 'related_search',
        'expected_answer': 'Should find research connected to graph neural networks and graph-based learning.',
        'expected_papers_min': 1,
        'expected_keywords': ['connected', 'graph', 'neural networks', 'graph-based']
    },

    # comparePapers Route (10 cases)
    {
        'id': 56, 
        'query': 'compare BERT and GPT models', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'category': 'model_comparison',
        'expected_answer': 'Should compare BERT and GPT models, highlighting their differences in architecture, training objectives, and applications.',
        'expected_papers_min': 1,
        'expected_keywords': ['compare', 'BERT', 'GPT', 'models', 'difference']
    },
    {
        'id': 57, 
        'query': 'difference between transformer and CNN', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'category': 'model_comparison',
        'expected_answer': 'Should explain differences between transformer and CNN architectures.',
        'expected_papers_min': 1,
        'expected_keywords': ['difference', 'transformer', 'CNN', 'architecture']
    },
    {
        'id': 58, 
        'query': 'BERT versus RoBERTa comparison', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'category': 'model_comparison',
        'expected_answer': 'Should compare BERT and RoBERTa models, highlighting improvements and differences.',
        'expected_papers_min': 1,
        'expected_keywords': ['BERT', 'RoBERTa', 'comparison', 'versus']
    },
    {
        'id': 59, 
        'query': 'GPT-3 vs GPT-4 analysis', 
        'expected_route': 'comparePapers', 
        'difficulty': 'hard', 
        'category': 'model_comparison',
        'expected_answer': 'Should analyze differences between GPT-3 and GPT-4 models.',
        'expected_papers_min': 0,
        'expected_keywords': ['GPT-3', 'GPT-4', 'analysis', 'comparison']
    },
    {
        'id': 60, 
        'query': 'compare ResNet and DenseNet architectures', 
        'expected_route': 'comparePapers', 
        'difficulty': 'hard', 
        'category': 'model_comparison',
        'expected_answer': 'Should compare ResNet and DenseNet architectures and their characteristics.',
        'expected_papers_min': 0,
        'expected_keywords': ['compare', 'ResNet', 'DenseNet', 'architectures']
    },
    {
        'id': 61, 
        'query': 'LSTM versus transformer comparison', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'category': 'model_comparison',
        'expected_answer': 'Should compare LSTM and transformer architectures for sequence modeling.',
        'expected_papers_min': 1,
        'expected_keywords': ['LSTM', 'transformer', 'comparison', 'sequence']
    },
    {
        'id': 62, 
        'query': 'compare attention mechanisms', 
        'expected_route': 'comparePapers', 
        'difficulty': 'hard', 
        'category': 'method_comparison',
        'expected_answer': 'Should compare different types of attention mechanisms.',
        'expected_papers_min': 1,
        'expected_keywords': ['compare', 'attention', 'mechanisms', 'types']
    },
    {
        'id': 63, 
        'query': 'RAG vs traditional retrieval', 
        'expected_route': 'comparePapers', 
        'difficulty': 'hard', 
        'category': 'method_comparison',
        'expected_answer': 'Should compare RAG with traditional retrieval methods.',
        'expected_papers_min': 1,
        'expected_keywords': ['RAG', 'traditional', 'retrieval', 'compare']
    },
    {
        'id': 64, 
        'query': 'hallucination detection methods comparison', 
        'expected_route': 'comparePapers', 
        'difficulty': 'hard', 
        'category': 'method_comparison',
        'expected_answer': 'Should compare different methods for detecting hallucinations in language models.',
        'expected_papers_min': 1,
        'expected_keywords': ['hallucination', 'detection', 'methods', 'comparison']
    },
    {
        'id': 65, 
        'query': 'supervised vs unsupervised learning comparison', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'category': 'method_comparison',
        'expected_answer': 'Should compare supervised and unsupervised learning approaches.',
        'expected_papers_min': 1,
        'expected_keywords': ['supervised', 'unsupervised', 'learning', 'comparison']
    },

    # trendAnalysis Route (8 cases)
    {
        'id': 66, 
        'query': 'trends in machine learning', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'category': 'field_trends',
        'expected_answer': 'Should analyze trends and developments in machine learning research over time.',
        'expected_papers_min': 2,
        'expected_keywords': ['trends', 'machine learning', 'development', 'evolution']
    },
    {
        'id': 67, 
        'query': 'evolution of language models', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'category': 'field_trends',
        'expected_answer': 'Should trace the evolution of language models from early approaches to modern transformers.',
        'expected_papers_min': 1,
        'expected_keywords': ['evolution', 'language models', 'development', 'progress']
    },
    {
        'id': 68, 
        'query': 'development in transformer architecture', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'category': 'field_trends',
        'expected_answer': 'Should analyze the development and evolution of transformer architectures.',
        'expected_papers_min': 1,
        'expected_keywords': ['development', 'transformer', 'architecture', 'evolution']
    },
    {
        'id': 69, 
        'query': 'progress in hallucination detection', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'hard', 
        'category': 'field_trends',
        'expected_answer': 'Should analyze progress and trends in hallucination detection research.',
        'expected_papers_min': 1,
        'expected_keywords': ['progress', 'hallucination', 'detection', 'trends']
    },
    {
        'id': 70, 
        'query': 'AI research direction analysis', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'hard', 
        'category': 'field_trends',
        'expected_answer': 'Should analyze current and future directions in AI research.',
        'expected_papers_min': 1,
        'expected_keywords': ['AI', 'research', 'direction', 'analysis', 'future']
    },
    {
        'id': 71, 
        'query': 'trends in artificial intelligence research over time', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'category': 'temporal_trends',
        'expected_answer': 'Should analyze trends in AI research over different time periods.',
        'expected_papers_min': 1,
        'expected_keywords': ['trends', 'artificial intelligence', 'research', 'time']
    },
    {
        'id': 72, 
        'query': 'evolution of language models from 2017 to 2024', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'hard', 
        'category': 'temporal_trends',
        'expected_answer': 'Should trace the evolution of language models in the specified time period.',
        'expected_papers_min': 1,
        'expected_keywords': ['evolution', 'language models', '2017', '2024', 'timeline']
    },
    {
        'id': 73, 
        'query': 'development timeline of deep learning', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'category': 'temporal_trends',
        'expected_answer': 'Should provide a timeline of deep learning development and milestones.',
        'expected_papers_min': 1,
        'expected_keywords': ['development', 'timeline', 'deep learning', 'milestones']
    },

    # journalAnalysis Route (5 cases)
    {
        'id': 74, 
        'query': 'best publication venues for machine learning research', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'easy', 
        'category': 'venue_analysis',
        'expected_answer': 'Should recommend top conferences and journals for machine learning research (e.g., NeurIPS, ICML, JMLR).',
        'expected_papers_min': 0,
        'expected_keywords': ['publication', 'venues', 'machine learning', 'conferences', 'journals']
    },
    {
        'id': 75, 
        'query': 'top conferences for AI papers', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'easy', 
        'category': 'venue_analysis',
        'expected_answer': 'Should list top AI conferences like AAAI, IJCAI, NeurIPS, ICML.',
        'expected_papers_min': 0,
        'expected_keywords': ['top', 'conferences', 'AI', 'papers', 'venues']
    },
    {
        'id': 76, 
        'query': 'journal impact factor analysis for NLP', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'medium', 
        'category': 'venue_analysis',
        'expected_answer': 'Should analyze impact factors of NLP journals and conferences.',
        'expected_papers_min': 0,
        'expected_keywords': ['journal', 'impact factor', 'NLP', 'analysis']
    },
    {
        'id': 77, 
        'query': 'publication venue comparison for deep learning', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'medium', 
        'category': 'venue_analysis',
        'expected_answer': 'Should compare different publication venues for deep learning research.',
        'expected_papers_min': 0,
        'expected_keywords': ['publication', 'venue', 'comparison', 'deep learning']
    },
    {
        'id': 78, 
        'query': 'best venues for computer vision research', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'medium', 
        'category': 'venue_analysis',
        'expected_answer': 'Should recommend best venues for computer vision research (e.g., CVPR, ICCV, ECCV).',
        'expected_papers_min': 0,
        'expected_keywords': ['best', 'venues', 'computer vision', 'research', 'conferences']
    },
]

def get_dataset_statistics() -> Dict[str, Any]:
    """Get comprehensive statistics about the QA dataset"""
    total_cases = len(COMPREHENSIVE_QA_DATASET)
    
    # Route distribution
    route_counts = {}
    for case in COMPREHENSIVE_QA_DATASET:
        route = case['expected_route']
        route_counts[route] = route_counts.get(route, 0) + 1
    
    # Difficulty distribution
    difficulty_counts = {}
    for case in COMPREHENSIVE_QA_DATASET:
        difficulty = case['difficulty']
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    return {
        'total_cases': total_cases,
        'route_distribution': route_counts,
        'difficulty_distribution': difficulty_counts,
    }

def get_cases_by_route(route: str) -> List[Dict[str, Any]]:
    """Get all test cases for a specific route"""
    return [case for case in COMPREHENSIVE_QA_DATASET if case['expected_route'] == route]

def get_cases_by_difficulty(difficulty: str) -> List[Dict[str, Any]]:
    """Get all test cases for a specific difficulty level"""
    return [case for case in COMPREHENSIVE_QA_DATASET if case['difficulty'] == difficulty]

def get_expected_answer(test_id: int) -> Dict[str, Any]:
    """Get expected answer information for a specific test case"""
    for case in COMPREHENSIVE_QA_DATASET:
        if case['id'] == test_id:
            return {
                'expected_answer': case['expected_answer'],
                'expected_papers_min': case['expected_papers_min'],
                'expected_keywords': case['expected_keywords']
            }
    return {}

def main():
    """Display QA dataset information"""
    print("📊 Comprehensive QA Dataset Analysis")
    print("=" * 50)
    
    stats = get_dataset_statistics()
    
    print(f"📈 Dataset Overview:")
    print(f"   Total test cases: {stats['total_cases']}")
    print()
    
    print(f"🎯 Route Distribution:")
    for route, count in stats['route_distribution'].items():
        percentage = (count / stats['total_cases']) * 100
        print(f"   {route}: {count} cases ({percentage:.1f}%)")
    print()
    
    print(f"⚡ Difficulty Distribution:")
    for difficulty, count in stats['difficulty_distribution'].items():
        percentage = (count / stats['total_cases']) * 100
        print(f"   {difficulty}: {count} cases ({percentage:.1f}%)")
    
    print(f"\n✅ All test cases now include expected answers for comparison!")

if __name__ == "__main__":
    main() 