#!/usr/bin/env python3
"""
Simple QA Dataset for Tool Learning System
Realistic expected answers based on actual arxiv dataset content.
"""

from typing import Dict, List, Any

# ============================================================================
# SIMPLE QA DATASET WITH REALISTIC ANSWERS
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
    {
        'id': 6, 
        'query': 'foundation models for vision tasks', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Found foundation model papers like "Segment Anything" by Kirillov et al. (2023), "Florence: A New Foundation Model for Computer Vision" by Yuan et al. (2021), and "InternImage" by Wang et al. (2022).',
        'expected_papers_min': 2,
        'expected_keywords': ['vision', 'image', 'Florence', 'foundation models']
    },
    {
        'id': 7, 
        'query': 'find papers on deep neural networks', 
        'expected_route': 'searchPapers', 
        'difficulty': 'easy', 
        'expected_answer': 'Found deep neural network papers like "Deep Residual Learning for Image Recognition" by He et al. (2015) and "Adam: A Method for Stochastic Optimization" by Kingma & Ba (2014).',
        'expected_papers_min': 2,
        'expected_keywords': ['deep neural network', 'ResNet', 'Adam optimizer']
    },
    {
        'id': 8, 
        'query': 'surveys on self-supervised learning', 
        'expected_route': 'searchPapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Found papers on self-supervised learning like "A Survey of Self‑Supervised Learning: Algorithms, Applications, and Trends" by Gui et al. (2023).',
        'expected_papers_min': 2,
        'expected_keywords': ['self-supervised', 'unsupervised learning', 'SimCLR', 'MoCo']
    },
    {
        'id': 9, 
        'query': 'what are some papers on explainable AI', 
        'expected_route': 'searchPapers', 
        'difficulty': 'hard', 
        'expected_answer': 'Found papers on explainable AI like "Explainable Artificial Intelligence: A Survey of Needs, Techniques, Applications, and Future Direction" by Mersha et al. (2024) and "Concept-based Explainable Artificial Intelligence: A Survey" by Poeta et al. (2023).',
        'expected_papers_min': 2,
        'expected_keywords': ['XAI', 'interpretability', 'model explanation', 'black box', 'transparency']
    },

    # getAuthorInfo Route  
    {
        'id': 10, 
        'query': 'who is Elad Hazan', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'expected_answer': 'Elad Hazan is a professor of Computer Science at Princeton University and author of "Lecture Notes: Optimization for Machine Learning" (2019). His research focuses on machine learning and mathematical optimization.',
        'expected_papers_min': 1,
        'expected_keywords': ['Elad Hazan', 'optimization', 'Princeton', 'machine learning']
    },
    {
        'id': 11, 
        'query': 'research by Wei-Hung Weng', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'expected_answer': 'Wei-Hung Weng authored "Machine Learning for Clinical Predictive Analytics" and "Representation Learning for Electronic Health Records". His research focuses on clinical machine learning applications.',
        'expected_papers_min': 1,
        'expected_keywords': ['Wei-Hung Weng', 'clinical', 'healthcare', 'machine learning']
    },
    {
        'id': 12, 
        'query': 'papers by Xiaojin Zhu', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'expected_answer': 'Xiaojin Zhu authored "An Optimal Control View of Adversarial Machine Learning" (2018), exploring adversarial ML through optimal control and reinforcement learning perspectives.',
        'expected_papers_min': 1,
        'expected_keywords': ['Xiaojin Zhu', 'adversarial', 'optimal control', 'reinforcement learning']
    },
    {
        'id': 13, 
        'query': 'papers by Qianru Sun', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'expected_answer': 'Qianru Sun authored "A Domain Based Approach to Social Relation Recognition" (2017) and "Meta-Transfer Learning for Few-Shot Learning" (2019). Her key research areas are computer vision, domain adaptation and diffusion models.',
        'expected_papers_min': 1,
        'expected_keywords': ['Qianru Sun', 'computer vision', 'few-shot', 'recognition']
    },
    {
        'id': 14, 
        'query': 'papers by Yang Deng', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'expected_answer': 'Yang Deng authored "Proactive Conversational AI: A Comprehensive Survey of Advancements and Opportunities" and "On the Multi-turn Instruction Following for Conversational Web Agents". His research focuses on natural language processing, information retrieval and large language models.',
        'expected_papers_min': 1,
        'expected_keywords': ['Yang Deng', 'natural language processing', 'information retrieval', 'large language models']
    },
    {
        'id': 15, 
        'query': 'papers by Bingtian Dai', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'medium', 
        'expected_answer': 'Bingtian Dai authored "MWPToolkit: An Open‑Source Framework for Deep Learning‑Based Math Word Problem Solvers" (2021). He focuses on math word problem solving and neural network classification.',
        'expected_papers_min': 1,
        'expected_keywords': ['Bingtian Dai', 'neural network', 'math word problem', 'applied machine learning']
    },
{
        'id': 16, 
        'query': 'papers by Fei‑Fei Li', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'easy', 
        'expected_answer': 'Fei‑Fei Li authored "ImageNet: A large-scale hierarchical image database" (2009), a foundational dataset paper for computer vision training and testing. Her research focuses on large-scale computer vision and image recognition.',
        'expected_papers_min': 1,
        'expected_keywords': ['Fei‑Fei Li', 'computer vision', 'ImageNet', 'image recognition']
    },
{
        'id': 17, 
        'query': 'papers by Ian Goodfellow', 
        'expected_route': 'getAuthorInfo', 
        'difficulty': 'easy', 
        'expected_answer': 'Ian Goodfellow authored "Generative Adversarial Networks" and "Adversarial Machine Learning at Scale". His research focuses on adversarial networks, machine learning and semi-supervised learning.',
        'expected_papers_min': 1,
        'expected_keywords': ['Ian Goodfellow', 'GAN', 'semi-supervised learning', 'adversarial system']
    },

    # getCitations Route
    {
        'id': 18, 
        'query': 'citation analysis for machine learning papers', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'expected_answer': 'Citation analysis shows high impact papers include optimization surveys, healthcare applications, and quantum ML reviews with significant academic influence.',
        'expected_papers_min': 1,
        'expected_keywords': ['citation', 'analysis', 'impact', 'machine learning']
    },
    {
        'id': 19, 
        'query': 'impact of automated machine learning research', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'expected_answer': 'AutoML research has significant impact with frameworks like AutoCompete and comprehensive surveys driving adoption in industry and academia.',
        'expected_papers_min': 1,
        'expected_keywords': ['impact', 'automated machine learning', 'AutoML', 'research']
    },
    {
        'id': 20, 
        'query': 'how influential is the ResNet paper', 
        'expected_route': 'getCitations', 
        'difficulty': 'easy', 
        'expected_answer': 'The ResNet paper by He et al. (2015) has over 200,000 citations and significantly impacted deep learning in computer vision.',
        'expected_papers_min': 1,
        'expected_keywords': ['ResNet', 'impact', 'deep learning', 'computer vision']
    },
    {
        'id': 21, 
        'query': 'what is the most cited paper authored by Geoffrey Hinton', 
        'expected_route': 'getCitations', 
        'difficulty': 'easy', 
        'expected_answer': 'Geoffrey Hinton\'s most cited paper is "Imagenet classification with deep convolutional neural networks" (2012), with over 100,000 citations.',
        'expected_papers_min': 1,
        'expected_keywords': ['ImageNet', 'deep convolutional neural networks', 'citation']
    },
    {
        'id': 22, 
        'query': 'what are the most cited NeurIPS 2017 papers', 
        'expected_route': 'getCitations', 
        'difficulty': 'medium', 
        'expected_answer': 'One of the most cited NeurIPS 2017 papers is "Attention Is All You Need" by Vaswani et al., which introduced the Transformer architecture.',
        'expected_papers_min': 1,
        'expected_keywords': ['NeurIPS', 'transformer', 'attention', 'citation']
    },

    # getRelatedPapers Route
    {
        'id': 23, 
        'query': 'papers related to machine learning optimization', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Related optimization papers include surveys on optimization methods, Bayesian optimization guides, and gradient-based learning techniques for ML.',
        'expected_papers_min': 3,
        'expected_keywords': ['related', 'optimization', 'machine learning', 'gradient']
    },
    {
        'id': 24, 
        'query': 'similar research to healthcare ML', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Related healthcare research includes clinical predictive analytics, EHR representation learning, and probabilistic ML for medical applications.',
        'expected_papers_min': 2,
        'expected_keywords': ['similar', 'healthcare', 'clinical', 'medical ML']
    },
    {
        'id': 25, 
        'query': 'research similar to LoRA', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Research similar to LoRA includes BitFit, IA3, UniPELT and other parameter-efficient fine-tuning methods.',
        'expected_papers_min': 2,
        'expected_keywords': ['similar', 'parameter-efficient', 'fine-tuning', 'adapter-based tuning']
    },
    {
        'id': 26, 
        'query': 'papers related to reinforcement learning', 
        'expected_route': 'getRelatedPapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Research similar to reinforcement learning includes imitation learning, reward shaping, bandit algorithms and RLHF.',
        'expected_papers_min': 2,
        'expected_keywords': ['related', 'imitation learning', 'bandit algorithm', 'RLHF']
    },

    # comparePapers Route
    {
        'id': 27, 
        'query': 'compare supervised and unsupervised learning', 
        'expected_route': 'comparePapers', 
        'difficulty': 'easy', 
        'expected_answer': 'Supervised learning uses labeled data for prediction while unsupervised learning finds patterns in unlabeled data. They have different applications and evaluation methods.',
        'expected_papers_min': 2,
        'expected_keywords': ['labeled', 'supervised', 'unsupervised', 'learning']
    },
    {
        'id': 28, 
        'query': 'classical ML versus deep learning approaches', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Classical ML uses handcrafted features while deep learning learns representations automatically. Deep learning excels with large data, classical ML better for small datasets.',
        'expected_papers_min': 2,
        'expected_keywords': ['classical ML', 'deep learning', 'comparison', 'features']
    },
    {
        'id': 29, 
        'query': 'what is the difference between online and offline RL', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Online RL learns by interacting with the environment during training, which balances exploration and exploitation. Offline RL relies on a static dataset collected ahead of time, making it safer but more susceptible to distributional shift.',
        'expected_papers_min': 2,
        'expected_keywords': ['reinforcement learning', 'difference', 'exploration', 'interaction']
    },
    {
        'id': 30, 
        'query': 'compare BERT and GPT', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'expected_answer': 'BERT is an encoder-only transformer that uses masked language modeling to learn bidirectional representations from unlabeled text. GPT is a decoder-only transformer that uses causal language modeling for next token prediction.',
        'expected_papers_min': 2,
        'expected_keywords': ['transformer', 'encoder', 'decoder', 'language modeling']
    },
    {
        'id': 31, 
        'query': 'compare LIME and SHAP', 
        'expected_route': 'comparePapers', 
        'difficulty': 'medium', 
        'expected_answer': 'LIME explains model predictions by training local surrogate models around individual instances. SHAP uses Shapley values from game theory to assign consistent and theoretically grounded feature attributions, and is slower to compute than LIME.',
        'expected_papers_min': 2,
        'expected_keywords': ['XAI', 'feature attribution', 'Shapley values', 'local explanation']
    },

    # trendAnalysis Route
    {
        'id': 32, 
        'query': 'trends in machine learning research', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'expected_answer': 'Current trends include quantum ML, automated ML, healthcare applications, and interpretable AI. Growing focus on safety and ethical considerations.',
        'expected_papers_min': 3,
        'expected_keywords': ['trends', 'machine learning', 'quantum', 'automated']
    },
    {
        'id': 33, 
        'query': 'evolution of optimization algorithms', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'expected_answer': 'Optimization evolved from simple gradient descent to advanced methods like Adam, Bayesian optimization, and quantum-inspired algorithms for ML.',
        'expected_papers_min': 2,
        'expected_keywords': ['evolution', 'optimization', 'algorithms', 'gradient descent']
    },
    {
        'id': 34,
        'query': 'recent trends in graph neural networks', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'expected_answer': 'Graph neural networks have evolved from basic message-passing models like GCNs and GATs to scalable architectures like GraphSAINT and Graphormer. Recent work focuses on GNNs for molecules and proteins, and integrating attention mechanisms.',
        'expected_papers_min': 2,
        'expected_keywords': ['evolution', 'scalable architecture', 'integration', 'graph neural networks']
    },
    {
        'id': 35,
        'query': 'evolution of fairness and bias mitigation methods', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'expected_answer': 'Fairness research has evolved from early methods like re-weighting and pre-processing to in-processing methods, such as adversarial debiasing and fair representations, and post-processing methods, like equalized odds.',
        'expected_papers_min': 2,
        'expected_keywords': ['post-processing', 'in-processing', 'bias mitigation', 'fairness', 'evolution']
    },
    {
        'id': 36,
        'query': 'how has prompt engineering grown over time', 
        'expected_route': 'trendAnalysis', 
        'difficulty': 'medium', 
        'expected_answer': 'Prompt engineering has evolved from manual prompt crafting in GPT-3 to advanced techniques like prompt tuning, soft prompts, instruction tuning, and auto-prompt generation.',
        'expected_papers_min': 2,
        'expected_keywords': ['LLM', 'instruction tuning', 'auto-prompt generation', 'prompt crafting', 'GPT']
    },

    # journalAnalysis Route
    {
        'id': 37, 
        'query': 'best venues for machine learning research', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'medium', 
        'expected_answer': 'Top ML venues include journals like JMLR and TPAMI, and conferences like ICML, ICLR, NeurIPS, and other specialized conferences for domains like healthcare AI and quantum computing.',
        'expected_papers_min': 1,
        'expected_keywords': ['journals', 'JMLR', 'ICML', 'conferences']
    },
    {
        'id': 38, 
        'query': 'publication patterns in AI research', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'medium', 
        'expected_answer': 'AI research shows increasing publication volume with arXiv preprints becoming standard. Growing specialization in subfields like quantum ML and healthcare.',
        'expected_papers_min': 1,
        'expected_keywords': ['publication', 'patterns', 'AI research', 'arXiv']
    },
    {
        'id': 39, 
        'query': 'best journals for computer vision papers', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'easy', 
        'expected_answer': 'Top journals for computer vision include IJCV, TPAMI and CVIU. These journals regularly publish foundational and state-of-the-art vision models including work on CNNs, ViTs, and generative models.',
        'expected_papers_min': 1,
        'expected_keywords': ['journals', 'computer vision', 'image understanding', 'AI']
    },
    {
        'id': 40, 
        'query': 'cross-disciplinary ML paper publications', 
        'expected_route': 'journalAnalysis', 
        'difficulty': 'medium', 
        'expected_answer': 'Cross-disciplinary ML research is frequently published in journals like Big Data & Society, Computational Linguistics (MIT Press), IEEE Journal of Biomedical and Health Informatics, and npj Digital Medicine.',
        'expected_papers_min': 1,
        'expected_keywords': ['cross-disciplinary', 'social science', 'digital medicine', 'computational linguistics']
    },

    # Multi-Route Cases
    {
        'id': 41, 
        'query': 'find papers by Elad Hazan on optimization', 
        'expected_route': 'searchPapers, getAuthorInfo', 
        'difficulty': 'medium', 
        'expected_answer': 'Elad Hazan authored "Lecture Notes: Optimization for Machine Learning" (2019). The paper covers optimization fundamentals for ML derived from Princeton University courses and tutorials.',
        'expected_papers_min': 1,
        'expected_keywords': ['Elad Hazan', 'optimization', 'Princeton', 'lecture notes']
    },
    {
        'id': 42, 
        'query': 'compare quantum ML approaches and analyze trends', 
        'expected_route': 'comparePapers, trendAnalysis', 
        'difficulty': 'hard', 
        'expected_answer': 'Quantum ML includes NISQ and fault-tolerant approaches. Trends show growing interest in quantum advantages for specific ML tasks, with challenges in near-term implementations.',
        'expected_papers_min': 2,
        'expected_keywords': ['quantum ML', 'NISQ', 'fault-tolerant', 'trends']
    },
    {
        'id': 43, 
        'query': 'top researchers publishing in prompt engineering', 
        'expected_route': 'getAuthorInfo, trendAnalysis', 
        'difficulty': 'hard', 
        'expected_answer': 'Notable researchers in prompt engineering include Jason Wei (PromptSource, Chain-of-Thought prompting), Suchin Gururangan (instruction tuning), and Yao Fu (auto-prompting).',
        'expected_papers_min': 2,
        'expected_keywords': ['prompt engineering', 'instruction tuning', 'auto-prompting', 'chain-of-thought']
    },
    {
        'id': 44, 
        'query': 'areas of focus in fairness and bias mitigation research journals', 
        'expected_route': 'comparePapers, journalAnalysis', 
        'difficulty': 'hard', 
        'expected_answer': 'Fairness research appears in journals like Nature Machine Intelligence and ACM FAccT. ACM FAccT focuses on sociotechnical fairness and system-level bias mitigation, while Nature MI leans toward ML systems and evaluation metrics.',
        'expected_papers_min': 2,
        'expected_keywords': ['fairness', 'journals', 'evaluation metrics', 'comparison', 'bias mitigation']
    },
    {
        'id': 45, 
        'query': 'recent papers on using LoRA for NLP and other related methods', 
        'expected_route': 'searchPapers, getRelatedPapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Recent NLP papers using LoRA include "QLoRA: Efficient Finetuning of Quantized LLMs" by Dettmers et al. (2023) and "LoRA: Low-Rank Adaptation of Large Language Models" by Hu et al. (2021). Related methods include BitFit, IA3, and AdapterFusion.',
        'expected_papers_min': 2,
        'expected_keywords': ['parameter-efficient', 'fine-tuning', 'NLP', 'adapter']
    },
    {
        'id': 46, 
        'query': 'papers on using GNNs and other related graph-based methods', 
        'expected_route': 'searchPapers, getRelatedPapers', 
        'difficulty': 'medium', 
        'expected_answer': 'Papers on GNNs include "Neural Message Passing for Quantum Chemistry" by Gilmer et al. (2017) and "Geom-GCN: Geometric Graph Neural Networks" by Pei et al. (2020). Related topics include Graph Transformers, spectral GNNs, and attention-based models like Graphormer.',
        'expected_papers_min': 2,
        'expected_keywords': ['graphs', 'GNN', 'geometric GNN', 'message passing']
    },
    {
        'id': 47, 
        'query': 'compare papers on LoRA and BitFit for NLP', 
        'expected_route': 'searchPapers, comparePapers', 
        'difficulty': 'medium', 
        'expected_answer': '"LoRA: Low-Rank Adaptation of LLMs" by Hu et al. (2021) introduces low-rank matrices injected into attention layers to fine-tune LLMs efficiently, which allows greater flexibility. "BitFit: Simple Parameter-Efficient Fine-Tuning" by Zaken et al. (2021) only updates bias terms in the model, making it simpler and faster.',
        'expected_papers_min': 2,
        'expected_keywords': ['LoRA', 'BitFit', 'parameter-efficient', 'fine-tuning', 'attention layers']
    },
    {
        'id': 48, 
        'query': 'evolution of RL over time in journals', 
        'expected_route': 'journalAnalysis, trendAnalysis', 
        'difficulty': 'medium', 
        'expected_answer': 'Reinforcement learning research has evolved from tabular and policy-gradient methods to deep RL, offline RL, and RLHF. It is commonly published in journals such as JMLR and IEEE Transactions on Neural Networks and Learning Systems.',
        'expected_papers_min': 2,
        'expected_keywords': ['reinforcement learning', 'journals', 'evolution', 'RLHF', 'deep RL']
    },
    {
        'id': 49, 
        'query': 'what is the difference between BERT and GPT and which one is more popular', 
        'expected_route': 'getCitations, comparePapers', 
        'difficulty': 'medium', 
        'expected_answer': 'BERT and GPT differ in architecture and use: BERT is an encoder-only transformer trained with masked language modeling, while GPT is a decoder-only model trained with autoregressive objectives. BERT (Devlin et al., 2018) has more citations than GPT-2 or GPT-3, making it more popular academically, but GPT models dominate generative AI applications.',
        'expected_papers_min': 2,
        'expected_keywords': ['BERT', 'GPT', 'encoder-only', 'decoder-only', 'difference', 'popularity']
    },
    {
        'id': 50, 
        'query': 'most cited recent works on XAI', 
        'expected_route': 'searchPapers, getCitations', 
        'difficulty': 'medium', 
        'expected_answer': 'Some of the most cited recent works on explainable AI (XAI) include "Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI)" by Adadi & Berrada (2018) and "Explainable Artificial Intelligence: Understanding, Visualizing and Interpreting Deep Learning Models" by Samek et al. (2017).',
        'expected_papers_min': 2,
        'expected_keywords': ['explainable AI', 'citations', 'deep learning models', 'black box']
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