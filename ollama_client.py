#!/usr/bin/env python3
"""
Ollama Client for Tool Learning Chatbot

Integrates with Ollama API to generate intelligent responses
using tool selection and paper search results.
"""

import json
import requests
from typing import Dict, List, Any, Optional, Tuple
import logging
from prompts.base_prompts import PromptTemplates, SystemMessages

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.available = self._check_availability()
    
    def set_model(self, model_name: str):
        self.model = model_name
        
    def _check_availability(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate_response(self, query: str, route: str, route_explanation: str, 
                         paper_results: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate intelligent response using Ollama
        
        Args:
            query: User's original query
            route: Selected tool route
            route_explanation: Why this route was selected
            paper_results: Results from paper search (if applicable)
            
        Returns:
            Generated response string
        """
        if not self.available:
            return self._fallback_response(query, route, route_explanation, paper_results)
        
        # Create context-aware prompt
        prompt = self._build_prompt(query, route, route_explanation, paper_results)
        
        try:
            response = self._call_ollama(prompt)
            return response
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            return self._fallback_response(query, route, route_explanation, paper_results)
    
    def _build_prompt(self, query: str, route: str, route_explanation: str, 
                      paper_results: Optional[List[Dict[str, Any]]]) -> str:
        """Build context-aware prompt for Ollama"""
        return PromptTemplates.build_prompt(query, route, route_explanation, paper_results)
    
    def _call_ollama(self, prompt: str) -> str:
        """Make API call to Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 300
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    
    def _fallback_response(self, query: str, route: str, route_explanation: str, 
                          paper_results: Optional[List[Dict[str, Any]]]) -> str:
        """Generate fallback response when Ollama is not available"""
        return PromptTemplates.build_fallback_response(query, route, route_explanation, paper_results)
    
    def check_model_available(self) -> bool:
        """Check if the currently selected model is available on Ollama"""
        if not self.available:
            return False

        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return any(self.model in model for model in models)

        except Exception as e:
            logger.error(f"Failed to check model: {e}")

        return False
    
def main():
    """Test the Ollama client"""
    print("🤖 Testing Ollama Client")
    print("=" * 30)
    
    client = OllamaClient()
    
    print(f"Ollama available: {client.available}")
    
    if client.available:
        model_available = client.check_model_available()
        print(f"deepseek-r1 model available: {model_available}")
        print(f"Using model: {client.model}")
    
    # Test query
    test_query = "find papers about transformer architecture"
    test_route = "searchPapers"
    test_explanation = "User wants to find research papers"
    
    print(f"\nTest Query: {test_query}")
    print("Generating response...")
    
    response = client.generate_response(
        test_query, 
        test_route, 
        test_explanation,
        paper_results=None
    )
    
    print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    main() 