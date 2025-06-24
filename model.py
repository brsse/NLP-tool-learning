import requests
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages available models and their configurations"""
    
    def __init__(self):
        self.available_models = {
            'llama3.2': {
                'name': 'LLaMA 3.2',
                'description': 'Meta\'s LLaMA 3.2 model - balanced performance',
                'size': '3B parameters',
                'strengths': ['General reasoning', 'Tool selection', 'Text generation'],
                'temperature': 0.1,
                'max_tokens': 1000,
                'context_window': 4096
            },
            'deepseek-r1': {
                'name': 'DeepSeek R1',
                'description': 'DeepSeek reasoning model - strong analytical capabilities',
                'size': '7B parameters',
                'strengths': ['Logical reasoning', 'Analysis', 'Problem solving'],
                'temperature': 0.0,
                'max_tokens': 1500,
                'context_window': 8192
            }
        }
        
        self.default_model = 'llama3.2'
        self.ollama_available = self._check_ollama_connection()
        
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model"""
        return self.available_models.get(model_name)
    
    def get_installed_models(self) -> List[str]:
        """Get list of models actually installed in Ollama"""
        if not self.ollama_available:
            return [self.default_model]  # Fallback to default
        
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=10)
            if response.status_code == 200:
                data = response.json()
                installed = []
                for model in data.get('models', []):
                    model_name = model.get('name', '').split(':')[0]
                    if model_name in self.available_models:
                        installed.append(model_name)
                return installed if installed else [self.default_model]
        except Exception as e:
            logger.warning(f"Failed to get installed models: {e}")
        
        return [self.default_model]
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available and installed"""
        if not self.ollama_available:
            return model_name == self.default_model
        
        installed_models = self.get_installed_models()
        return model_name in installed_models
    
    def generate_with_model(self, prompt: str, model_name: str = None, 
                          temperature: float = None, max_tokens: int = None) -> str:
        """Generate response using specified model"""
        if not self.ollama_available:
            return "Error: Ollama not available"
        
        if model_name is None:
            model_name = self.default_model
        
        config = self.get_model_config(model_name)
        if not config:
            return f"Error: Model {model_name} not found"
        
        # Use provided parameters or defaults from config
        temp = temperature if temperature is not None else config['temperature']
        max_tok = max_tokens if max_tokens is not None else config['max_tokens']
        
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': temp,
                        'num_predict': max_tok
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            logger.warning(f"Generation failed with {model_name}: {e}")
            return f"Error: {str(e)}"
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model availability and status"""
        installed = self.get_installed_models()
        
        return {
            'ollama_available': self.ollama_available,
            'total_configured_models': len(self.available_models),
            'installed_models': installed,
            'installed_count': len(installed)
        }

def main():
    """Test model management functionality"""
    print("🤖 Model Manager Test")
    print("=" * 30)
    
    manager = ModelManager()
    
    # Show model summary
    summary = manager.get_model_summary()
    print(f"📊 Model Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # List available models
    print(f"\n🔧 Available Models:")
    for model_name, config in manager.available_models.items():
        available = "✅" if manager.is_model_available(model_name) else "❌"
        print(f"   {available} {model_name}: {config['description']}")
        print(f"      Size: {config['size']}, Strengths: {', '.join(config['strengths'][:2])}")

if __name__ == "__main__":
    main() 