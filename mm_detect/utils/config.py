"""Configuration utilities for loading environment variables and API keys."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class APIConfig:
    """Configuration class for managing API keys and settings."""
    
    def __init__(self):
        """Initialize API configuration from environment variables."""
        self.load_config()
    
    def load_config(self):
        """Load configuration from environment variables."""
        # OpenAI Configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        
        # Google Gemini Configuration
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        # Anthropic Claude Configuration
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Azure OpenAI Configuration (optional)
        self.azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        
        # HuggingFace Configuration (optional)
        self.huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
        
        # Model Configuration
        self.default_model = os.getenv('DEFAULT_MODEL', 'gpt-4o')
        self.max_tokens = int(os.getenv('MAX_TOKENS', '4096'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        
        # Output Configuration
        self.output_dir = os.getenv('OUTPUT_DIR', './outputs')
        self.results_file = os.getenv('RESULTS_FILE', './outputs/results.json')
        self.enable_resume = os.getenv('ENABLE_RESUME', 'true').lower() == 'true'
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider.
        
        Args:
            provider: The API provider name ('openai', 'gemini', 'anthropic', etc.)
            
        Returns:
            The API key if available, None otherwise.
        """
        provider_lower = provider.lower()
        
        if provider_lower in ['openai', 'gpt']:
            return self.openai_api_key
        elif provider_lower in ['gemini', 'google']:
            return self.gemini_api_key
        elif provider_lower in ['anthropic', 'claude']:
            return self.anthropic_api_key
        elif provider_lower in ['azure', 'azure-openai']:
            return self.azure_openai_api_key
        elif provider_lower in ['huggingface', 'hf']:
            return self.huggingface_token
        else:
            return None
    
    def validate_config(self, provider: str) -> bool:
        """Validate that required configuration for a provider is available.
        
        Args:
            provider: The API provider name
            
        Returns:
            True if configuration is valid, False otherwise.
        """
        api_key = self.get_api_key(provider)
        if not api_key:
            print(f"Warning: No API key found for {provider}. Please check your .env file.")
            return False
        return True
    
    def ensure_output_dir(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)

# Global configuration instance
config = APIConfig()

def get_config() -> APIConfig:
    """Get the global configuration instance."""
    return config