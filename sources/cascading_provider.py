"""
Cascading LLM Provider System
Implements tiered fallback: LM Studio -> Local Ollama -> API (Claude)
"""

import configparser
import os
import asyncio
import concurrent.futures
from typing import Optional, Dict, Any, List
from sources.llm_provider import Provider
from sources.logger import Logger
from sources.utility import pretty_print

class CascadingProvider:
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self.logger = Logger("cascading_provider.log")
        self.providers = []
        self.current_provider_index = 0
        self.shared_models_dir = os.path.expanduser("~/Documents/agenticseek_models")
        
        # Ensure shared models directory exists
        os.makedirs(self.shared_models_dir, exist_ok=True)
        
        self._load_providers()
    
    def _load_providers(self):
        """Load all providers from config in fallback order"""
        config = configparser.ConfigParser()
        config.read(self.config_file)
        
        # Provider configurations in order of preference
        provider_configs = [
            ("MAIN", "Primary LM Studio"),
            ("FALLBACK_LOCAL", "Local Ollama"), 
            ("FALLBACK_API", "API (Claude)")
        ]
        
        for section_name, description in provider_configs:
            if section_name in config:
                section = config[section_name]
                try:
                    provider_config = {
                        'name': section.get('provider_name'),
                        'model': section.get('provider_model'),
                        'address': section.get('provider_server_address'),
                        'is_local': section.getboolean('is_local', True),
                        'description': description
                    }
                    self.providers.append(provider_config)
                    pretty_print(f"✅ Loaded {description}: {provider_config['name']}", color="success")
                except Exception as e:
                    self.logger.error(f"Failed to load {section_name}: {str(e)}")
                    pretty_print(f"❌ Failed to load {description}: {str(e)}", color="failure")
    
    def _test_provider_connection(self, provider_config: Dict[str, Any]) -> bool:
        """Test if a provider is available"""
        try:
            provider = Provider(
                provider_config['name'],
                provider_config['model'], 
                provider_config['address'],
                provider_config['is_local']
            )
            
            # Simple health check with minimal message
            test_history = [{"role": "user", "content": "test"}]
            
            if provider_config['name'] == 'lm_studio':
                # Check if LM Studio server is responding
                import requests
                response = requests.get(f"http://{provider_config['address']}/v1/models", timeout=5)
                return response.status_code == 200
            elif provider_config['name'] == 'ollama':
                # Check if Ollama is running
                return provider.is_ip_online(provider_config['address'])
            elif provider_config['name'] == 'anthropic':
                # Check if API key is available
                return os.getenv('ANTHROPIC_API_KEY') is not None
            
        except Exception as e:
            self.logger.warning(f"Provider {provider_config['description']} failed health check: {str(e)}")
            return False
        
        return True
    
    async def _test_provider_connection_async(self, provider_config: Dict[str, Any]) -> bool:
        """Async version of provider connection test"""
        try:
            if provider_config['name'] == 'lm_studio':
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://{provider_config['address']}/v1/models", timeout=5)
                    return response.status_code == 200
            elif provider_config['name'] == 'ollama':
                # Run sync test in executor
                loop = asyncio.get_event_loop()
                provider = Provider(
                    provider_config['name'],
                    provider_config['model'], 
                    provider_config['address'],
                    provider_config['is_local']
                )
                return await loop.run_in_executor(None, provider.is_ip_online, provider_config['address'])
            elif provider_config['name'] == 'anthropic':
                return os.getenv('ANTHROPIC_API_KEY') is not None
        except Exception as e:
            self.logger.warning(f"Provider {provider_config['description']} failed async health check: {str(e)}")
            return False
        return False

    async def check_all_providers_parallel(self) -> List[Dict[str, Any]]:
        """Check all providers in parallel"""
        async def check_provider(provider_config):
            is_available = await self._test_provider_connection_async(provider_config)
            return {
                'config': provider_config,
                'available': is_available,
                'index': self.providers.index(provider_config)
            }
        
        # Run all checks in parallel
        tasks = [check_provider(config) for config in self.providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        provider_status = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Provider check failed: {str(result)}")
                continue
            provider_status.append(result)
        
        return provider_status

    def get_available_provider(self) -> Optional[Provider]:
        """Get the first available provider in fallback order"""
        for i, provider_config in enumerate(self.providers):
            pretty_print(f"🔍 Testing {provider_config['description']}...", color="info")
            
            if self._test_provider_connection(provider_config):
                pretty_print(f"✅ Using {provider_config['description']}", color="success")
                self.current_provider_index = i
                
                return Provider(
                    provider_config['name'],
                    provider_config['model'],
                    provider_config['address'], 
                    provider_config['is_local']
                )
            else:
                pretty_print(f"❌ {provider_config['description']} unavailable, trying next...", color="warning")
        
        pretty_print("🚨 No providers available!", color="failure")
        return None

    async def get_available_provider_async(self) -> Optional[Provider]:
        """Async version that checks all providers in parallel"""
        pretty_print("🔍 Checking all providers in parallel...", color="info")
        
        provider_statuses = await self.check_all_providers_parallel()
        
        # Sort by original order and find first available
        provider_statuses.sort(key=lambda x: x['index'])
        
        for status in provider_statuses:
            if status['available']:
                config = status['config']
                pretty_print(f"✅ Using {config['description']}", color="success")
                self.current_provider_index = status['index']
                
                return Provider(
                    config['name'],
                    config['model'],
                    config['address'], 
                    config['is_local']
                )
            else:
                pretty_print(f"❌ {status['config']['description']} unavailable", color="warning")
        
        pretty_print("🚨 No providers available!", color="failure")
        return None
    
    def respond_with_fallback(self, history, verbose=True) -> str:
        """Try to get response with automatic fallback"""
        for i, provider_config in enumerate(self.providers):
            try:
                pretty_print(f"🤖 Attempting response with {provider_config['description']}", color="info")
                
                provider = Provider(
                    provider_config['name'],
                    provider_config['model'],
                    provider_config['address'],
                    provider_config['is_local']
                )
                
                response = provider.respond(history, verbose)
                
                if response and "REQUEST_EXIT" not in response:
                    pretty_print(f"✅ Success with {provider_config['description']}", color="success")
                    self.logger.info(f"Successful response from {provider_config['description']}")
                    return response
                    
            except Exception as e:
                error_msg = str(e)
                pretty_print(f"❌ {provider_config['description']} failed: {error_msg}", color="failure")
                self.logger.error(f"Provider {provider_config['description']} failed: {error_msg}")
                
                # If this is the last provider, re-raise the exception
                if i == len(self.providers) - 1:
                    raise Exception(f"All providers failed. Last error: {error_msg}")
                    
                pretty_print(f"🔄 Falling back to next provider...", color="info")
                continue
        
        return "All LLM providers failed. Please check your configuration."
    
    def get_current_provider_info(self) -> Dict[str, str]:
        """Get information about the currently active provider"""
        if self.current_provider_index < len(self.providers):
            config = self.providers[self.current_provider_index]
            return {
                "name": config['name'],
                "model": config['model'], 
                "description": config['description'],
                "is_local": str(config['is_local'])
            }
        return {"name": "none", "model": "none", "description": "No provider active", "is_local": "false"}
    
    def setup_shared_models(self):
        """Setup shared models directory for Ollama and LM Studio"""
        pretty_print(f"📁 Shared models directory: {self.shared_models_dir}", color="info")
        
        # Create symlinks or setup shared directory
        ollama_models_dir = os.path.expanduser("~/.ollama/models")
        lm_studio_models_dir = os.path.expanduser("~/.cache/lm-studio/models")
        
        try:
            if os.path.exists(ollama_models_dir) and not os.path.islink(ollama_models_dir):
                pretty_print("💡 Consider linking Ollama models to shared directory to save space", color="info")
            
            if os.path.exists(lm_studio_models_dir) and not os.path.islink(lm_studio_models_dir):
                pretty_print("💡 Consider linking LM Studio models to shared directory to save space", color="info")
                
        except Exception as e:
            self.logger.warning(f"Could not check model directories: {str(e)}")


if __name__ == "__main__":
    # Test the cascading provider
    provider = CascadingProvider()
    provider.setup_shared_models()
    
    available = provider.get_available_provider()
    if available:
        print(f"Available provider: {provider.get_current_provider_info()}")
        
        # Test response
        test_history = [{"role": "user", "content": "Hello, which LLM provider are you?"}]
        response = provider.respond_with_fallback(test_history)
        print(f"Response: {response}")
    else:
        print("No providers available")