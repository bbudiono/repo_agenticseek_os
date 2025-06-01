#!/usr/bin/env python3
"""
Configuration Management System for AgenticSeek
Manages API keys, provider settings, and model configurations
"""

import os
import json
import configparser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import base64
from cryptography.fernet import Fernet

@dataclass
class ProviderConfig:
    name: str
    display_name: str
    model: str
    server_address: str
    is_local: bool
    is_enabled: bool
    api_key_required: bool
    api_key_set: bool = False
    status: str = "unknown"  # "available", "unavailable", "error"
    
    def to_dict(self):
        return asdict(self)

@dataclass
class APIKeyInfo:
    provider: str
    display_name: str
    is_set: bool
    last_updated: Optional[str] = None
    is_valid: Optional[bool] = None
    
    def to_dict(self):
        return asdict(self)

class ConfigManager:
    def __init__(self):
        self.config_file = Path("config.ini")
        self.env_file = Path(".env")
        self.secure_config_file = Path(".agenticseek_secure")
        
        # Generate or load encryption key
        self.key_file = Path(".encryption_key")
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Available providers configuration
        self.available_providers = {
            "lm_studio": {
                "display_name": "LM Studio",
                "api_key_required": False,
                "default_model": "any",
                "default_address": "192.168.1.37:1234"
            },
            "ollama": {
                "display_name": "Ollama",
                "api_key_required": False,
                "default_model": "deepseek-r1:14b",
                "default_address": "127.0.0.1:11434"
            },
            "anthropic": {
                "display_name": "Anthropic (Claude)",
                "api_key_required": True,
                "default_model": "claude-3-5-sonnet-20241022",
                "default_address": "api.anthropic.com"
            },
            "openai": {
                "display_name": "OpenAI",
                "api_key_required": True,
                "default_model": "gpt-4",
                "default_address": "api.openai.com"
            },
            "deepseek": {
                "display_name": "DeepSeek API",
                "api_key_required": True,
                "default_model": "deepseek-chat",
                "default_address": "api.deepseek.com"
            },
            "google": {
                "display_name": "Google (Gemini)",
                "api_key_required": True,
                "default_model": "gemini-pro",
                "default_address": "generativelanguage.googleapis.com"
            }
        }
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get existing or create new encryption key"""
        if self.key_file.exists():
            return self.key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            self.key_file.write_bytes(key)
            # Make key file readable only by owner
            os.chmod(self.key_file, 0o600)
            return key
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration from config.ini"""
        config = configparser.ConfigParser()
        
        if self.config_file.exists():
            config.read(self.config_file)
        
        # Parse current configuration
        current_config = {
            "main_provider": {},
            "fallback_local": {},
            "fallback_api": {},
            "general_settings": {}
        }
        
        # Main provider
        if "MAIN" in config:
            main = config["MAIN"]
            current_config["main_provider"] = {
                "provider_name": main.get("provider_name", "lm_studio"),
                "provider_model": main.get("provider_model", "any"),
                "provider_server_address": main.get("provider_server_address", "192.168.1.37:1234"),
                "is_local": main.getboolean("is_local", True)
            }
        
        # Fallback providers
        if "FALLBACK_LOCAL" in config:
            fallback = config["FALLBACK_LOCAL"]
            current_config["fallback_local"] = {
                "provider_name": fallback.get("provider_name", "ollama"),
                "provider_model": fallback.get("provider_model", "deepseek-r1:14b"),
                "provider_server_address": fallback.get("provider_server_address", "127.0.0.1:11434"),
                "is_local": fallback.getboolean("is_local", True)
            }
        
        if "FALLBACK_API" in config:
            fallback = config["FALLBACK_API"]
            current_config["fallback_api"] = {
                "provider_name": fallback.get("provider_name", "anthropic"),
                "provider_model": fallback.get("provider_model", "claude-3-5-sonnet-20241022"),
                "provider_server_address": fallback.get("provider_server_address", "api.anthropic.com"),
                "is_local": fallback.getboolean("is_local", False)
            }
        
        # General settings
        if "MAIN" in config:
            main = config["MAIN"]
            current_config["general_settings"] = {
                "agent_name": main.get("agent_name", "AgenticSeek_AI"),
                "work_dir": main.get("work_dir", str(Path.home() / "Documents" / "agenticseek_workspace")),
                "save_session": main.getboolean("save_session", False),
                "recover_last_session": main.getboolean("recover_last_session", False)
            }
        
        return current_config
    
    def get_provider_configs(self) -> List[ProviderConfig]:
        """Get list of all provider configurations"""
        current_config = self.get_current_config()
        api_keys = self.get_api_keys_status()
        
        providers = []
        
        for provider_id, provider_info in self.available_providers.items():
            # Find if this provider is currently configured
            is_main = current_config["main_provider"].get("provider_name") == provider_id
            is_fallback_local = current_config["fallback_local"].get("provider_name") == provider_id
            is_fallback_api = current_config["fallback_api"].get("provider_name") == provider_id
            
            # Determine current configuration
            if is_main:
                config_source = current_config["main_provider"]
            elif is_fallback_local:
                config_source = current_config["fallback_local"]
            elif is_fallback_api:
                config_source = current_config["fallback_api"]
            else:
                config_source = {
                    "provider_model": provider_info["default_model"],
                    "provider_server_address": provider_info["default_address"],
                    "is_local": not provider_info["api_key_required"]
                }
            
            # Check if API key is set
            api_key_set = False
            if provider_info["api_key_required"]:
                api_key_set = any(key.provider == provider_id and key.is_set for key in api_keys)
            
            provider_config = ProviderConfig(
                name=provider_id,
                display_name=provider_info["display_name"],
                model=config_source.get("provider_model", provider_info["default_model"]),
                server_address=config_source.get("provider_server_address", provider_info["default_address"]),
                is_local=config_source.get("is_local", not provider_info["api_key_required"]),
                is_enabled=is_main or is_fallback_local or is_fallback_api,
                api_key_required=provider_info["api_key_required"],
                api_key_set=api_key_set
            )
            
            providers.append(provider_config)
        
        return providers
    
    def get_api_keys_status(self) -> List[APIKeyInfo]:
        """Get status of all API keys"""
        api_keys = []
        
        # Load environment variables
        env_vars = {}
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value
        
        # Check secure storage
        secure_keys = {}
        if self.secure_config_file.exists():
            try:
                encrypted_data = self.secure_config_file.read_bytes()
                decrypted_data = self.cipher.decrypt(encrypted_data)
                secure_keys = json.loads(decrypted_data.decode())
            except:
                pass  # Handle corruption gracefully
        
        for provider_id, provider_info in self.available_providers.items():
            if provider_info["api_key_required"]:
                key_var = f"{provider_id.upper()}_API_KEY"
                
                # Check if key is set (from env or secure storage)
                env_key_set = (
                    key_var in env_vars and env_vars[key_var].strip() and 
                    env_vars[key_var] != "your_api_key_here"
                )
                secure_key_set = (
                    provider_id in secure_keys and secure_keys[provider_id].get("api_key")
                )
                is_set = bool(env_key_set or secure_key_set)
                
                last_updated = None
                if provider_id in secure_keys and "last_updated" in secure_keys[provider_id]:
                    last_updated = secure_keys[provider_id]["last_updated"]
                
                api_keys.append(APIKeyInfo(
                    provider=provider_id,
                    display_name=provider_info["display_name"],
                    is_set=is_set,
                    last_updated=last_updated
                ))
        
        return api_keys
    
    def set_api_key(self, provider: str, api_key: str) -> bool:
        """Set API key for a provider (encrypted storage)"""
        try:
            # Load existing secure config
            secure_config = {}
            if self.secure_config_file.exists():
                try:
                    encrypted_data = self.secure_config_file.read_bytes()
                    decrypted_data = self.cipher.decrypt(encrypted_data)
                    secure_config = json.loads(decrypted_data.decode())
                except:
                    pass  # Start fresh if corrupted
            
            # Update API key
            if provider not in secure_config:
                secure_config[provider] = {}
            
            secure_config[provider]["api_key"] = api_key
            secure_config[provider]["last_updated"] = datetime.now().isoformat()
            
            # Encrypt and save
            json_data = json.dumps(secure_config).encode()
            encrypted_data = self.cipher.encrypt(json_data)
            self.secure_config_file.write_bytes(encrypted_data)
            
            # Make secure file readable only by owner
            os.chmod(self.secure_config_file, 0o600)
            
            # Also update .env file for compatibility
            self._update_env_file(f"{provider.upper()}_API_KEY", api_key)
            
            return True
        except Exception as e:
            print(f"Error setting API key for {provider}: {str(e)}")
            return False
    
    def _update_env_file(self, key: str, value: str):
        """Update or add key-value pair in .env file"""
        lines = []
        key_found = False
        
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                lines = f.readlines()
        
        # Update existing key or add new one
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                key_found = True
                break
        
        if not key_found:
            lines.append(f"{key}={value}\n")
        
        with open(self.env_file, 'w') as f:
            f.writelines(lines)
    
    def update_provider_config(self, provider_role: str, provider_name: str, model: str, server_address: str) -> bool:
        """Update provider configuration"""
        try:
            config = configparser.ConfigParser()
            
            if self.config_file.exists():
                config.read(self.config_file)
            
            # Map provider roles to config sections
            section_map = {
                "main": "MAIN",
                "fallback_local": "FALLBACK_LOCAL", 
                "fallback_api": "FALLBACK_API"
            }
            
            section_name = section_map.get(provider_role)
            if not section_name:
                return False
            
            if section_name not in config:
                config.add_section(section_name)
            
            # Update provider settings
            config.set(section_name, "provider_name", provider_name)
            config.set(section_name, "provider_model", model)
            config.set(section_name, "provider_server_address", server_address)
            
            # Set is_local based on provider type
            provider_info = self.available_providers.get(provider_name, {})
            is_local = not provider_info.get("api_key_required", True)
            config.set(section_name, "is_local", str(is_local))
            
            # Write config file
            with open(self.config_file, 'w') as f:
                config.write(f)
            
            return True
        except Exception as e:
            print(f"Error updating provider config: {str(e)}")
            return False
    
    def get_available_models_for_provider(self, provider: str) -> List[str]:
        """Get list of available models for a specific provider"""
        model_catalogs = {
            "ollama": [
                "llama3.2:3b", "deepseek-r1:14b", "qwen2.5:14b", 
                "codellama:13b", "mistral:7b", "phi3:mini"
            ],
            "anthropic": [
                "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229", "claude-3-sonnet-20240229"
            ],
            "openai": [
                "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"
            ],
            "deepseek": [
                "deepseek-chat", "deepseek-coder"
            ],
            "google": [
                "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"
            ],
            "lm_studio": [
                "any", "local-model"
            ]
        }
        
        return model_catalogs.get(provider, ["default"])

# Global config manager instance
config_manager = ConfigManager()

if __name__ == "__main__":
    # Test configuration manager
    print("🧪 Testing Configuration Manager...")
    
    # Test current config
    current = config_manager.get_current_config()
    print(f"📋 Current main provider: {current['main_provider'].get('provider_name', 'None')}")
    
    # Test provider configs
    providers = config_manager.get_provider_configs()
    print(f"🔧 Available providers: {len(providers)}")
    
    # Test API key status
    api_keys = config_manager.get_api_keys_status()
    print(f"🔑 API providers: {len(api_keys)}")
    for key in api_keys:
        print(f"  - {key.display_name}: {'Set' if key.is_set else 'Not set'}")