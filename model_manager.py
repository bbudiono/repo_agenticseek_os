#!/usr/bin/env python3
"""
Model Management System for AgenticSeek
Integrated with cascading provider system
"""

import asyncio
import httpx
import json
import os
import subprocess
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ModelInfo:
    name: str
    provider: str
    size_gb: float
    status: str  # "available", "downloading", "not_downloaded", "error"
    description: str
    tags: List[str]
    last_used: Optional[str] = None
    download_progress: float = 0.0
    file_path: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

class ModelManager:
    def __init__(self):
        self.models_dir = Path.home() / "Documents" / "agenticseek_models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Shared directories for different providers
        self.ollama_models_dir = Path.home() / ".ollama" / "models"
        self.lm_studio_models_dir = Path.home() / ".cache" / "lm-studio" / "models"
        
        self.downloading_models = set()
        
    async def get_available_models_catalog(self) -> Dict[str, List[ModelInfo]]:
        """Get catalog of available models for each provider"""
        catalog = {
            "ollama": await self._get_ollama_models_catalog(),
            "lm_studio": await self._get_lm_studio_models_catalog(),
            "huggingface": await self._get_huggingface_models_catalog()
        }
        return catalog
    
    async def _get_ollama_models_catalog(self) -> List[ModelInfo]:
        """Get Ollama model catalog"""
        popular_models = [
            ModelInfo(
                name="llama3.2:3b",
                provider="ollama",
                size_gb=2.0,
                status="not_downloaded",
                description="Meta's Llama 3.2 3B - Fast and efficient for general tasks",
                tags=["general", "fast", "small"]
            ),
            ModelInfo(
                name="deepseek-r1:14b",
                provider="ollama", 
                size_gb=8.1,
                status="not_downloaded",
                description="DeepSeek R1 14B - Advanced reasoning model",
                tags=["reasoning", "advanced", "medium"]
            ),
            ModelInfo(
                name="qwen2.5:14b",
                provider="ollama",
                size_gb=8.2,
                status="not_downloaded", 
                description="Qwen 2.5 14B - Multilingual capabilities",
                tags=["multilingual", "advanced", "medium"]
            ),
            ModelInfo(
                name="codellama:13b",
                provider="ollama",
                size_gb=7.3,
                status="not_downloaded",
                description="Code Llama 13B - Specialized for code generation",
                tags=["coding", "programming", "medium"]
            ),
            ModelInfo(
                name="mistral:7b",
                provider="ollama",
                size_gb=4.1,
                status="not_downloaded",
                description="Mistral 7B - Balanced performance and efficiency",
                tags=["general", "efficient", "small"]
            )
        ]
        
        # Check which models are actually downloaded
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    installed_models = {model["name"] for model in data.get("models", [])}
                    
                    for model in popular_models:
                        if model.name in installed_models:
                            model.status = "available"
                            model.last_used = "Recently"
        except:
            pass  # Ollama not running, models marked as not_downloaded
            
        return popular_models
    
    async def _get_lm_studio_models_catalog(self) -> List[ModelInfo]:
        """Get LM Studio compatible models catalog"""
        return [
            ModelInfo(
                name="microsoft/DialoGPT-medium",
                provider="lm_studio",
                size_gb=1.2,
                status="not_downloaded",
                description="Dialog GPT Medium - Conversational AI",
                tags=["conversation", "dialog", "small"]
            ),
            ModelInfo(
                name="microsoft/DialoGPT-large",
                provider="lm_studio", 
                size_gb=2.3,
                status="not_downloaded",
                description="Dialog GPT Large - Enhanced conversational AI",
                tags=["conversation", "dialog", "medium"]
            )
        ]
    
    async def _get_huggingface_models_catalog(self) -> List[ModelInfo]:
        """Get Hugging Face models catalog"""
        return [
            ModelInfo(
                name="microsoft/DialoGPT-small",
                provider="huggingface",
                size_gb=0.5,
                status="not_downloaded", 
                description="Dialog GPT Small - Lightweight conversational model",
                tags=["conversation", "lightweight", "small"]
            )
        ]
    
    async def get_installed_models(self) -> List[ModelInfo]:
        """Get list of all installed models across providers"""
        installed = []
        
        # Check Ollama models
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("models", []):
                        installed.append(ModelInfo(
                            name=model["name"],
                            provider="ollama",
                            size_gb=model.get("size", 0) / (1024**3),  # Convert to GB
                            status="available",
                            description=f"Ollama model: {model['name']}",
                            tags=["installed"],
                            last_used=model.get("modified_at", "Unknown")
                        ))
        except:
            pass
            
        # Check LM Studio models (scan directory)
        if self.lm_studio_models_dir.exists():
            for model_path in self.lm_studio_models_dir.rglob("*.gguf"):
                size_gb = model_path.stat().st_size / (1024**3)
                installed.append(ModelInfo(
                    name=model_path.stem,
                    provider="lm_studio",
                    size_gb=size_gb,
                    status="available",
                    description=f"LM Studio GGUF model: {model_path.stem}",
                    tags=["installed", "gguf"],
                    file_path=str(model_path)
                ))
        
        return installed
    
    async def download_model(self, model_name: str, provider: str) -> bool:
        """Download a model for the specified provider"""
        if model_name in self.downloading_models:
            return False
            
        self.downloading_models.add(model_name)
        
        try:
            if provider == "ollama":
                return await self._download_ollama_model(model_name)
            elif provider == "lm_studio":
                return await self._download_lm_studio_model(model_name)
            elif provider == "huggingface":
                return await self._download_huggingface_model(model_name)
            else:
                return False
        finally:
            self.downloading_models.discard(model_name)
    
    async def _download_ollama_model(self, model_name: str) -> bool:
        """Download model via Ollama"""
        try:
            process = await asyncio.create_subprocess_exec(
                "ollama", "pull", model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                print(f"✅ Successfully downloaded {model_name}")
                return True
            else:
                print(f"❌ Failed to download {model_name}: {stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {str(e)}")
            return False
    
    async def _download_lm_studio_model(self, model_name: str) -> bool:
        """Download model for LM Studio (placeholder - requires LM Studio API)"""
        # LM Studio doesn't have a direct API for downloading
        # This would need to be implemented based on LM Studio's capabilities
        print(f"📋 LM Studio model download for {model_name} requires manual installation")
        return False
    
    async def _download_huggingface_model(self, model_name: str) -> bool:
        """Download model from Hugging Face"""
        try:
            # This would use huggingface_hub to download models
            print(f"📋 Hugging Face model download for {model_name} not implemented yet")
            return False
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {str(e)}")
            return False
    
    async def delete_model(self, model_name: str, provider: str) -> bool:
        """Delete a model"""
        try:
            if provider == "ollama":
                process = await asyncio.create_subprocess_exec(
                    "ollama", "rm", model_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                return process.returncode == 0
            else:
                print(f"Delete not implemented for {provider}")
                return False
        except Exception as e:
            print(f"❌ Error deleting {model_name}: {str(e)}")
            return False
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information for models"""
        try:
            disk_usage = psutil.disk_usage(str(self.models_dir.parent))
            
            total_gb = disk_usage.total / (1024**3)
            used_gb = disk_usage.used / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            
            # Calculate model storage usage
            model_usage = 0
            if self.ollama_models_dir.exists():
                model_usage += sum(f.stat().st_size for f in self.ollama_models_dir.rglob("*") if f.is_file())
            if self.lm_studio_models_dir.exists():
                model_usage += sum(f.stat().st_size for f in self.lm_studio_models_dir.rglob("*") if f.is_file())
            
            model_usage_gb = model_usage / (1024**3)
            
            return {
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2), 
                "free_gb": round(free_gb, 2),
                "model_usage_gb": round(model_usage_gb, 2),
                "usage_percentage": round((used_gb / total_gb) * 100, 1)
            }
        except Exception as e:
            return {
                "error": str(e),
                "total_gb": 0,
                "used_gb": 0,
                "free_gb": 0,
                "model_usage_gb": 0,
                "usage_percentage": 0
            }

# Global model manager instance
model_manager = ModelManager()

if __name__ == "__main__":
    async def test_model_manager():
        print("🧪 Testing Model Manager...")
        
        # Test catalog
        catalog = await model_manager.get_available_models_catalog()
        print(f"📋 Available models: {len(catalog['ollama'])} Ollama, {len(catalog['lm_studio'])} LM Studio")
        
        # Test installed models
        installed = await model_manager.get_installed_models()
        print(f"💾 Installed models: {len(installed)}")
        
        # Test storage info
        storage = model_manager.get_storage_info()
        print(f"💽 Storage: {storage['free_gb']}GB free of {storage['total_gb']}GB total")
        
    asyncio.run(test_model_manager())