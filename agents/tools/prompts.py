from pydantic import BaseModel

import yaml

import os
import yaml
from typing import Dict, Any

class PromptLoader(BaseModel):
    """Loads and manages prompt templates."""
    prompt_config_path: str = "agents/config/prompts.yml"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_base_config(self) -> Dict[str, Dict[str, Any]]:
        """Load base configuration from file."""

        prompt_configs = {}
        if self.prompt_config_path and os.path.exists(self.prompt_config_path):
            with open(self.prompt_config_path, 'r') as f:
                prompt_configs = yaml.safe_load(f)

        return prompt_configs

if __name__ == "__main__":
    prompt_loader = PromptLoader(prompt_config_path="agents/config/prompts.yml")
    prompt_configs = prompt_loader._load_base_config()
    print("Prompt Configs:", prompt_configs['ask_rag_agent'])