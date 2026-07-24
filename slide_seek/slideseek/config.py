"""
SlideSeek configuration dataclasses.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpenAILLMConfig:
    model: str = "gpt-4o-mini"
    # If None, falls back to the OPENAI_API_KEY environment variable
    api_key: Optional[str] = None
    # Override for Azure OpenAI or other OpenAI-compatible endpoints
    api_base_url: Optional[str] = None
    temperature: float = 0.1
    max_new_tokens: int = 1024


@dataclass
class OllamaConfig:
    base_url: str = "http://192.168.63.184:11434"
    model: str = "qwen3:4b"
    enable_thinking: bool = False
    temperature: float = 0.1
    timeout: int = 120


@dataclass
class HFLLMConfig:
    model_id: str = "Qwen/Qwen3-4B"
    device: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "bfloat16"
    enable_thinking: bool = False
    temperature: float = 0.1
    max_new_tokens: int = 2048


@dataclass
class HFVisionLLMConfig:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "bfloat16"
    temperature: float = 0.1
    max_new_tokens: int = 2048


@dataclass
class PathoModelConfig:
    model_id: str = "WenchuanZhang/Patho-R1-3B"
    device: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "float16"
    use_flash_attn: bool = True
    max_new_tokens: int = 50
    max_output_words: int = 30


@dataclass
class SlideConfig:
    roi_size: int = 896
    max_rois_per_explorer: int = 3
    max_rois_final: int = 5


@dataclass
class AgentConfig:
    max_supervisor_iterations: int = 10
    max_explorer_iterations: int = 10
    max_parallel_explorers: int = 3
    max_tasks_per_iteration: int = 1


@dataclass
class SlideSeekConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    hf_llm: HFLLMConfig = field(default_factory=HFLLMConfig)
    hf_vision_llm: HFVisionLLMConfig = field(default_factory=HFVisionLLMConfig)
    openai_llm: OpenAILLMConfig = field(default_factory=OpenAILLMConfig)
    # Backend selection (mutually exclusive; openai takes priority)
    use_hf_llm: bool = True
    use_openai_llm: bool = False
    use_vision_supervisor: bool = True
    patho_model: PathoModelConfig = field(default_factory=PathoModelConfig)
    slide: SlideConfig = field(default_factory=SlideConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    output_dir: str = "./results"
    verbose: bool = False


DEFAULT_CONFIG = SlideSeekConfig()
