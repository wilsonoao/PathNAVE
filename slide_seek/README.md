# SlideSeek Local Pipeline

A local re-implementation of the SlideSeek multi-agent pathology diagnosis system from:
> "Evidence-based diagnostic reasoning with multi-agent copilot for human pathology" (2506.20964)

## Architecture

```
User Input (WSI + clinical context)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              Supervisor Agent                        │
│   (ollama qwen3:4b - planning & reasoning)          │
│   - Generates hypotheses                            │
│   - Creates exploration tasks                       │
│   - Synthesizes findings                            │
└──────────────┬──────────────────────────────────────┘
               │  delegates tasks (parallel)
               ▼
┌─────────────────────────────────────────────────────┐
│              Explorer Agents                         │
│   (ollama qwen3:4b - navigation decisions)          │
│   - Navigate WSI regions                            │
│   - Request ROI analysis from PathChat              │
└──────────────┬──────────────────────────────────────┘
               │  ROI images
               ▼
┌─────────────────────────────────────────────────────┐
│         PathChat / patho_r1                          │
│   (HuggingFace - morphology captioning)             │
│   - Analyzes pathology ROI images                   │
│   - Returns morphological descriptions              │
└──────────────┬──────────────────────────────────────┘
               │  morphology descriptions
               ▼
┌─────────────────────────────────────────────────────┐
│           Report Generation                          │
│   (ollama qwen3:4b)                                 │
│   - Differential diagnosis                          │
│   - Structured pathology report                     │
│   - Confidence assessment                           │
└─────────────────────────────────────────────────────┘
```

## Setup

```bash
pip install -r requirements.txt

# Make sure ollama is running with qwen3:4b
ollama pull qwen3:4b

# patho_r1 will be downloaded from HuggingFace on first run
# Model: linjc16/patho_r1 (or configured in config.yaml)
```

## Usage

```python
from slideseek import SlideSeekPipeline

pipeline = SlideSeekPipeline()
result = pipeline.run(
    wsi_path="path/to/slide.svs",
    tissue_site="lung",
    patient_sex="male",
    clinical_context="No additional context"
)
print(result.report)
```

## File Structure

```
slideseek/
├── pipeline.py          # Main pipeline orchestrator
├── agents/
│   ├── supervisor.py    # Supervisor agent
│   └── explorer.py      # Explorer agent
├── models/
│   ├── patho_model.py   # patho_r1 wrapper (HuggingFace)
│   └── llm.py           # Ollama qwen3:4b wrapper
├── wsi/
│   └── slide_viewer.py  # WSI navigation (OpenSlide)
├── config.py            # Configuration
└── utils.py             # Utilities
```
