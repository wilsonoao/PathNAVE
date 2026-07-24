#!/usr/bin/env python3
"""
SlideSeek QA Evaluation — run inference on roi_qa_dataset.json

Examples
--------
# Full run
python run_qa_eval.py \\
    --json  roi_qa_dataset.json \\
    --slides /path/to/tif/dir \\
    --output ./qa_results

# Single slide (for debugging)
python run_qa_eval.py \\
    --json  roi_qa_dataset.json \\
    --slides /path/to/tif/dir \\
    --output ./qa_results \\
    --slide-id test_069

# Mock run (no real WSI — smoke-tests the whole pipeline)
python run_qa_eval.py \\
    --json  roi_qa_dataset.json \\
    --slides /path/to/tif/dir \\
    --output ./qa_results \\
    --mock

# Low-VRAM, 4-bit patho model
python run_qa_eval.py \\
    --json  roi_qa_dataset.json \\
    --slides /path/to/tif \\
    --output ./qa_results \\
    --load-4bit
"""
import argparse
import json
import logging
import sys
import os
from pathlib import Path

# Load .env before anything else so env vars are available project-wide
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

logging.getLogger("slideseek").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slideseek import SlideSeekQARunner, QAResult, SlideSeekConfig
from slideseek.config import OllamaConfig, HFLLMConfig, HFVisionLLMConfig, OpenAILLMConfig, PathoModelConfig, SlideConfig, AgentConfig

log = logging.getLogger(__name__)


def build_config(args) -> SlideSeekConfig:

    return SlideSeekConfig(
        use_openai_llm=args.use_openai,
        use_hf_llm=(not args.use_ollama and not args.use_openai),
        use_vision_supervisor=not args.no_vision_supervisor,
        openai_llm=OpenAILLMConfig(
            model=args.openai_model,
            api_key=args.openai_api_key,
            api_base_url=args.openai_base_url,
        ),
        hf_llm=HFLLMConfig(
            model_id=args.hf_model,
            device=args.device,
            load_in_4bit=args.load_4bit,
            load_in_8bit=args.load_8bit,
            enable_thinking=args.enable_thinking,
        ),
        hf_vision_llm=HFVisionLLMConfig(
            model_id=args.vision_model,
            device=args.device,
            load_in_4bit=args.load_4bit,
            load_in_8bit=args.load_8bit,
        ),
        ollama=OllamaConfig(
            base_url=args.ollama_url,
            model=args.ollama_model,
            enable_thinking=args.enable_thinking,
        ),
        patho_model=PathoModelConfig(
            model_id=args.patho_model,
            device=args.device,
            load_in_4bit=args.load_4bit,
            load_in_8bit=args.load_8bit,
        ),
        slide=SlideConfig(roi_size=args.roi_size),
        agent=AgentConfig(
            max_supervisor_iterations=args.max_iter,
            max_parallel_explorers=args.max_explorers,
        ),
        output_dir=args.output,
        verbose=args.verbose,
    )


def run_mock(args, config):
    """Smoke-test with MockSlideViewer — reads real questions but uses a fake slide."""
    from slideseek.wsi.slide_viewer import MockSlideViewer
    from slideseek.agents.supervisor import SupervisorAgent
    from slideseek.agents.explorer import ExplorerAgent
    from slideseek.pipeline import QAResult
    import json as _json

    with open(args.json) as f:
        data = _json.load(f)

    by_id = {}
    for entry in data:
        by_id.setdefault(entry["Id"], {})[entry["type"]] = entry

    slide_ids = [args.slide_id] if args.slide_id else sorted(by_id.keys())[:3]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sid in slide_ids:
        entries = by_id.get(sid, {})
        for question_type, entry in sorted(entries.items()):
            task_key = f"{sid}_{question_type}"
            print(f"\n[MOCK] {task_key}: {entry['Question'][:70]}")

            slide = MockSlideViewer()
            supervisor = SupervisorAgent(config)
            supervisor.initialize_qa(
                slide_context=slide.get_slide_context(),
                question=entry["Question"],
                slide_id=sid,
            )

            all_rois = []
            for _ in range(min(2, config.agent.max_supervisor_iterations)):
                state = supervisor.plan_next_iteration()
                if state.finished or not state.tasks:
                    break
                for task in state.tasks[:1]:
                    agent = ExplorerAgent(task, slide, config)
                    finding = agent.run()
                    supervisor.add_explorer_finding(finding.task_id, finding.text_report, finding.key_rois)
                    all_rois.extend(finding.visited_rois)

            qa_answer = supervisor.generate_qa_answer()
            print(f"  → predicted: {qa_answer['presence']} ({qa_answer['num_lesions']} lesions)")
            print(f"  → gt:        {entry['Answer']['presence']} ({entry['Answer']['num_lesions']} lesions)")

            result_path = out_dir / f"{task_key}.json"
            with open(result_path, "w") as f:
                _json.dump({
                    "slide_id": sid,
                    "question_type": question_type,
                    "question": entry["Question"],
                    "predicted_presence": qa_answer["presence"],
                    "predicted_num_lesions": qa_answer["num_lesions"],
                    "predicted_lesions": qa_answer["predicted_lesions"],
                    "model_confidence": qa_answer["confidence"],
                    "reasoning": qa_answer["reasoning"],
                    "gt_presence": entry["Answer"]["presence"],
                    "gt_num_lesions": entry["Answer"]["num_lesions"],
                    "num_rois_examined": len(all_rois),
                    "mock": True,
                }, f, indent=2)
            print(f"  Saved → {result_path}")

    print("\n[MOCK] Done.")


def main():
    p = argparse.ArgumentParser(description="SlideSeek QA Dataset Evaluation")

    p.add_argument("--json",    required=True,  help="Path to roi_qa_dataset.json")
    p.add_argument("--slides",  required=True,  help="Directory containing .tif slide files")
    p.add_argument("--output",  default="./qa_results", help="Output directory")
    p.add_argument("--slide-id", default=None,  help="Run only this slide ID (for debugging)")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-run all slides even if result JSON already exists")
    p.add_argument("--start-from", type=int, default=0,
                   help="0-based index of the first case to process (use to split dataset across parallel runs)")
    p.add_argument("--end-at", type=int, default=None,
                   help="0-based exclusive end index (omit to process until end of dataset)")
    p.add_argument("--mock",    action="store_true",
                   help="Use MockSlideViewer (no real WSI, smoke-test only)")

    # Model config
    p.add_argument("--hf-model",      default="Qwen/Qwen3-4B",
                   help="HuggingFace model ID for the supervisor/explorer LLM")
    p.add_argument("--use-ollama",    action="store_true",
                   help="Use Ollama backend instead of HuggingFace")
    p.add_argument("--ollama-url",    default="http://192.168.63.184:11434")
    p.add_argument("--ollama-model",  default="qwen3:4b")
    p.add_argument("--use-openai",    action="store_true",
                   help="Use OpenAI API backend for text LLM (overrides --use-ollama / HF)")
    p.add_argument("--openai-model",  default="gpt-4o-mini",
                   help="OpenAI model name (default: gpt-4o-mini)")
    p.add_argument("--openai-api-key", default=None,
                   help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    p.add_argument("--openai-base-url", default=None,
                   help="Override OpenAI base URL (e.g. for Azure OpenAI)")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--patho-model",        default="WenchuanZhang/Patho-R1-3B")
    p.add_argument("--vision-model",       default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="HuggingFace VL model ID for the vision supervisor")
    p.add_argument("--no-vision-supervisor", action="store_true",
                   help="Disable vision supervisor (use text-only LLM instead)")
    p.add_argument("--device",        default="auto")
    p.add_argument("--load-4bit",     action="store_true")
    p.add_argument("--load-8bit",     action="store_true")
    p.add_argument("--no_vision_supervisor",     action="store_true")

    # Pipeline config
    p.add_argument("--roi-size",      type=int,   default=896)
    p.add_argument("--max-iter",      type=int,   default=10)
    p.add_argument("--max-explorers", type=int,   default=3)
    p.add_argument("--verbose",       action="store_true")

    args = p.parse_args()
    config = build_config(args)

    print(f"\n{'='*60}")
    print("SlideSeek — QA Dataset Evaluation")
    print(f"{'='*60}")
    print(f"  JSON:         {args.json}")
    print(f"  Slides dir:   {args.slides}")
    print(f"  Output:       {args.output}")
    if args.use_openai:
        print(f"  LLM:          {args.openai_model} (OpenAI API)")
    elif args.use_ollama:
        print(f"  LLM:          {args.ollama_model} @ {args.ollama_url} (Ollama)")
    else:
        print(f"  LLM:          {args.hf_model} (HuggingFace)")
    print(f"  Patho model:  {args.patho_model}")
    print(f"  Mock mode:    {args.mock}")
    print(f"{'='*60}\n")


    if args.mock:
        run_mock(args, config)
        return

    runner = SlideSeekQARunner(config=config, slides_dir=args.slides)

    if args.slide_id:
        # Wrap single-slide run through the dataset runner
        with open(args.json) as f:
            raw = json.load(f)
        by_id = {}
        for entry in raw:
            by_id.setdefault(entry["Id"], {})[entry["type"]] = entry

        if args.slide_id not in by_id:
            print(f"ERROR: slide_id '{args.slide_id}' not found in dataset")
            sys.exit(1)

        # Write a temp single-entry JSON and run
        import tempfile
        tmp_entries = [v for v in by_id[args.slide_id].values()]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(tmp_entries, tf)
            tmp_path = tf.name

        results = runner.run_dataset(tmp_path, output_dir=args.output, resume=not args.no_resume)
        os.unlink(tmp_path)
    else:
        results = runner.run_dataset(
            args.json,
            output_dir=args.output,
            resume=not args.no_resume,
            start_from=args.start_from,
            end_at=args.end_at,
        )

    # Print summary
    if results:
        from slideseek.pipeline import _presence_correct
        correct = sum(1 for r in results if _presence_correct(r.predicted_presence, r.gt_presence))
        print(f"\n{'='*60}")
        print(f"SUMMARY  ({len(results)} slides)")
        print(f"  Presence accuracy: {correct}/{len(results)} = {correct/len(results):.1%}")
        print(f"  Results saved to:  {args.output}/")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
