"""
SlideSeek Pipeline
------------------
架構：

  SlideSeekPipeline
  ├── run()               → SlideSeekResult   (diagnosis mode，開放式鑑別診斷)
  └── run_qa_task()       → QAResult          (QA mode，回答特定問題)

  SlideSeekQARunner
  └── run_dataset()       → list[QAResult]    (批次跑 roi_qa_dataset.json)
      └── 每個 slide 呼叫 SlideSeekPipeline.run_qa_task()

核心 loop 只有一份：_run_exploration_loop()
兩個 mode 都透過它跑 supervisor–explorer，差別只在最後的 finalize step。
"""
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

from slideseek.config import SlideSeekConfig, DEFAULT_CONFIG
from slideseek.agents.supervisor import SupervisorAgent
from slideseek.agents.explorer import ExplorerAgent
from slideseek.agents.patch_registry import reset_registry
from slideseek.models import patho_model
from slideseek.wsi.slide_viewer import SlideViewer, MockSlideViewer

logger = logging.getLogger(__name__)


def _presence_correct(predicted: str, gt: str) -> bool:
    """
    Compare predicted vs ground-truth presence, handling IDC/ILC short labels
    against full names like 'Invasive Ductal Carcinoma (IDC)'.
    """
    p, g = predicted.strip().upper(), gt.strip().upper()
    if p == g:
        return True
    # Short label match: "IDC" in "INVASIVE DUCTAL CARCINOMA (IDC)"
    if len(p) <= 5 and p in g:
        return True
    if len(g) <= 5 and g in p:
        return True
    return False


# ------------------------------------------------------------------ #
#  Result types                                                      #
# ------------------------------------------------------------------ #
@dataclass
class SlideSeekResult:
    """Diagnosis-mode result."""
    primary_diagnosis: str
    differential_diagnoses: list[str]
    confidence: str          # "HIGH" | "LOW"
    report: str
    num_rois_examined: int
    num_iterations: int
    final_rois: list[dict]


@dataclass
class QAResult:
    """QA-mode result for one entry in roi_qa_dataset.json."""
    slide_id: str
    question: str
    question_type: str       # "DetectionLocalization" | "MultiLesion"

    # Prediction
    predicted_presence: str  # "Yes" | "No"
    predicted_answer_text: str
    predicted_num_lesions: int
    predicted_lesions: list[dict]
    model_confidence: str    # "HIGH" | "LOW"
    reasoning: str

    # Ground truth
    gt_presence: str
    gt_num_lesions: int
    gt_lesions: list[dict] = field(default_factory=list)

    # Visited patches
    visited_rois_info: list[dict] = field(default_factory=list)

    # Meta
    num_rois_examined: int = 0
    num_iterations: int = 0


# ------------------------------------------------------------------ #
#  Core pipeline                                                       #
# ------------------------------------------------------------------ #
class SlideSeekPipeline:
    """
    SlideSeek pipeline — implements Algorithm 1 from the paper.

    Two public entry points:
      run()         — open-ended differential diagnosis
      run_qa_task() — answer a specific question (for QA dataset evaluation)

    Both share the same supervisor–explorer loop (_run_exploration_loop).
    """

    def __init__(self, config: Optional[SlideSeekConfig] = None):
        self.config = config or DEFAULT_CONFIG
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    # ------------------------------------------------------------------ #
    #  Public: diagnosis mode                                              #
    # ------------------------------------------------------------------ #
    def run(
        self,
        wsi_path: Optional[str] = None,
        tissue_site: str = "unknown",
        patient_sex: str = "unknown",
        clinical_context: str = "No additional clinical context provided.",
        mock: bool = False,
    ) -> SlideSeekResult:
        """
        Open-ended differential diagnosis on a whole-slide image.

        Args:
            wsi_path:  Path to .svs/.ndpi/.tif. None requires mock=True.
            tissue_site, patient_sex, clinical_context: clinical context.
            mock:      Use MockSlideViewer (no real file needed).
        """
        reset_registry()
        slide = self._open_slide(wsi_path, mock)
        try:
            supervisor = SupervisorAgent(self.config)
            supervisor._thumbnail = slide.thumbnail
            supervisor.initialize(
                slide_context=slide.get_slide_context(),
                tissue_site=tissue_site,
                patient_sex=patient_sex,
                clinical_context=clinical_context,
            )
            supervisor.analyze_thumbnail()

            visited_rois, _, n_iter = self._run_exploration_loop(supervisor, slide)

            # Finalize: patho_r1 differential + report
            final_roi_specs  = supervisor.select_final_rois()
            final_roi_images = self._collect_roi_images(final_roi_specs, visited_rois, slide)

            logger.info(f"Requesting differential from patho_r1 ({len(final_roi_images)} ROIs)...")
            differential = (
                patho_model.caption_multiple_rois(
                    images=final_roi_images,
                    tissue_site=tissue_site,
                    patient_sex=patient_sex,
                    config=self.config.patho_model,
                )
                if final_roi_images
                else "Insufficient ROI images for differential diagnosis."
            )

            report  = supervisor.generate_report(differential)
            primary, diffs, conf = self._parse_report(report, differential)

            result = SlideSeekResult(
                primary_diagnosis=primary,
                differential_diagnoses=diffs,
                confidence=conf,
                report=report,
                num_rois_examined=len(visited_rois),
                num_iterations=n_iter,
                final_rois=final_roi_specs,
            )
            self._save_diagnosis_result(result, tissue_site)
            return result

        finally:
            slide.close()

    # ------------------------------------------------------------------ #
    #  Public: QA mode                                                     #
    # ------------------------------------------------------------------ #
    def run_qa_task(
        self,
        wsi_path: Optional[str] = None,
        question: str = "",
        slide_id: str = "",
        question_type: str = "DetectionLocalization",
        gt_answer: Optional[dict] = None,
        mock: bool = False,
    ) -> QAResult:
        """
        Answer a specific pathology question about a slide.
        Used by SlideSeekQARunner to evaluate roi_qa_dataset.json.

        Args:
            wsi_path:  Path to .tif. None requires mock=True.
            question:  The QA question (e.g. "Does this slide contain tumor regions?")
            slide_id:  Slide identifier (used for logging/saving).
            gt_answer: Ground-truth dict {presence, num_lesions, lesions} (optional).
            mock:      Use MockSlideViewer.
        """
        # Normalize gt_answer: DetectionLocalization → dict, MultiLesion → plain string
        if isinstance(gt_answer, dict):
            gt = gt_answer
        elif isinstance(gt_answer, str):
            gt = {"presence": gt_answer, "num_lesions": 0, "lesions": []}
        else:
            gt = {}
        reset_registry()
        slide = self._open_slide(wsi_path, mock)
        try:
            supervisor = SupervisorAgent(self.config)
            supervisor._thumbnail = slide.thumbnail
            supervisor.initialize_qa(
                slide_context=slide.get_slide_context(),
                question=question,
                slide_id=slide_id,
            )
            supervisor.analyze_thumbnail()

            visited_rois, roi_descriptions, n_iter = self._run_exploration_loop(supervisor, slide)

            # Finalize: structured QA answer
            qa_answer = supervisor.generate_qa_answer()

            return QAResult(
                slide_id=slide_id,
                question=question,
                question_type=question_type,
                predicted_presence=qa_answer["presence"],
                predicted_answer_text=qa_answer["answer_text"],
                predicted_num_lesions=qa_answer["num_lesions"],
                predicted_lesions=qa_answer["predicted_lesions"],
                model_confidence=qa_answer["confidence"],
                reasoning=qa_answer["reasoning"],
                gt_presence=gt.get("presence", ""),
                gt_num_lesions=gt.get("num_lesions", 0),
                gt_lesions=gt.get("lesions", []),
                visited_rois_info=[
                    {"x": r.x, "y": r.y, "magnification": r.magnification, "size": r.size, "description": desc}
                    for r, desc in zip(visited_rois, roi_descriptions)
                ],
                num_rois_examined=len(visited_rois),
                num_iterations=n_iter,
            )
        finally:
            slide.close()

    # ------------------------------------------------------------------ #
    #  Shared exploration loop (Algorithm 1)                              #
    # ------------------------------------------------------------------ #
    def _run_exploration_loop(
        self,
        supervisor: SupervisorAgent,
        slide,
    ) -> tuple[list, int]:
        """
        Runs the supervisor–explorer loop until supervisor signals finished.
        Shared by both diagnosis mode and QA mode.

        Returns:
            (all_visited_rois, total_iterations)
        """
        all_visited_rois = []
        all_roi_descriptions = []
        total_iterations = 0

        while True:
            total_iterations += 1
            state = supervisor.plan_next_iteration()

            logger.info(
                f"Supervisor iter {state.iteration}: "
                f"tasks={len(state.tasks)}  finished={state.finished}"
            )

            if state.finished:
                break
            if not state.tasks:
                logger.warning("Supervisor produced no tasks — stopping loop.")
                break

            tasks = state.tasks[: self.config.agent.max_tasks_per_iteration]
            findings = self._run_explorers_parallel(tasks, slide)

            for finding in findings:
                supervisor.add_explorer_finding(
                    task_id=finding.task_id,
                    finding=finding.text_report,
                    key_rois=finding.key_rois,
                )
                all_visited_rois.extend(finding.visited_rois)
                all_roi_descriptions.extend(finding.roi_descriptions)

        logger.info(
            f"Exploration complete: {total_iterations} iterations, "
            f"{len(all_visited_rois)} ROIs visited."
        )
        return all_visited_rois, all_roi_descriptions, total_iterations

    # ------------------------------------------------------------------ #
    #  Parallel explorer execution                                         #
    # ------------------------------------------------------------------ #
    def _run_explorers_parallel(self, tasks, slide) -> list:
        max_workers = self.config.agent.max_parallel_explorers
        findings = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(lambda t: ExplorerAgent(t, slide, self.config).run(), task): task
                for task in tasks
            }
            # print(futures)
            for future in as_completed(futures):
                task = futures[future]
                try:
                    findings.append(future.result())
                    logger.info(f"Explorer {task.task_id} completed.")
                except Exception as e:
                    logger.error(f"Explorer {task.task_id} failed: {e}")
        return findings

    # ------------------------------------------------------------------ #
    #  ROI image collection                                                #
    # ------------------------------------------------------------------ #
    def _collect_roi_images(self, roi_specs, visited_rois, slide) -> list[np.ndarray]:
        visited_map = {(r.x, r.y, r.magnification): r.image for r in visited_rois}
        images = []
        for spec in roi_specs:
            key = (spec.get("x"), spec.get("y"), spec.get("magnification", 20.0))
            if key in visited_map:
                images.append(visited_map[key])
            else:
                try:
                    images.append(slide.navigate(*key).image)
                except Exception as e:
                    logger.warning(f"Could not re-navigate to ROI {spec}: {e}")
        return images

    # ------------------------------------------------------------------ #
    #  Slide open helper                                                   #
    # ------------------------------------------------------------------ #
    def _open_slide(self, wsi_path: Optional[str], mock: bool):
        if mock or wsi_path is None:
            logger.info("Using MockSlideViewer")
            return MockSlideViewer()
        return SlideViewer(wsi_path, roi_size=self.config.slide.roi_size)

    # ------------------------------------------------------------------ #
    #  Helpers: diagnosis mode                                             #
    # ------------------------------------------------------------------ #
    def _parse_report(self, report: str, differential: str) -> tuple[str, list[str], str]:
        primary, diffs, confidence = "", [], "LOW"
        for line in (report + "\n" + differential).splitlines():
            low = line.lower()
            if "primary diagnosis" in low and not primary:
                primary = line.split(":", 1)[-1].strip().strip("*").strip()
            if "high confidence" in low:
                confidence = "HIGH"
            if "low confidence" in low:
                confidence = "LOW"

        in_other = False
        for line in differential.splitlines():
            if "other possibilities" in line.lower():
                in_other = True
                continue
            if in_other and line.strip():
                text = line.strip().lstrip("12.)-").strip()
                if text:
                    diffs.append(text)
            if len(diffs) >= 2:
                break

        if not primary:
            primary = differential.split("\n")[0][:120]
        return primary, diffs, confidence

    def _save_diagnosis_result(self, result: SlideSeekResult, tissue_site: str):
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"report_{tissue_site}.txt").write_text(result.report)
        (out / f"summary_{tissue_site}.json").write_text(json.dumps({
            "primary_diagnosis":      result.primary_diagnosis,
            "differential_diagnoses": result.differential_diagnoses,
            "confidence":             result.confidence,
            "num_rois_examined":      result.num_rois_examined,
            "num_iterations":         result.num_iterations,
            "final_rois":             result.final_rois,
        }, indent=2))
        logger.info(f"Diagnosis result saved to {out}")


# ------------------------------------------------------------------ #
#  QA dataset runner                                                   #
# ------------------------------------------------------------------ #
class SlideSeekQARunner:
    """
    Batch-runs roi_qa_dataset.json through SlideSeekPipeline.run_qa_task().

    Usage:
        runner = SlideSeekQARunner(config, slides_dir="/path/to/tifs")
        results = runner.run_dataset("roi_qa_dataset.json", output_dir="./qa_results")
    """

    def __init__(
        self,
        config: Optional[SlideSeekConfig] = None,
        slides_dir: str = ".",
    ):
        self.config = config or DEFAULT_CONFIG
        self.slides_dir = Path(slides_dir)
        self.pipeline = SlideSeekPipeline(config)

    # ------------------------------------------------------------------ #
    #  Dataset-level entry point                                           #
    # ------------------------------------------------------------------ #
    def run_dataset(
        self,
        json_path: str,
        output_dir: str = "./qa_results",
        resume: bool = True,
        start_from: int = 0,
        end_at: Optional[int] = None,
    ) -> list[QAResult]:
        """
        Run inference on every slide in roi_qa_dataset.json.

        Args:
            json_path:   Path to roi_qa_dataset.json
            output_dir:  Where to write per-slide JSONs + manifest
            resume:      Skip slides that already have a result file
            start_from:  0-based index of the first case to process (for parallel runs)
            end_at:      0-based exclusive end index (omit to process until end)
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(json_path) as f:
            raw = json.load(f)

        # Group by slide id
        by_id: dict[str, dict] = {}
        for entry in raw:
            by_id.setdefault(entry["Id"], {})[entry["type"]] = entry

        all_results: list[QAResult] = []
        skipped: list[str] = []

        sorted_slides = sorted(by_id.items())
        total = len(sorted_slides)
        effective_end = end_at if (end_at is not None and end_at <= total) else total
        sorted_slides = sorted_slides[start_from:effective_end]
        if start_from > 0 or end_at is not None:
            print(f"[run_dataset] Processing cases [{start_from}, {effective_end}) "
                  f"of {total} total")

        for slide_id, entries in sorted_slides:
            tif_path = self._find_tif(slide_id)

            for question_type, entry in sorted(entries.items()):
                task_key = f"{slide_id}_{question_type}"
                result_path = out_dir / f"{task_key}.json"

                # Resume: skip cases whose output JSON already exists
                if resume and result_path.exists():
                    try:
                        with open(result_path) as f:
                            d = json.load(f)
                        d.setdefault("visited_rois_info", d.pop("visited_rois", []))
                        d.setdefault("gt_lesions", [])
                        all_results.append(QAResult(**{k: v for k, v in d.items() if k in QAResult.__dataclass_fields__}))
                        print(f"[SKIP] {task_key}: result already exists")
                    except Exception as e:
                        logger.warning(f"[SKIP] {task_key}: result exists but unreadable ({e}), skipping anyway")
                        skipped.append(task_key)
                    continue

                if tif_path is None:
                    logger.warning(f"{slide_id}: .tif not found in {self.slides_dir}")
                    skipped.append(task_key)
                    continue

                logger.info(f"\n{'='*60}\n{task_key}  ({tif_path.name})\n{'='*60}")
                try:
                    result = self.pipeline.run_qa_task(
                        wsi_path=str(tif_path),
                        question=entry["Question"],
                        slide_id=slide_id,
                        question_type=question_type,
                        gt_answer=entry["Answer"],
                    )
                    all_results.append(result)

                    with open(result_path, "w") as f:
                        json.dump(self._to_dict(result), f, indent=2)

                    lesions_path = out_dir / f"{task_key}_lesions.json"
                    with open(lesions_path, "w") as f:
                        json.dump({
                            "gt_lesions": result.gt_lesions,
                            "visited_rois": result.visited_rois_info,
                        }, f, indent=2)

                    logger.info(
                        f"{task_key}: pred={result.predicted_presence} "
                        f"({result.predicted_num_lesions}), "
                        f"gt={result.gt_presence} ({result.gt_num_lesions})"
                    )
                except Exception as e:
                    logger.error(f"{task_key}: FAILED — {e}", exc_info=True)
                    skipped.append(task_key)

        self._write_manifest(all_results, out_dir, skipped, start_from=start_from, end_at=end_at)
        return all_results

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _find_tif(self, slide_id: str) -> Optional[Path]:
        for ext in [".tif", ".tiff", ".svs", ".ndpi"]:
            p = self.slides_dir / f"{slide_id}{ext}"
            if p.exists():
                return p
        matches = list(self.slides_dir.glob(f"*{slide_id}*"))
        return matches[0] if matches else None

    @staticmethod
    def _to_dict(r: QAResult) -> dict:
        return {
            "slide_id":              r.slide_id,
            "question":              r.question,
            "question_type":         r.question_type,
            "predicted_presence":    r.predicted_presence,
            "predicted_answer_text": r.predicted_answer_text,
            "predicted_num_lesions": r.predicted_num_lesions,
            "predicted_lesions":     r.predicted_lesions,
            "model_confidence":      r.model_confidence,
            "reasoning":             r.reasoning,
            "gt_presence":           r.gt_presence,
            "gt_num_lesions":        r.gt_num_lesions,
            "visited_rois":          r.visited_rois_info,
            "num_rois_examined":     r.num_rois_examined,
            "num_iterations":        r.num_iterations,
        }

    def _write_manifest(self, results: list[QAResult], out_dir: Path, skipped: list[str],
                        start_from: int = 0, end_at: Optional[int] = None):
        n = len(results)
        correct = sum(
            1 for r in results
            if _presence_correct(r.predicted_presence, r.gt_presence)
        )
        manifest = {
            "total_slides":      n + len(skipped),
            "processed":         n,
            "skipped":           skipped,
            "presence_accuracy": round(correct / n, 4) if n else 0,
            "results": [
                {
                    "slide_id":             r.slide_id,
                    "predicted_presence":   r.predicted_presence,
                    "gt_presence":          r.gt_presence,
                    "presence_correct":     _presence_correct(r.predicted_presence, r.gt_presence),
                    "predicted_num_lesions": r.predicted_num_lesions,
                    "gt_num_lesions":       r.gt_num_lesions,
                    "model_confidence":     r.model_confidence,
                    "num_rois_examined":    r.num_rois_examined,
                }
                for r in results
            ],
        }
        if start_from > 0 or end_at is not None:
            suffix = f"_{start_from}-{end_at if end_at is not None else 'end'}"
        else:
            suffix = ""
        manifest_path = out_dir / f"manifest{suffix}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info(f"Manifest → {manifest_path}")
        logger.info(f"Presence accuracy: {correct}/{n} = {manifest['presence_accuracy']:.1%}")
