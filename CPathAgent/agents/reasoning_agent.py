"""
Stage 3 – Multi-scale Multi-view Sequence Reasoning (Reasoning Agent)

New two-phase flow
------------------
Phase A (VLM): For each cropped view, call the VLM individually with
               STEP3_PATCH_DESC_USER to produce a per-patch description.

Phase B (Text LLM): Feed all per-patch descriptions to the text LLM with
                    STEP3_SYNTHESIS_USER (or VQA variant) to produce a
                    unified pathological report.

Post-processing: Extract a short diagnosis term (e.g. "IDC") from the report
                 using the text LLM with STEP3_DIAGNOSIS_EXTRACT_PROMPT.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from models.base_model import BaseModel
from utils.image_utils import crop_view, resize_for_model
from .prompts import (
    STEP3_PATCH_DESC_USER,
    STEP3_SYNTHESIS_USER,
    STEP3_SYNTHESIS_VQA_USER,
    STEP3_DIAGNOSIS_EXTRACT_PROMPT,
)


class ReasoningAgent:

    def __init__(
        self,
        model: BaseModel,
        model_input_size: int = 1008,
        verbose: bool = True,
    ):
        self.model = model
        self.model_input_size = model_input_size
        self.verbose = verbose

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        region_image: Image.Image | List[Image.Image],
        nav_steps: List[Dict[str, Any]] | List[List[Dict[str, Any]]],
        tissue_type: str = "unknown",
        question: Optional[str] = None,
        wsi_loader: Optional[Any] = None,
    ) -> Dict[str, str]:
        """
        Parameters
        ----------
        region_image : a single region PIL Image, or a list of region images
        nav_steps    : output of NavigateAgent.run() for the corresponding region(s)
        tissue_type  : tissue origin string
        question     : if provided, answer the question (VQA mode)

        Returns
        -------
        dict with keys:
          - "patch_descriptions" : list of per-view VLM descriptions
          - "reasoning"          : full synthesised narrative from the text LLM
          - "report"             : extracted **Pathological Report:** paragraph
          - "diagnosis"          : short diagnosis term, e.g. "IDC"
          - "answer"             : final answer letter/option (VQA mode only)
        """
        # Normalise to list-of-pairs
        if isinstance(region_image, list):
            pairs = list(zip(region_image, nav_steps))
        else:
            pairs = [(region_image, nav_steps)]

        # ── Phase A: crop views and VLM-describe each one ─────────────────────
        views: List[Image.Image] = []
        step_descs: List[str] = []
        obs_points_all: List[List[str]] = []
        for img, steps in pairs:
            v, d, o = self._crop_views(img, steps, wsi_loader)
            views.extend(v)
            step_descs.extend(d)
            obs_points_all.extend(o)

        if self.verbose:
            print(
                f"[ReasoningAgent] Cropped {len(views)} views from {len(pairs)} region(s). "
                "Running per-patch VLM descriptions …"
            )

        patch_descriptions = self._describe_views(views, step_descs, obs_points_all)

        # ── Phase B: Text LLM synthesises all descriptions ────────────────────
        combined_observations = self._build_combined_observations(
            step_descs, patch_descriptions
        )

        if question:
            synthesis_prompt = STEP3_SYNTHESIS_VQA_USER.format(
                question=question,
                combined_observations=combined_observations,
            )
        else:
            synthesis_prompt = STEP3_SYNTHESIS_USER.format(
                tissue_type=tissue_type,
                combined_observations=combined_observations,
            )

        if self.verbose:
            print("[ReasoningAgent] Sending descriptions to text LLM for synthesis …")

        synthesis = self.model.generate_text(synthesis_prompt)

        if self.verbose:
            print("[ReasoningAgent] Synthesis (first 600 chars):\n", synthesis, "…")

        # ── Post-processing: extract structured outputs ────────────────────────
        report = self._extract_report(synthesis)
        diagnosis = self._extract_diagnosis(synthesis)
        answer = self._extract_answer(synthesis) if question else ""

        return {
            "patch_descriptions": patch_descriptions,
            "reasoning": synthesis,
            "report": report,
            "diagnosis": diagnosis,
            "answer": answer,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _crop_views(
        self,
        region_image: Image.Image,
        nav_steps: List[Dict[str, Any]],
        wsi_loader: Optional[Any] = None,
    ) -> Tuple[List[Image.Image], List[str], List[List[str]]]:
        """Crop each nav step at the correct magnification.

        Returns views, step descriptors, and per-step observation_points
        (propagated from Stage 1 screening).
        """
        views = []
        descs = []
        obs_points_list = []

        for s in nav_steps:
            x1, y1, x2, y2 = s["region_coordinate"]
            mag = float(s.get("magnification", 2.0))
            need = s.get("need_to_see", "")
            obs_points = s.get("observation_points", [])

            if wsi_loader is not None:
                W, H = wsi_loader.width, wsi_loader.height
                view = wsi_loader.get_region(
                    x1 / W, y1 / H, x2 / W, y2 / H,
                    self.model_input_size,
                )
                desc = (
                    f"Step {s['step']} | {mag}× | "
                    f"WSI px [{int(x1)},{int(y1)},{int(x2)},{int(y2)}] "
                    f"| focus: {need}"
                )
            else:
                view = crop_view(region_image, x1, y1, x2, y2, self.model_input_size)
                desc = (
                    f"Step {s['step']} | {mag}× | "
                    f"region [{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}] | focus: {need}"
                )

            views.append(view)
            descs.append(desc)
            obs_points_list.append(obs_points)

        return views, descs, obs_points_list

    def _describe_views(
        self,
        views: List[Image.Image],
        step_descs: List[str],
        obs_points_all: Optional[List[List[str]]] = None,
    ) -> List[str]:
        """Call VLM once per view to obtain a per-patch description."""
        descriptions = []
        for i, (view, step_info) in enumerate(zip(views, step_descs)):
            obs = obs_points_all[i] if obs_points_all else []
            obs_text = (
                "\n".join(f"- {p}" for p in obs)
                if obs
                else "- General tissue architecture, cellular density, and nuclear morphology"
            )
            prompt = STEP3_PATCH_DESC_USER.format(
                step_info=step_info,
                observation_points=obs_text,
            )
            if self.verbose:
                print(f"[ReasoningAgent] Describing view {i + 1}/{len(views)} …")
            desc = self.model.generate_with_images(prompt, [view])
            descriptions.append(desc.strip())
        return descriptions

    @staticmethod
    def _build_combined_observations(
        step_descs: List[str],
        patch_descriptions: List[str],
    ) -> str:
        """Interleave step context lines with VLM descriptions."""
        lines = []
        for step_info, patch_desc in zip(step_descs, patch_descriptions):
            lines.append(f"[{step_info}]")
            lines.append(patch_desc)
            lines.append("")
        return "\n".join(lines).strip()

    def _extract_diagnosis(self, text: str) -> str:
        """
        Extract the short diagnosis term (e.g. "IDC") from synthesised text.

        Strategy:
          1. Look for **Diagnosis:** marker and grab the following token(s).
          2. If not found, ask the text LLM directly.
        """
        # Try regex first – handles "**Diagnosis:** IDC" and "Diagnosis: IDC"
        match = re.search(
            r"\*?\*?Diagnosis\*?\*?:\s*(.+?)(?:\n|$)",
            text,
            re.IGNORECASE,
        )
        if match:
            raw = match.group(1).strip().strip("*").strip()
            # Take only the first line / sentence fragment
            raw = re.split(r"[.\n]", raw)[0].strip()
            if raw:
                return raw

        # Fallback: ask the text LLM
        prompt = STEP3_DIAGNOSIS_EXTRACT_PROMPT.format(report_text=text[:2000])
        result = self.model.generate_text(prompt).strip()
        # Sanitise – keep only the first non-empty line
        for line in result.splitlines():
            line = line.strip().strip("*").strip('"').strip("'")
            if line:
                return line
        return ""

    @staticmethod
    def _extract_report(text: str) -> str:
        """Extract the paragraph after **Pathological Report:**"""
        marker = "**Pathological Report:**"
        idx = text.find(marker)
        if idx == -1:
            idx = text.lower().find("pathological report:")
        if idx == -1:
            return ""
        after = text[idx:].strip()
        # Stop before **Diagnosis:** or next double newline
        stop = re.search(r"\n\n|\*\*Diagnosis\*\*:", after[len(marker):])
        if stop:
            return after[: len(marker) + stop.start()].strip()
        return after.strip()

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract 'Therefore, the answer is (X).' from VQA output."""
        match = re.search(
            r"[Tt]herefore[,\s]+the answer is\s*\(?([A-Ea-e])\)?",
            text,
        )
        if match:
            return match.group(1).upper()
        match2 = re.search(r"\b([A-E])\b[.\s]*$", text.strip())
        if match2:
            return match2.group(1).upper()
        return ""
