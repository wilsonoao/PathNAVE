"""
Stage 1 (revised) – Patch Screening Agent

Four-step pipeline:
  1. Cut thumbnail into patches
  2. Each patch  → VLM description
  3. Collect descriptions
  4. LLM → cluster by pathological similarity + select ROI

The output format is identical to TraceAgent so the rest of the pipeline
(NavigateAgent, ReasoningAgent) is unaffected.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from PIL import Image

from models.base_model import BaseModel
from utils.image_utils import crop_view
from .prompts import PATCH_DESC_USER, CLUSTER_USER, JSON_EXTRACT_PROMPT


class PatchScreeningAgent:

    def __init__(
        self,
        model: BaseModel,
        patch_input_size: int = 512,
        verbose: bool = True,
    ):
        self.model = model
        self.patch_input_size = patch_input_size
        self.verbose = verbose

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        thumbnail: Image.Image,
        tiles: List[Tuple[int, float, float, float, float]],
        tissue_type: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        thumbnail   : low-resolution WSI overview (PIL Image)
        tiles       : list of (patch_id, x1, y1, x2, y2) – normalised coords
        tissue_type : e.g. "breast", "lung" …

        Returns
        -------
        dict with keys:
          "groups_all"   – all groups before background filtering
          "groups"       – groups where is_background is false
          "descriptions" – per-patch VLM descriptions {patch_id: text}
        """
        # ── Step 1: crop patches from thumbnail ───────────────────────────────
        if self.verbose:
            print(f"[PatchScreeningAgent] Step 1 – cropping {len(tiles)} patches …")
        patches = self._crop_patches(thumbnail, tiles)

        # ── Step 2: per-patch VLM descriptions ───────────────────────────────
        if self.verbose:
            print("[PatchScreeningAgent] Step 2 – generating per-patch descriptions …")
        descriptions = self._describe_patches(patches)

        # ── Step 3: descriptions already collected ────────────────────────────
        nonempty = sum(1 for v in descriptions.values() if v.strip())
        if self.verbose:
            print(f"[PatchScreeningAgent] Step 3 – collected {len(descriptions)} descriptions "
                  f"({nonempty} non-empty).")

        # ── Step 4: LLM clustering + ROI selection ────────────────────────────
        if self.verbose:
            print("[PatchScreeningAgent] Step 4 – clustering & selecting ROI via LLM …")
        result = self._cluster_and_select_roi(descriptions, tissue_type)
        result["descriptions"] = descriptions
        result = self._filter_background(result)
        return result

    def get_high_priority_regions(
        self, result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Return groups that need high-magnification, sorted by priority."""
        high_mag = [
            g for g in result.get("groups", [])
            if g.get("require_high_magnification", False)
        ]
        return sorted(high_mag, key=lambda g: g.get("diagnostic_priority", 99))

    # ── private ───────────────────────────────────────────────────────────────

    def _crop_patches(
        self,
        thumbnail: Image.Image,
        tiles: List[Tuple[int, float, float, float, float]],
    ) -> List[Tuple[int, Image.Image]]:
        """Return list of (patch_id, cropped_image)."""
        result = []
        for pid, x1, y1, x2, y2 in tiles:
            patch = crop_view(thumbnail, x1, y1, x2, y2, self.patch_input_size)
            result.append((pid, patch))
        return result

    def _describe_patches(
        self,
        patches: List[Tuple[int, Image.Image]],
    ) -> Dict[int, str]:
        """Call VLM once per patch; return {patch_id: description}."""
        descriptions: Dict[int, str] = {}
        total = len(patches)
        for i, (pid, patch_img) in enumerate(patches):
            if self.verbose:
                print(f"  [{i+1}/{total}] Describing patch {pid} …")
            desc = self.model.generate_with_images(PATCH_DESC_USER, [patch_img])
            descriptions[pid] = desc.strip()
        return descriptions

    def _cluster_and_select_roi(
        self,
        descriptions: Dict[int, str],
        tissue_type: str,
    ) -> Dict[str, Any]:
        """Feed all descriptions to the text LLM; parse cluster + ROI JSON.

        If the first call produces unparseable output, retry once with 2×
        the token budget before falling back to the catch-all group.
        """
        lines = [
            f"Patch {pid}: {desc if desc.strip() else '(no description)'}"
            for pid, desc in sorted(descriptions.items())
        ]
        patch_descriptions_text = "\n\n".join(lines)

        prompt = CLUSTER_USER.format(
            tissue_type=tissue_type,
            patch_descriptions=patch_descriptions_text,
        )

        all_patch_ids = list(descriptions.keys())

        # ── first attempt (4096 tokens – enough for any realistic group count) ──
        raw = self.model.generate_text(prompt, max_new_tokens=4096)
        if self.verbose:
            print("[PatchScreeningAgent] LLM cluster output (first 400 chars):\n",
                  raw[:400], "…")

        result = self._try_parse_json(raw)

        # ── retry with doubled token budget if first attempt failed ────────────
        if result is None:
            if self.verbose:
                print("[PatchScreeningAgent] JSON parse failed – retrying with 2× token budget …")
            raw_retry = self.model.generate_text(prompt, max_new_tokens=8192)
            if self.verbose:
                print("[PatchScreeningAgent] Retry output (first 400 chars):\n",
                      raw_retry[:400], "…")
            result = self._try_parse_json(raw_retry)
            if result is not None:
                raw = raw_retry  # keep the successful output for logging

        if result is None:
            if self.verbose:
                print("[PatchScreeningAgent] Clustering failed – using fallback single group.")
            ids = sorted(all_patch_ids)
            result = {
                "groups": [{
                    "name": "Unclassified tissue",
                    "description": "Automatic clustering failed; all patches queued for review.",
                    "id_list": ids,
                    "is_background": False,
                    "require_high_magnification": True,
                    "diagnostic_priority": 1,
                    "observation_points": [
                        "Overall tissue architecture and organisation",
                        "Cellular density and distribution",
                        "Nuclear morphology and pleomorphism",
                        "Any architecturally or cytologically abnormal pattern",
                    ],
                }]
            }

        result["raw_cluster_output"] = raw
        return result

    def _try_parse_json(self, raw: str) -> Dict[str, Any] | None:
        """Try to parse a JSON groups dict from raw LLM output.
        Returns the parsed dict if successful, None otherwise."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if result.get("groups"):
                    return result
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _filter_background(result: Dict[str, Any]) -> Dict[str, Any]:
        groups = result.get("groups", [])
        result["groups_all"] = groups
        result["groups"] = [g for g in groups if not g.get("is_background", False)]
        return result
