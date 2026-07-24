"""
Stage 1 – Global Screening (Trace Agent)

Analyses the WSI thumbnail, groups patches by pathological characteristics,
assigns severity scores and decides which regions need high-magnification review.

VLM (patho_r1)  → visual analysis
Text (Qwen3 4B) → JSON extraction / repair
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from PIL import Image

from models.base_model import BaseModel
from utils.image_utils import create_thumbnail_with_ids
from .prompts import TRACE_SYSTEM, TRACE_USER, JSON_EXTRACT_PROMPT


class TraceAgent:

    def __init__(self, model: BaseModel, verbose: bool = True):
        self.model = model
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
        thumbnail  : low-resolution WSI overview (PIL Image)
        tiles      : list of (patch_id, x1, y1, x2, y2) from WSILoader.partition_grid()
        tissue_type: e.g. "breast", "lung" …

        Returns
        -------
        dict with key "groups" – list of group dicts.
        Important keys per group:
          - id_list                  : patch IDs in this group
          - is_background            : True if group has no tissue
          - require_high_magnification
          - diagnostic_priority      : 1-5 (lower = higher priority)
          - observation_points       : key features for high-mag examination
        """
        annotated = create_thumbnail_with_ids(thumbnail, tiles)
        print(f"[annotated] {annotated}")

        prompt = TRACE_USER.format(tissue_type=tissue_type)
        full_prompt = prompt
        # full_prompt = f"{TRACE_SYSTEM}\n\n{prompt}"

        if self.verbose:
            print("[TraceAgent] Sending annotated thumbnail to VLM …")

        raw = self.model.generate_with_images(full_prompt, [annotated])

        if self.verbose:
            print("[TraceAgent] Raw VLM output:\n", raw[:500], "…")

        result = self._parse_json(raw)
        result = self._filter_background(result)
        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """Try direct parse, then ask text model to extract/repair."""
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        # Find first JSON object
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: ask Qwen3 to repair
        if self.verbose:
            print("[TraceAgent] JSON parse failed – asking text model to repair …")
        repair_prompt = JSON_EXTRACT_PROMPT.format(raw_text=raw)
        repaired = self.model.generate_text(repair_prompt)
        cleaned2 = re.sub(r"```(?:json)?", "", repaired).strip()
        match2 = re.search(r"\{.*\}", cleaned2, flags=re.DOTALL)
        if match2:
            try:
                return json.loads(match2.group())
            except json.JSONDecodeError:
                pass

        # Ultimate fallback: empty result
        return {"groups": []}

    @staticmethod
    def _filter_background(result: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only groups that contain tissue (is_background == False)."""
        groups = result.get("groups", [])
        result["groups_all"] = groups
        result["groups"] = [g for g in groups if not g.get("is_background", False)]
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
