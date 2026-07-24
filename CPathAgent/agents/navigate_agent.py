"""
Stage 2 – Navigation Planning (Navigate Agent)

For each preserved high-magnification region, generate a dynamic multi-scale
viewing path by:
  1. Tiling the region into a 16×16 grid (256 sub-patches)
  2. Sending each sub-patch to the VLM for a brief description
  3. Feeding all descriptions to the text LLM to decide which sub-patches
     to examine and at what magnification
  4. Converting the LLM plan to the standard step format

VLM (patho_r1)  → per-sub-patch visual description
Text (Qwen3 4B) → viewing plan (which patches, what magnification)
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from models.base_model import BaseModel
from utils.image_utils import crop_view
from .prompts import (
    NAV_SUBPATCH_DESC_USER,
    NAV_GRID_PLAN_USER,
    NAV_GRID_PLAN_VQA_USER,
    JSON_EXTRACT_PROMPT,
)


class NavigateAgent:

    def __init__(
        self,
        model: BaseModel,
        max_steps: int = 8,
        grid_size: int = 2,
        subpatch_size: int = 512,
        verbose: bool = True,
    ):
        self.model = model
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.subpatch_size = subpatch_size
        self.verbose = verbose

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        region_image: Image.Image,
        tissue_type: str = "unknown",
        question: Optional[str] = None,
        region_bbox: Optional[List[float]] = None,
        wsi_size: Optional[Tuple[int, int]] = None,
        observation_points: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parameters
        ----------
        region_image : the full region (or its proxy) as a PIL Image
        tissue_type  : tissue origin string
        question     : if provided, generate a question-oriented navigation path
        region_bbox  : [x1, y1, x2, y2] of this region in WSI normalised coords.
        wsi_size     : (width_px, height_px) of the full WSI.
                       When both region_bbox and wsi_size are provided, output
                       region_coordinate values are absolute pixel coordinates.
                       Without wsi_size, WSI-normalised coords are stored.

        Returns
        -------
        list of step dicts, each containing:
          - step               : int
          - magnification      : float (2.0–5.0)
          - region_coordinate  : [x1, y1, x2, y2]
                                 absolute pixels  when wsi_size provided,
                                 WSI-normalised   when only region_bbox provided,
                                 region-relative  [0, 1] otherwise
          - reasoning          : str
          - need_to_see        : str
        """
        # ── Step 1: tile region into grid_size×grid_size sub-patches ─────────
        if self.verbose:
            n = self.grid_size ** 2
            print(f"[NavigateAgent] Step 1 – tiling region into {self.grid_size}×{self.grid_size} "
                  f"grid ({n} sub-patches) …")
        subpatches = self._tile_region(region_image)

        # ── Step 2: VLM describes each sub-patch ─────────────────────────────
        if self.verbose:
            print("[NavigateAgent] Step 2 – generating per-sub-patch descriptions via VLM …")
        descriptions = self._describe_subpatches(subpatches)

        # ── Step 3: text LLM plans viewing path ──────────────────────────────
        if self.verbose:
            print("[NavigateAgent] Step 3 – LLM planning viewing path …")
        plan = self._plan_from_descriptions(descriptions, tissue_type, question, observation_points)

        # ── Step 4: convert plan → standard step format ───────────────────────
        steps = self._plan_to_steps(plan, subpatches, region_bbox, wsi_size)
        steps = self._validate_steps(steps, wsi_size=wsi_size)

        # Fallback: LLM returned an empty plan – examine one random sub-patch at max magnification
        if not steps:
            import random
            chosen = random.choice(subpatches)
            if self.verbose:
                print(
                    f"[NavigateAgent] Empty plan – falling back to random sub-patch {chosen[0]} at 5.0×"
                )
            fallback_plan = [{
                "subpatch_id": chosen[0],
                "magnification": 5.0,
                "reasoning": "Fallback: LLM returned empty plan",
                "need_to_see": "General tissue architecture and pathological features",
            }]
            steps = self._plan_to_steps(fallback_plan, subpatches, region_bbox, wsi_size)
            steps = self._validate_steps(steps, wsi_size=wsi_size)

        if self.verbose:
            print(f"[NavigateAgent] {len(steps)} valid steps planned.")

        return steps

    # ── private ───────────────────────────────────────────────────────────────

    def _tile_region(
        self, image: Image.Image
    ) -> List[Tuple[int, float, float, float, float, Image.Image]]:
        """Return list of (id, x1, y1, x2, y2, cropped_image) for a grid_size×grid_size grid."""
        g = self.grid_size
        step = 1.0 / g
        tiles = []
        for r in range(g):
            for c in range(g):
                pid = r * g + c
                x1, y1 = c * step, r * step
                x2, y2 = (c + 1) * step, (r + 1) * step
                patch = crop_view(image, x1, y1, x2, y2, self.subpatch_size)
                tiles.append((pid, x1, y1, x2, y2, patch))
        return tiles

    def _describe_subpatches(
        self,
        subpatches: List[Tuple[int, float, float, float, float, Image.Image]],
    ) -> Dict[int, str]:
        """Call VLM once per sub-patch; return {sub_patch_id: description}."""
        descriptions: Dict[int, str] = {}
        total = len(subpatches)
        g = self.grid_size
        for i, (pid, x1, y1, x2, y2, patch_img) in enumerate(subpatches):
            if self.verbose:
                row, col = pid // g, pid % g
                print(f"  [{i+1}/{total}] Describing sub-patch {pid} (row={row}, col={col}) …")
            desc = self.model.generate_with_images(NAV_SUBPATCH_DESC_USER, [patch_img])
            descriptions[pid] = desc.strip()
            if self.verbose:
                print(f"    → {desc.strip()}")
        return descriptions

    def _plan_from_descriptions(
        self,
        descriptions: Dict[int, str],
        tissue_type: str,
        question: Optional[str],
        observation_points: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Feed all sub-patch descriptions to text LLM; return the plan list."""
        g = self.grid_size
        lines = [
            f"Sub-patch {pid} (row={pid // g}, col={pid % g}): {desc}"
            for pid, desc in sorted(descriptions.items())
        ]
        desc_text = "\n\n".join(lines)

        obs_text = (
            "\n".join(f"- {p}" for p in observation_points)
            if observation_points
            else "- General tissue architecture, cellular density, and nuclear morphology"
        )

        total = g * g
        grid_kwargs = dict(
            grid_size=g,
            total_patches=total,
            max_id=total - 1,
            observation_points=obs_text,
            subpatch_descriptions=desc_text,
            max_steps=self.max_steps,
        )
        if question:
            prompt = NAV_GRID_PLAN_VQA_USER.format(question=question, **grid_kwargs)
        else:
            prompt = NAV_GRID_PLAN_USER.format(**grid_kwargs)

        raw = self.model.generate_text(prompt)

        if self.verbose:
            print("[NavigateAgent] LLM plan output (first 400 chars):\n", raw[:400], "…")

        return self._parse_plan(raw)

    def _parse_plan(self, raw: str) -> List[Dict[str, Any]]:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()

        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("plan", [])
            except json.JSONDecodeError:
                pass

        if self.verbose:
            print("[NavigateAgent] JSON parse failed – asking text model to repair …")
        repair_prompt = JSON_EXTRACT_PROMPT.format(raw_text=raw)
        repaired = self.model.generate_text(repair_prompt)
        cleaned2 = re.sub(r"```(?:json)?", "", repaired).strip()
        match2 = re.search(r"\{.*\}", cleaned2, flags=re.DOTALL)
        if match2:
            try:
                data = json.loads(match2.group())
                return data.get("plan", [])
            except json.JSONDecodeError:
                pass

        return []

    def _plan_to_steps(
        self,
        plan: List[Dict[str, Any]],
        subpatches: List[Tuple[int, float, float, float, float, Image.Image]],
        region_bbox: Optional[List[float]],
        wsi_size: Optional[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        """
        Convert LLM plan entries to the standard step dict format.

        Magnification is applied here: region_coordinate stores the actual
        zoomed crop area (sub-patch centre ± sub-patch_size / (2 * mag)).

        Output coordinate space:
          - wsi_size provided : absolute pixel coords  [px1, py1, px2, py2]
          - region_bbox only  : WSI-normalised [0, 1]
          - neither           : region-relative [0, 1]
        """
        rel_map = {pid: (x1, y1, x2, y2) for pid, x1, y1, x2, y2, _ in subpatches}

        steps = []
        for i, item in enumerate(plan):
            pid = item.get("subpatch_id")
            if pid is None or pid not in rel_map:
                continue
            rx1, ry1, rx2, ry2 = rel_map[pid]
            mag = float(item.get("magnification", 2.0))

            if region_bbox is not None:
                bx1, by1, bx2, by2 = region_bbox
                bw, bh = bx2 - bx1, by2 - by1
                # Map sub-patch to WSI-normalised coords
                nx1 = bx1 + rx1 * bw
                ny1 = by1 + ry1 * bh
                nx2 = bx1 + rx2 * bw
                ny2 = by1 + ry2 * bh
                if wsi_size is not None:
                    W, H = wsi_size
                    # Convert to absolute pixels then apply magnification zoom
                    px1, py1, px2, py2 = nx1 * W, ny1 * H, nx2 * W, ny2 * H
                    cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                    hw = (px2 - px1) / (2.0 * mag)
                    hh = (py2 - py1) / (2.0 * mag)
                    coord = [
                        int(max(0.0, cx - hw)),
                        int(max(0.0, cy - hh)),
                        int(min(float(W), cx + hw)),
                        int(min(float(H), cy + hh)),
                    ]
                else:
                    # Stay in WSI-normalised space, apply magnification zoom
                    cx, cy = (nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0
                    hw = (nx2 - nx1) / (2.0 * mag)
                    hh = (ny2 - ny1) / (2.0 * mag)
                    coord = [
                        max(0.0, cx - hw),
                        max(0.0, cy - hh),
                        min(1.0, cx + hw),
                        min(1.0, cy + hh),
                    ]
            else:
                # Region-relative [0, 1], apply magnification zoom
                cx, cy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
                hw = (rx2 - rx1) / (2.0 * mag)
                hh = (ry2 - ry1) / (2.0 * mag)
                coord = [
                    max(0.0, cx - hw),
                    max(0.0, cy - hh),
                    min(1.0, cx + hw),
                    min(1.0, cy + hh),
                ]

            steps.append({
                "step": i + 1,
                "magnification": mag,
                "region_coordinate": coord,
                "reasoning": str(item.get("reasoning", "")),
                "need_to_see": str(item.get("need_to_see", "")),
            })
        return steps

    @staticmethod
    def _validate_steps(
        steps: List[Dict[str, Any]],
        wsi_size: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Clamp coordinates and magnification to valid ranges."""
        if wsi_size is not None:
            W, H = wsi_size
            max_x, max_y = float(W), float(H)
            min_side_x, min_side_y = 1.0, 1.0  # at least 1 pixel
        else:
            max_x, max_y = 1.0, 1.0
            min_side_x = min_side_y = 0.0625

        valid = []
        for s in steps:
            if not isinstance(s, dict):
                continue
            coords = s.get("region_coordinate", [0.0, 0.0, max_x, max_y])
            if len(coords) != 4:
                coords = [0.0, 0.0, max_x, max_y]
            x1 = max(0.0, min(max_x, float(coords[0])))
            y1 = max(0.0, min(max_y, float(coords[1])))
            x2 = max(0.0, min(max_x, float(coords[2])))
            y2 = max(0.0, min(max_y, float(coords[3])))
            if x2 <= x1:
                x2 = min(max_x, x1 + min_side_x)
            if y2 <= y1:
                y2 = min(max_y, y1 + min_side_y)

            mag = float(s.get("magnification", 2.0))
            mag = max(2.0, min(5.0, mag))

            valid.append({
                "step": int(s.get("step", len(valid) + 1)),
                "magnification": mag,
                "region_coordinate": [x1, y1, x2, y2],
                "reasoning": str(s.get("reasoning", "")),
                "need_to_see": str(s.get("need_to_see", "")),
            })
        return valid
