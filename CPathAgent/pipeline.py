"""
CPathAgent full pipeline.

Orchestrates the three stages for a single WSI:
  1. TraceAgent   – global screening
  2. NavigateAgent – navigation planning per region
  3. ReasoningAgent – multi-scale sequence reasoning per region

Can also run in single-region mode (no WSI partitioning) when given a
pre-cropped region image directly.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from config import CPathAgentConfig
from models import build_model
from utils import WSILoader, create_thumbnail_with_ids
from agents import TraceAgent, NavigateAgent, ReasoningAgent, PatchScreeningAgent


class CPathAgentPipeline:

    def __init__(self, cfg: CPathAgentConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Pipeline] Backend: {cfg.model.backend.upper()}")
        self.model = build_model(cfg.model)

        wsi_cfg = cfg.wsi
        self.trace_agent = TraceAgent(self.model, verbose=cfg.verbose)
        self.patch_screening_agent = PatchScreeningAgent(
            self.model,
            patch_input_size=wsi_cfg.model_input_size,
            verbose=cfg.verbose,
        )
        self.navigate_agent = NavigateAgent(
            self.model,
            max_steps=wsi_cfg.max_nav_steps,
            grid_size=2,
            subpatch_size=wsi_cfg.model_input_size,
            verbose=cfg.verbose,
        )
        self.reasoning_agent = ReasoningAgent(
            self.model,
            model_input_size=wsi_cfg.model_input_size,
            verbose=cfg.verbose,
        )

    # ── full WSI pipeline ─────────────────────────────────────────────────────

    def run_wsi(
        self,
        wsi_path: str,
        tissue_type: str = "unknown",
        question: Optional[str] = None,
        save_prefix: str = "result",
    ) -> Dict[str, Any]:
        """
        Run the complete 3-stage CPathAgent pipeline on a WSI file.

        Returns a result dict with keys:
          - screening      : TraceAgent output
          - regions        : list of per-region results (nav_steps + reasoning)
          - final_report   : concatenated pathological reports
        """
        cfg = self.cfg.wsi

        with WSILoader(wsi_path, thumbnail_scale=cfg.thumbnail_scale) as loader:
            thumbnail = loader.get_thumbnail()
            # thumbnail.save(self.out_dir / f"{save_prefix}_thumbnail.png")
            tiles = loader.partition_grid(
                patch_size_px=cfg.patch_size_px,
                overlap=cfg.overlap_ratio,
            )

            # ── Stage 1: Global Screening (patch-describe → cluster → ROI) ─────
            print("\n" + "=" * 60)
            print("STAGE 1 – Patch Screening (describe → cluster → ROI)")
            print("=" * 60)
            screening = self.patch_screening_agent.run(thumbnail, tiles, tissue_type)
            high_priority = self.patch_screening_agent.get_high_priority_regions(screening)

            if self.cfg.save_images:
                annotated_thumb = create_thumbnail_with_ids(thumbnail, tiles)
                annotated_thumb.save(self.out_dir / f"{save_prefix}_thumbnail.png")

            print(f"[Pipeline] {len(high_priority)} high-priority regions selected.")

            # ── Stage 2 per region ─────────────────────────────────────────────
            region_results = []
            all_region_imgs: List[Image.Image] = []
            all_nav_steps: List[List[Dict[str, Any]]] = []

            for gi, group in enumerate(high_priority):
                group_name = group.get("name", f"group_{gi}")
                print(f"\n{'=' * 60}")
                print(f"REGION {gi+1}: {group_name}")
                print("=" * 60)

                # Take the first patch from the group as the representative region
                patch_ids = group.get("id_list", [])
                if not patch_ids:
                    continue

                pid = patch_ids[0]
                tile = next((t for t in tiles if t[0] == pid), None)
                if tile is None:
                    continue

                _, x1, y1, x2, y2 = tile
                region_img = loader.get_region(x1, y1, x2, y2, cfg.model_input_size)

                if self.cfg.save_images:
                    region_img.save(self.out_dir / f"{save_prefix}_region_{gi}.png")

                # Stage 2: Navigation Planning
                print(f"\nSTAGE 2 – Navigation Planning for region {gi+1}")
                nav_steps = self.navigate_agent.run(
                    region_img,
                    tissue_type=tissue_type,
                    question=question,
                    region_bbox=[x1, y1, x2, y2],
                    wsi_size=(loader.width, loader.height),
                    observation_points=group.get("observation_points", []),
                )

                # Attach Step 1 observation_points to each nav step so
                # the Stage 3 VLM knows what to focus on in this region.
                obs_points = group.get("observation_points", [])
                for step in nav_steps:
                    step.setdefault("observation_points", obs_points)

                # Save navigation thumbnail
                if self.cfg.save_images:
                    self._save_nav_thumbnail(
                        region_img, nav_steps, save_prefix, gi,
                        region_bbox=[x1, y1, x2, y2],
                        wsi_size=(loader.width, loader.height),
                    )

                all_region_imgs.append(region_img)
                all_nav_steps.append(nav_steps)

                region_results.append({
                    "group": group,
                    "region_bbox": [x1, y1, x2, y2],
                    "nav_steps": nav_steps,
                })

            # ── Stage 3: Reasoning (once, all regions together) ────────────────
            print(f"\nSTAGE 3 – Multi-scale Reasoning ({len(all_region_imgs)} regions)")
            if not all_region_imgs:
                # Stage 1 produced no regions (e.g. clustering failed entirely).
                # Fall back to using the full thumbnail with a single overview step
                # so Stage 3 still has at least one view and can produce an answer.
                print("[Pipeline] No regions found – falling back to thumbnail for Stage 3.")
                all_region_imgs = [thumbnail]
                all_nav_steps = [[{
                    "step": 1,
                    "magnification": 1.0,
                    "region_coordinate": [0.0, 0.0, 1.0, 1.0],
                    "reasoning": "Full thumbnail overview (fallback – Stage 1 produced no groups)",
                    "need_to_see": "Overall tissue architecture and any pathological changes",
                }]]

            reasoning_result = self.reasoning_agent.run(
                all_region_imgs,
                all_nav_steps,
                tissue_type=tissue_type,
                question=question,
                wsi_loader=loader,
            )

        # ── Compile final report ───────────────────────────────────────────────
        final_report = self._compile_report(region_results, tissue_type, reasoning_result)

        full_result = {
            "wsi_path": wsi_path,
            "tissue_type": tissue_type,
            "question": question,
            "screening": screening,
            "regions": region_results,
            "reasoning": reasoning_result,
            "final_report": reasoning_result["reasoning"],
        }

        # Save JSON result
        out_json = self.out_dir / f"{save_prefix}_result.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(full_result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[Pipeline] Saved results to {out_json}")

        return full_result

    # ── single region mode ────────────────────────────────────────────────────

    def run_region(
        self,
        region_path: str,
        tissue_type: str = "unknown",
        question: Optional[str] = None,
        save_prefix: str = "region_result",
    ) -> Dict[str, Any]:
        """
        Run stages 2 + 3 on a single pre-cropped region image.
        Useful for evaluating on the PathMMU-HR² benchmark.
        """
        region_img = Image.open(region_path).convert("RGB")

        print(f"\n[Pipeline] Region mode: {region_path}")

        # Stage 2
        print("\nSTAGE 2 – Navigation Planning")
        nav_steps = self.navigate_agent.run(
            region_img, tissue_type=tissue_type, question=question
        )

        if self.cfg.save_images:
            self._save_nav_thumbnail(region_img, nav_steps, save_prefix, 0)

        # Stage 3
        print("\nSTAGE 3 – Multi-scale Reasoning")
        reasoning_result = self.reasoning_agent.run(
            region_img, nav_steps, tissue_type=tissue_type, question=question
        )

        result = {
            "region_path": region_path,
            "tissue_type": tissue_type,
            "question": question,
            "nav_steps": nav_steps,
            **reasoning_result,
        }

        out_json = self.out_dir / f"{save_prefix}_result.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[Pipeline] Saved results to {out_json}")

        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    def _save_nav_thumbnail(
        self,
        region_img: Image.Image,
        nav_steps: List[Dict[str, Any]],
        prefix: str,
        region_idx: int,
        region_bbox: Optional[List[float]] = None,
        wsi_size: Optional[tuple] = None,
    ):
        from PIL import ImageDraw
        img = region_img.copy()
        draw = ImageDraw.Draw(img)
        iW, iH = img.size
        colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]

        for s in nav_steps:
            x1, y1, x2, y2 = s["region_coordinate"]
            c = colors[s["step"] % len(colors)]

            if wsi_size is not None and region_bbox is not None:
                # coords are absolute WSI pixels → convert to region-image pixels
                W_wsi, H_wsi = wsi_size
                bx1, by1, bx2, by2 = region_bbox
                bw = (bx2 - bx1) * W_wsi  # region width in WSI pixels
                bh = (by2 - by1) * H_wsi
                ox = bx1 * W_wsi          # region origin in WSI pixels
                oy = by1 * H_wsi
                rx1 = (x1 - ox) / bw if bw else 0
                ry1 = (y1 - oy) / bh if bh else 0
                rx2 = (x2 - ox) / bw if bw else 1
                ry2 = (y2 - oy) / bh if bh else 1
            else:
                # coords are region-relative [0, 1]
                rx1, ry1, rx2, ry2 = x1, y1, x2, y2

            draw.rectangle(
                [int(rx1 * iW), int(ry1 * iH), int(rx2 * iW), int(ry2 * iH)],
                outline=c, width=2,
            )
            draw.text(
                (int(rx1 * iW) + 3, int(ry1 * iH) + 3),
                str(s["step"]), fill=c,
            )

        img.save(self.out_dir / f"{prefix}_region_{region_idx}_nav.png")

    @staticmethod
    def _compile_report(
        region_results: List[Dict[str, Any]],
        tissue_type: str,
        reasoning_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        header = f"## Pathological Summary – {tissue_type.title()} Tissue\n\n"

        if reasoning_result:
            rp = reasoning_result.get("report", "")
            if rp:
                return header + rp

        # Fallback: collect any per-region reports
        reports = []
        for i, r in enumerate(region_results):
            rp = r.get("report", "")
            if rp:
                name = r.get("group", {}).get("name", f"Region {i+1}")
                reports.append(f"### {name}\n{rp}")

        if not reports:
            return "No pathological report generated."

        return header + "\n\n".join(reports)
