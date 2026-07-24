"""
Explorer Agent
--------------
Performs tasks assigned by the supervisor agent.
Responsibilities (matching paper Extended Data Figure 5):
  1. Receive task + context from supervisor
  2. LLM (text) receives ROI description + whole-slide coordinate range and decides
     which coordinates and magnification to examine next
  3. VLM (patho_r1) captions each navigated ROI; absolute positions are recorded
  4. Recorded observations (position + description) are fed back to the LLM
  5. Submit a structured report back to the supervisor
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from slideseek.models.llm import OllamaLLM, HFLLM, get_hfllm, get_openaillm
from slideseek.models import patho_model
from slideseek.agents.supervisor import ExplorerTask
from slideseek.agents.patch_registry import get_registry
from slideseek.wsi.slide_viewer import SlideViewer, ROI
from slideseek.config import SlideSeekConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Data structures                                                     #
# ------------------------------------------------------------------ #
@dataclass
class NavigationRequest:
    x: int
    y: int
    magnification: float
    rationale: str


@dataclass
class ExplorerFinding:
    task_id: str
    text_report: str
    key_rois: list[dict] = field(default_factory=list)
    visited_rois: list[ROI] = field(default_factory=list)
    roi_descriptions: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
#  Prompts (matching Extended Data Figure 5)                          #
# ------------------------------------------------------------------ #
EXPLORER_PROMPT = """You are a pathology exploration agent. Output ONLY valid JSON (start with '{{', end with '}}').

Navigate if more evidence needed:
{{"action":"navigate","navigation_requests":[{{"x":int,"y":int,"magnification":float,"rationale":"string"}}],"interim_findings":"string"}}

Submit when sufficient:
{{"action":"submit_report","findings":{{"examined_regions":[{{"roi_name":"string","description":"string"}}],"key_morphological_features":["..."],"abnormalities":["..."],"overall_impression":"string","key_rois":[{{"x":int,"y":int,"magnification":float,"reason":"string"}}]}}}}

# RECORDED OBSERVATIONS
{roi_descriptions}

# ALREADY VISITED (do not revisit)
{globally_visited}
"""


EXPLORER_INITIAL_PROMPT_TEMPLATE = """Task: {name}
Context: {context}
Instructions: {description}
Slide info: {slide_context}
Target regions: {target_regions}
Features: {features}"""


def _format_target_regions(regions: list[dict]) -> str:
    if not regions:
        return "No specific coordinates given — use judgment based on tissue bounding boxes."
    lines = []
    for r in regions:
        lines.append(
            f"  - x={r.get('x', '?')}, y={r.get('y', '?')}, "
            f"magnification={r.get('magnification', 20.0)}x"
            f"{' — ' + r['description'] if 'description' in r else ''}"
        )
    return "\n".join(lines)


def _format_observations(observations: list[dict]) -> str:
    """Format all recorded observations (with absolute positions) for the LLM."""
    if not observations:
        return "No regions examined yet."
    lines = []
    for obs in observations:
        desc = obs['description']
        if len(desc) > 200:
            desc = desc[:200] + "..."
        lines.append(
            f"{obs['roi_name']} x={obs['absolute_x']} y={obs['absolute_y']} @{obs['magnification']}x: {desc}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Explorer Agent                                                      #
# ------------------------------------------------------------------ #
class ExplorerAgent:
    """
    Single explorer agent that investigates a specific task from the supervisor.

    Flow per iteration:
      1. Text LLM receives ROI task description + whole-slide coordinate range
         and decides which coordinates / magnification to examine.
      2. patho_r1 VLM captions each navigated ROI image.
      3. Absolute positions + VLM descriptions are recorded and fed back to the LLM.
    """

    def __init__(
        self,
        task: ExplorerTask,
        slide_viewer: SlideViewer,
        config: Optional[SlideSeekConfig] = None,
    ):
        self.task = task
        self.slide = slide_viewer
        self.config = config or DEFAULT_CONFIG
        # Text LLM for navigation decisions (respects backend config)
        self.decision_llm = self._get_decision_llm()
        self._visited_rois: list[ROI] = []
        self._roi_descriptions: list[str] = []
        # Structured record of every observed position: {roi_name, absolute_x, absolute_y,
        # magnification, description}
        self._visited_positions: list[dict] = []
        self._conversation: list[dict] = []

    def _get_decision_llm(self):
        """Return the text LLM for navigation decisions based on config backend selection."""
        if self.config.use_openai_llm:
            return get_openaillm(self.config.openai_llm)
        elif self.config.use_hf_llm:
            return get_hfllm(self.config.hf_llm)
        else:
            return OllamaLLM(self.config.ollama)

    # ------------------------------------------------------------------ #
    #  Main exploration loop                                               #
    # ------------------------------------------------------------------ #
    def run(self) -> ExplorerFinding:
        """
        Execute the exploration task until the explorer submits a report.

        Step 1 — LLM receives the task description and the whole-slide coordinate range
                 (tissue bounding boxes) and decides which coordinate + magnification to examine.
        Step 2 — patho_r1 VLM captions the navigated ROI; absolute position is recorded.
        Step 3 — All recorded observations are passed back to the LLM for the next iteration.
        """
        max_iter = self.config.agent.max_explorer_iterations
        iteration = 0

        # System prompt: task description + full slide coordinate range
        system_prompt = EXPLORER_INITIAL_PROMPT_TEMPLATE.format(
            name=self.task.name,
            context=self.task.context,
            description=self.task.description,
            slide_context=self.slide.get_slide_context(),
            target_regions=_format_target_regions(self.task.target_regions),
            features=", ".join(self.task.features_to_assess) or "General morphological assessment",
        )

        registry = get_registry()
        initial_prompt = (
            EXPLORER_PROMPT
            .replace("{roi_descriptions}", "No regions examined yet.")
            .replace("{globally_visited}", registry.summary())
        )
        self._conversation = [{"role": "user", "content": initial_prompt}]

        while iteration < max_iter:
            iteration += 1
            logger.info(
                f"Explorer [{self.task.task_id}] iteration {iteration}/{max_iter}"
            )

            # Step 1: LLM decides which coordinate + magnification to examine
            response = self.decision_llm.chat_json(
                messages=self._conversation,
                system=system_prompt,
            )

            action = response.get("action", "navigate").strip().lower()

            if action == "submit_report" or action == "submitreport":
                return self._build_finding(response)

            elif action == "navigate" or action == "navigating":
                nav_requests = response.get("navigation_requests", [])
                if not nav_requests and self.task.target_regions:
                    # Fall back to initial target regions
                    nav_requests = [
                        {
                            "x": r.get("x", 10000),
                            "y": r.get("y", 10000),
                            "magnification": r.get("magnification", 20.0),
                            "rationale": "Initial target region from supervisor",
                        }
                        for r in self.task.target_regions[:3]
                    ]
                elif not nav_requests:
                    # No regions specified — default to center of first tissue bbox
                    bboxes = self.slide.tissue_bboxes
                    if bboxes:
                        bbox = bboxes[0]
                        nav_requests = [{
                            "x": bbox.center_x,
                            "y": bbox.center_y,
                            "magnification": 20.0,
                            "rationale": "Default center of first tissue region",
                        }]

                # Step 2: patho_r1 VLM captions each ROI; absolute positions recorded
                new_observations = self._process_navigation_with_vlm(nav_requests)

                # Step 3: pass all recorded observations back to LLM
                registry = get_registry()
                followup = (
                    EXPLORER_PROMPT
                    .replace("{roi_descriptions}", _format_observations(self._visited_positions))
                    .replace("{globally_visited}", registry.summary())
                )
                self._conversation = [{"role": "user", "content": followup}]

            else:
                # Unexpected action — treat as done
                logger.warning(f"Explorer got unexpected action: {action}")
                break

        # Max iterations reached — force submit
        logger.warning(
            f"Explorer [{self.task.task_id}] reached max iterations, force-submitting."
        )
        return self._force_submit()

    # ------------------------------------------------------------------ #
    #  Navigation + patho_r1 VLM captioning + absolute position recording  #
    # ------------------------------------------------------------------ #
    def _process_navigation_with_vlm(self, nav_requests: list[dict]) -> list[dict]:
        """
        Navigate to requested ROIs, caption each with patho_r1 VLM,
        and record the absolute slide position of every observation.
        Returns the list of new observation dicts added this turn.
        """
        new_observations = []
        max_per_turn = self.config.slide.max_rois_per_explorer

        registry = get_registry()

        for req in nav_requests[:max_per_turn]:
            x = int(req.get("x", 10000))
            y = int(req.get("y", 10000))
            mag = float(req.get("magnification", 20.0))
            rationale = req.get("rationale", "")

            if registry.is_visited(x, y, mag):
                msg = (
                    f"[GlobalRegistry] Explorer [{self.task.task_id}] "
                    f"skipping already-visited patch ({x}, {y}) @ {mag}x"
                )
                logger.info(msg)
                print(msg)
                continue

            try:
                roi = self.slide.navigate(x, y, mag)
                self._visited_rois.append(roi)

                # patho_r1 VLM reads the ROI image and produces a morphological description
                caption = patho_model.caption_roi(
                    roi.image, config=self.config.patho_model
                )
                self._roi_descriptions.append(caption)

                # Record absolute slide position + VLM description
                obs = {
                    "roi_name": roi.name,
                    "absolute_x": roi.x,
                    "absolute_y": roi.y,
                    "magnification": roi.magnification,
                    "rationale": rationale,
                    "description": caption,
                }
                self._visited_positions.append(obs)
                new_observations.append(obs)

                # Register in global registry so other explorers skip it
                registry.register(x, y, mag, description=caption)
                logger.debug(
                    f"[GlobalRegistry] Registered patch ({x}, {y}) @ {mag}x "
                    f"(total visited: {len(registry)})"
                )
                logger.debug(f"Explorer [{self.task.task_id}] navigated to {roi.name}")

            except Exception as e:
                logger.error(f"Navigation failed at ({x},{y}): {e}")

        return new_observations

    # ------------------------------------------------------------------ #
    #  Build final finding                                                 #
    # ------------------------------------------------------------------ #
    def _build_finding(self, response: dict) -> ExplorerFinding:
        findings_data = response.get("findings", {})

        examined = findings_data.get("examined_regions", [])
        examined_text = "\n".join(
            f"  [{r.get('roi_name', '?')}] {r.get('description', '')}"
            for r in examined
        )
        key_features = findings_data.get("key_morphological_features", [])
        abnormalities = findings_data.get("abnormalities", [])
        impression = findings_data.get("overall_impression", "")

        report_text = (
            f"=== Explorer Report: {self.task.name} ===\n"
            f"Examined regions:\n{examined_text}\n\n"
            f"Key features: {', '.join(key_features)}\n"
            f"Abnormalities: {', '.join(abnormalities)}\n"
            f"Impression: {impression}"
        )

        key_rois = findings_data.get("key_rois", [])

        return ExplorerFinding(
            task_id=self.task.task_id,
            text_report=report_text,
            key_rois=key_rois,
            visited_rois=self._visited_rois,
            roi_descriptions=self._roi_descriptions,
        )

    def _force_submit(self) -> ExplorerFinding:
        """Build a partial report from whatever was collected."""
        if self._visited_positions:
            text = _format_observations(self._visited_positions)
            report = f"=== Explorer Report (partial): {self.task.name} ===\n{text}"
        else:
            report = f"=== Explorer Report (no data): {self.task.name} ==="

        key_rois = []
        for roi in self._visited_rois[:5]:
            key_rois.append({
                "x": roi.x,
                "y": roi.y,
                "magnification": roi.magnification,
                "reason": "Visited during exploration",
            })

        return ExplorerFinding(
            task_id=self.task.task_id,
            text_report=report,
            key_rois=key_rois,
            visited_rois=self._visited_rois,
            roi_descriptions=self._roi_descriptions,
        )
