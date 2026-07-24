"""
Supervisor Agent
----------------
Central coordinator of SlideSeek.
Supports two modes:
  - diagnosis mode  : open-ended differential diagnosis (original)
  - qa mode         : answer a specific question about the slide (QA dataset evaluation)
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from slideseek.models.llm import OllamaLLM, HFLLM, get_hfllm, get_hf_vision_llm, get_openaillm
from slideseek.agents.patch_registry import get_registry
from slideseek.config import SlideSeekConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data structures                                                     #
# ------------------------------------------------------------------ #
@dataclass
class ExplorerTask:
    task_id: str
    name: str
    description: str
    context: str
    target_regions: list[dict]
    features_to_assess: list[str]


@dataclass
class SupervisorState:
    hypotheses: str = ""
    plan: str = ""
    current_step: str = ""
    tasks: list[ExplorerTask] = field(default_factory=list)
    justifications: str = ""
    finished: bool = False
    iteration: int = 0
    all_findings: list[str] = field(default_factory=list)
    key_rois: list[dict] = field(default_factory=list)


# ------------------------------------------------------------------ #
#  Diagnosis-mode prompts                                              #
# ------------------------------------------------------------------ #
SUPERVISOR_SYSTEM_PROMPT = """# ROLE: SUPERVISING PATHOLOGIST
You are supervising a team of expert pathologists examining histopathology slides.
Your goal is to coordinate their exploration systematically to accurately complete your diagnostic task.

## TASK
Your task is to diagnose the disease in this whole-slide image.
Provide the most likely primary diagnosis. Please also suggest two other possible diagnoses to consider.

## SLIDE OVERVIEW
{slide_overview}

## GUIDANCE
- Review the slide overview carefully.
- Identify tissue regions and prioritize based on relevance.
- Define a clear, sequential exploration strategy specifying:
  - Exact regions (x, y coordinates from the tissue bounding boxes)
  - Magnifications: low-magnification (1.25-2.5x), medium-magnification (5-10x), high-magnification (20-40x)
  - Key morphological features to assess
- Continually update your hypothesis as findings arrive.
- Conclude only when confident the slide has been sufficiently explored.

## RESPONSE FORMAT
Always respond with a valid JSON object:
{
  "hypotheses": "string",
  "plan": "string",
  "current_step": "string",
  "tasks": [
    {
      "task_id": "task_1",
      "name": "Low-magnification survey of tissue regions",
      "description": "string",
      "context": "string",
      "target_regions": [{"x": [9500, 10500], "y": [7500, 8500],, "magnification": 1.25, "description": "..."}],
      "features_to_assess": ["nuclear pleomorphism", "architecture", "..."]
    }
  ],
  "justifications": "string",
  "finished": false
}
Set "finished": true only when sufficient high-magnification evidence is collected.
"""

ROI_SELECTION_PROMPT = """Based on your exploration of the slide, identify up to 10 regions of interest (ROIs)
that contain morphological features most relevant to diagnosis.

Explorer findings so far:
{findings}

Select the most diagnostically relevant ROIs. Respond with JSON:
{{
  "selected_rois": [
    {{"x": int, "y": int, "magnification": float, "reason": "why this ROI was selected"}}
  ]
}}

Avoid selecting normal morphology or overlapping regions. Maximum 10 ROIs."""

REPORT_GENERATION_PROMPT = """# COMPREHENSIVE PATHOLOGY REPORT GENERATION

Based on the slide exploration and the PathChat differential diagnosis, generate a structured pathology report.

## Context
Tissue site: {tissue_site}
Patient sex: {patient_sex}
Clinical context: {clinical_context}

## Explorer Findings
{findings}

## PathChat Differential Diagnosis
{differential}

## Instructions
Generate a report with exactly these three sections:

### 1. MICROSCOPIC FINDINGS
- Summarize key morphological features observed across the slide
- Reference specific ROIs by name (e.g., 'region-mag_20-x_1000-y_2000')
- Include architectural patterns, cellular characteristics, nuclear features, stroma

### 2. DIFFERENTIAL DIAGNOSIS
- Primary diagnosis (from PathChat)
- Two differential diagnoses
- For each, summarize supporting morphological evidence from ROIs

### 3. CRITICAL ASSESSMENT
- Evaluate if the diagnosis is well-supported
- Note inconsistencies or alternative diagnoses
- Recommend additional testing if needed
- Confidence rating: HIGH CONFIDENCE or LOW CONFIDENCE
- Justify with specific morphological references
"""


# ------------------------------------------------------------------ #
#  QA-mode prompts (tumor detection / localization)                   #
# ------------------------------------------------------------------ #
QA_SUPERVISOR_SYSTEM_PROMPT = """Supervising pathologist. Answer: {question}
Output valid JSON only:
{{"hypotheses":"...","plan":"...","current_step":"...","tasks":[{{"task_id":"task_1","name":"...","description":"...","context":"...","target_regions":[{{"x":int,"y":int,"magnification":float}}],"features_to_assess":["..."]}}],"justifications":"...","finished":false}}
Set finished=true when confident.
"""

QA_ANSWER_PROMPT = """Question: {question}
Findings: {findings}
JSON: {{"presence":"Yes"or"No","answer_text":"string","num_lesions":int,"predicted_lesions":[{{"center":{{"x":float,"y":float}},"bbox":{{"xmin":float,"ymin":float,"xmax":float,"ymax":float}},"confidence":"high"or"low","description":"string"}}],"confidence":"HIGH"or"LOW","reasoning":"string"}}
presence=No → num_lesions=0, predicted_lesions=[].
"""

QA_SUBTYPE_ANSWER_PROMPT = """Classify breast tumor (IDC or ILC).
Question: {question}
Findings: {findings}
IDC: irregular glands, sheets/nests, desmoplastic stroma. ILC: single-file pattern, small uniform cells, targetoid.
JSON: {{"presence":"IDC"or"ILC","answer_text":"IDC"or"ILC","num_lesions":0,"predicted_lesions":[],"confidence":"HIGH"or"LOW","reasoning":"string"}}
"""


# ------------------------------------------------------------------ #
#  Supervisor Agent                                                    #
# ------------------------------------------------------------------ #
class SupervisorAgent:
    """
    The supervisor agent manages high-level strategy.
    Uses ollama qwen3:4b for planning and reasoning.
    Supports diagnosis mode and QA mode.
    """

    def __init__(self, config: Optional[SlideSeekConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self._use_vision = self.config.use_vision_supervisor
        # Start with text LLM; analyze_thumbnail() will temporarily load VLM
        if self.config.use_openai_llm:
            self.llm = get_openaillm(self.config.openai_llm)
        elif self.config.use_hf_llm:
            self.llm = get_hfllm(self.config.hf_llm)
        else:
            self.llm = OllamaLLM(self.config.ollama)
        self._thumbnail = None   # set by pipeline before first use
        self.state = SupervisorState()
        self._qa_mode = False
        self._qa_question = ""
        self._qa_slide_id = ""

    # ------------------------------------------------------------------ #
    #  Initialization                                                      #
    # ------------------------------------------------------------------ #
    def initialize(
        self,
        slide_context: str,
        tissue_site: str,
        patient_sex: str,
        clinical_context: str = "No additional clinical context provided.",
    ):
        """Diagnosis mode: open-ended differential diagnosis."""
        self._slide_context = slide_context
        self._tissue_site = tissue_site
        self._patient_sex = patient_sex
        self._clinical_context = clinical_context
        self._qa_mode = False
        self._qa_question = ""
        self.state = SupervisorState()

    def initialize_qa(
        self,
        slide_context: str,
        question: str,
        slide_id: str = "",
    ):
        """
        QA evaluation mode: answer a specific question about the slide.
        Used for the roi_qa_dataset benchmark.
        """
        self._slide_context = slide_context
        self._tissue_site = "unknown"
        self._patient_sex = "unknown"
        self._clinical_context = f"QA evaluation task"
        self._qa_question = question
        self._qa_slide_id = slide_id
        self._qa_mode = True
        self.state = SupervisorState()

    # ------------------------------------------------------------------ #
    #  Step 1: Vision thumbnail analysis (call once before main loop)     #
    # ------------------------------------------------------------------ #
    def analyze_thumbnail(self) -> str:
        """
        Step 1: Load VLM, analyze the slide thumbnail, store initial hypotheses,
        then unload VLM and switch self.llm to the text LLM (Qwen3-4B).
        Call this once after initialize()/initialize_qa() and before plan_next_iteration().
        Returns the initial impression string.
        """
        if not self._use_vision or self._thumbnail is None:
            logger.info("Vision supervisor disabled or no thumbnail — skipping step 1.")
            return ""

        from PIL import Image
        import numpy as np

        logger.info("Step 1: Loading VLM for thumbnail analysis...")
        vlm = get_hf_vision_llm(self.config.hf_vision_llm)

        thumb = self._thumbnail
        if isinstance(thumb, np.ndarray):
            thumb = Image.fromarray(thumb.astype("uint8"))

        if self._qa_mode:
            prompt = (
                f"You are a pathologist. Look at this whole-slide image thumbnail and give an initial "
                f"impression relevant to this question: {self._qa_question}\n"
                f"Slide context: {self._slide_context}\n"
                f"Respond concisely with your initial observations and hypotheses."
            )
        else:
            prompt = (
                f"You are a pathologist. Look at this whole-slide image thumbnail.\n"
                f"Tissue site: {self._tissue_site}, Patient sex: {self._patient_sex}\n"
                f"Slide context: {self._slide_context}\n"
                f"Give an initial impression: tissue type, overall architecture, any visible abnormalities."
            )

        response = vlm.chat(
            messages=[{"role": "user", "content": prompt}],
            system="You are an expert pathologist analyzing a whole-slide image thumbnail.",
            images=[thumb],
        )

        logger.info(f"Step 1 thumbnail impression: {response[:200]}...")

        # Store as initial hypothesis
        self.state.hypotheses = f"[Thumbnail analysis] {response}"
        logger.info("Step 1 done.")

        return response

    # ------------------------------------------------------------------ #
    #  Main planning step                                                  #
    # ------------------------------------------------------------------ #
    def plan_next_iteration(self) -> SupervisorState:
        """Generate the next exploration plan based on current findings."""
        self.state.iteration += 1
        max_iter = self.config.agent.max_supervisor_iterations
        logger.info(f"Supervisor iteration {self.state.iteration}/{max_iter}")

        findings_text = self._format_findings()

        if self._qa_mode:
            system = QA_SUPERVISOR_SYSTEM_PROMPT.format(question=self._qa_question)
            context_block = f"QUESTION TO ANSWER: {self._qa_question}"
        else:
            system = SUPERVISOR_SYSTEM_PROMPT.format(slide_overview=self._slide_context)
            context_block = (
                # f"Tissue site: {self._tissue_site}\n"
                # f"Patient sex: {self._patient_sex}\n"
                f"Clinical context: {self._clinical_context}"
            )

        visited_summary = get_registry().summary()
        user_message = f"""## SLIDE OVERVIEW
{self._slide_context}

## CONTEXT
{context_block}

## ALREADY VISITED PATCHES (DO NOT assign tasks to these coordinates — they have been fully examined)
{visited_summary}

## EXPLORER FINDINGS SO FAR
{findings_text if findings_text else "No findings yet — this is the initial exploration."}

## CURRENT HYPOTHESES
{self.state.hypotheses if self.state.hypotheses else "Not yet formulated."}

## ITERATION {self.state.iteration} of {max_iter}
{"WARNING: This is the final iteration. You MUST set finished=true." if self.state.iteration >= max_iter else "Continue exploration if more evidence is needed."}

Now update your hypotheses and generate the next set of tasks (or set finished=true if done)."""

        response = self.llm.chat_json(
            messages=[{"role": "user", "content": user_message}],
            system=system,
        )

        self._parse_supervisor_response(response)
        return self.state

    # ------------------------------------------------------------------ #
    #  Receive explorer findings                                           #
    # ------------------------------------------------------------------ #
    def add_explorer_finding(self, task_id: str, finding: str, key_rois: list[dict] = None):
        entry = f"[Task {task_id}] {finding}"
        self.state.all_findings.append(entry)
        if key_rois:
            self.state.key_rois.extend(key_rois)
        logger.info(f"Supervisor received finding from task {task_id}")

    # ------------------------------------------------------------------ #
    #  QA answer generation                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_subtype_question(question: str) -> bool:
        q = question.lower()
        return "idc" in q and "ilc" in q

    @staticmethod
    def _normalize_subtype(text: str) -> str:
        """Extract 'IDC' or 'ILC' from any model response string."""
        t = text.upper()
        if "ILC" in t:
            return "ILC"
        if "IDC" in t:
            return "IDC"
        return "IDC"  # default fallback

    def generate_qa_answer(self) -> dict:
        """
        Generate a structured answer to the QA question based on all explorer findings.
        Returns a dict with: presence, answer_text, num_lesions, predicted_lesions,
                             confidence, reasoning.
        """
        findings_text = self._format_findings()
        is_subtype = self._is_subtype_question(self._qa_question)

        if is_subtype:
            prompt = QA_SUBTYPE_ANSWER_PROMPT.format(
                question=self._qa_question,
                findings=findings_text,
            )
        else:
            prompt = QA_ANSWER_PROMPT.format(
                question=self._qa_question,
                findings=findings_text,
            )

        response = self.llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are an expert pathologist answering questions about whole-slide images. "
                "Respond only with valid JSON."
            ),
        )

        presence = response.get("presence", "No")
        answer_text = response.get("answer_text", "")

        # For subtype questions, normalize presence and answer_text to exactly "IDC" or "ILC"
        if is_subtype:
            combined = f"{presence} {answer_text} {response.get('reasoning', '')}"
            normalized = self._normalize_subtype(combined)
            presence = normalized
            answer_text = normalized

        result = {
            "presence": presence,
            "answer_text": answer_text,
            "num_lesions": int(response.get("num_lesions", 0)),
            "predicted_lesions": response.get("predicted_lesions", []),
            "confidence": response.get("confidence", "LOW"),
            "reasoning": response.get("reasoning", ""),
        }
        return result

    # ------------------------------------------------------------------ #
    #  ROI selection (diagnosis mode)                                      #
    # ------------------------------------------------------------------ #
    def select_final_rois(self) -> list[dict]:
        max_rois = self.config.slide.max_rois_final
        findings_text = self._format_findings()

        prompt = ROI_SELECTION_PROMPT.format(findings=findings_text)
        response = self.llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            system="You are an expert pathologist selecting the most diagnostically relevant tissue regions.",
        )

        selected = response.get("selected_rois", [])[:max_rois]

        merged = {(r["x"], r["y"], r["magnification"]): r for r in selected}
        for roi in self.state.key_rois:
            key = (roi.get("x"), roi.get("y"), roi.get("magnification", 20.0))
            if key not in merged and len(merged) < max_rois:
                merged[key] = roi

        final = list(merged.values())[:max_rois]
        logger.info(f"Selected {len(final)} final ROIs for PathChat diagnosis")
        return final

    # ------------------------------------------------------------------ #
    #  Report generation (diagnosis mode)                                  #
    # ------------------------------------------------------------------ #
    def generate_report(self, differential: str) -> str:
        findings_text = self._format_findings()
        prompt = REPORT_GENERATION_PROMPT.format(
            tissue_site=self._tissue_site,
            patient_sex=self._patient_sex,
            clinical_context=self._clinical_context,
            findings=findings_text,
            differential=differential,
        )
        return self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are an expert surgical pathologist writing a structured diagnostic report. "
                "Be precise, evidence-based, and reference specific ROIs from the exploration."
            ),
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _format_findings(self) -> str:
        if not self.state.all_findings:
            return "No findings yet."
        return "\n\n".join(
            f"Finding {i+1}:\n{f}" for i, f in enumerate(self.state.all_findings)
        )

    def _parse_supervisor_response(self, response: dict):
        if "raw" in response and len(response) == 1:
            logger.warning("Supervisor returned unparseable response.")
            return

        self.state.hypotheses = response.get("hypotheses", self.state.hypotheses)
        self.state.plan = response.get("plan", self.state.plan)
        self.state.current_step = response.get("current_step", "")
        self.state.justifications = response.get("justifications", "")
        self.state.finished = bool(response.get("finished", False))

        raw_tasks = response.get("tasks", [])
        self.state.tasks = []
        for t in raw_tasks:
            self.state.tasks.append(ExplorerTask(
                task_id=t.get("task_id", f"task_{self.state.iteration}"),
                name=t.get("name", "Explore region"),
                description=t.get("description", ""),
                context=t.get("context", self.state.hypotheses),
                target_regions=t.get("target_regions", []),
                features_to_assess=t.get("features_to_assess", []),
            ))

        logger.info(
            f"Supervisor: {len(self.state.tasks)} tasks, "
            f"finished={self.state.finished}, "
            f"hypotheses='{self.state.hypotheses[:80]}...'"
        )
