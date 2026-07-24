"""
Prompt templates for all three CPathAgent stages.

Pure-text post-processing (JSON extraction / repair) uses the TEXT model (Qwen3 4B).
Visual reasoning uses the VLM (patho_r1).
"""

# ── Stage 1 : Global Screening (Trace Agent) ──────────────────────────────────
# Input  : WSI thumbnail annotated with patch-ID grid
# Output : JSON with groups, severity, and high-mag flag

TRACE_SYSTEM = (
    "You are an expert pathologist performing systematic whole-slide image screening. "
    "Analyse the provided WSI thumbnail and group the numbered patches by their visual "
    "pathological characteristics. Follow standard diagnostic workflow."
)

TRACE_USER = """\
This is a thumbnail view of a pathological whole slide image from {tissue_type}.
As an expert pathologist, I need you to analyze the given pathology image which has been divided into multiple patches with pre-drawn IDs.

Please:
1. Design a systematic reading path by grouping the patch IDs based on their visual patterns (should include all patches), each patch ID should only belong to one group
2. Order the groups according to standard pathological diagnostic workflow

Note: Be aware that whole slide images may contain:
- Multiple discrete tissue fragments
- Serial sections from the same tissue block
- Different tissue types/organs on the same slide These variations should inform your grouping strategy and reading path design.

Output ONLY valid JSON in this exact format:
```json
[
{{
  "groups": [
  {{
    "name": "[group name]", ## brief description of this group
    "description": "[description of this region]",
    "id_list": [patch IDs in recommended viewing order],
    "is_background": [true/false],  // = true if the group contains no tissue (empty slide, artifact only)
    "require_high_magnification": [true/false],  // = true if the area require detailed high-power examination to confirm diagnosis or unsure area
    "diagnostic_priority": [1-5],  // Lower number indicates higher examination priority (1 is highest)
    "observation_points": [ "key morphological feature to examine", "key structural pattern to look for", ... ]
  }},
  ...
  ]
}}
]
```
"""

# ── Stage 2 : Navigation Planning (Navigate Agent) ────────────────────────────
# Input  : region image annotated with coordinate grid
# Output : JSON list of viewing steps

NAVIGATE_SYSTEM = (
    "You are an expert pathologist planning a systematic multi-scale examination of a "
    "high-resolution pathology region. Design an optimal viewing path."
)

NAVIGATE_USER = """\
Please carefully examine this pathology image and simulate a pathologist's viewing path.
You can zoom in and out to observe the image. In the default state (1.0x), you see the complete image at 1/5 of its original resolution.

For any zoom level Z:
- Field of view: 1/Z² of the total image area (1/Z of each side length)
- Downscale ratio: 5/Z

Your viewing path steps should:
- Check and Avoid empty/background/unimportant area
- Pay special attention to transition zones
- Selectively analysis areas with representative features (avoid redundant areas)
- Ensure minimal overlap between sampled patches (except when examining the same region at different magnifications) to reduce redundancy
- Demonstrate logical continuity between each step
- movements between steps, including zooming in/out, should follow a systematic and logical pattern
- Consider the blue coordinate grid for position reference (you can specify positions between grid points)

Please provide the viewing path as a JSON list where each step includes:
```json
[ 
  {{ "step": [step number],
  "magnification": [zoom level from 2.0x-5.0x],
  "region_coordinate": [x1, y1, x2, y2], # # Coordinates represent [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
  ## Avoid empty area
  "reasoning": [reasoning for moving to this position],
  "need_to_see": [description of what should be observed at this position]
  }}, 
  ...
]
 ```
"""

# ── Stage 3 : Multi-scale Multi-view Reasoning (Reasoning Agent) ──────────────
# Input  : sequence of cropped views from navigation path
# Output : step-by-step reasoning + **Pathological Report:**

REASONING_SYSTEM = (
    "You are an expert pathologist performing a systematic multi-scale examination. "
    "Think aloud as you analyse each field of view, then synthesise your findings."
)

REASONING_USER = """\
You are examining a {tissue_type} pathology region through the following viewing path:

{step_descriptions}

Each image above corresponds to one step. Analyse each view in sequence using first-person
narrative. Follow this framework:

1. Initial overview – architecture and dominant patterns.
2. Progressive zoom – cellular morphology, nuclear features, stromal reaction.
3. Transition zones – normal-to-abnormal interfaces.
4. Differential reasoning – establish hypothesis, seek evidence, refine.
5. Synthesis – integrate observations across all scales.

Write as a flowing narrative (not bullet points). Record real-time reasoning
including position coordinates and zoom level.

End with a one-paragraph pathology report starting exactly with:
**Pathological Report:**"""

# ── VQA variant of Stage 2 ────────────────────────────────────────────────────

# NAVIGATE_VQA_USER = """\
# Please analyze this pathology image by creating a step-by-step viewing path that simulates how a professional pathologist would examine the specimen to answer the given question.
# Focus specifically on areas relevant to the question and potential pathological changes.

# ## Image Viewing Mechanics
# - The default view (1.0x) shows the complete image at 1/5 of its original resolution
# - You can zoom to any level up to 5.0x (including decimal values like 2.3x)
# - Image coordinates use relative values from 0 to 1
# - At any zoom level Z:
#   * Field of view: 1/Z² of total image area (1/Z of each side length)
#   * Downscale ratio: 5/Z

# ## Zoom Level Examples
# - 1.0x: Full image (downscaled 5x)
# - 2.0x: Shows 1/4 area (downscaled 2.5x)
# - 3.0x: Shows 1/9 area (downscaled 1.67x)
# - 4.0x: Shows 1/16 area (downscaled 1.25x)
# - 5.0x: Shows 1/25 area (original resolution)

# ## Viewing Path Guidelines
# Your viewing path should follow these principles:
# 1. Begin with low-power scanning to assess tissue architecture and identify key patterns
# 2. Focus exclusively on areas relevant to the given question
# 3. Avoid empty or background areas lacking diagnostic value
# 4. Prioritize regions showing key diagnostic criteria for the specific question
# 5. Use appropriate magnification for each observation (only changing zoom when necessary)
# 6. Follow a logical diagnostic hierarchy (primary lesions before secondary changes)
# 7. Ensure minimal redundant overlap between examined areas
# 8. Demonstrate systematic movement and reasoning between viewing steps

# Please provide the viewing path as a JSON list:

# ```json
# [
#   {{
#     "step": [step number],
#     "magnification": [zoom level from 2.0x-5.0x],
#     "region_coordinate": [x1, y1, x2, y2], # # Coordinates represent [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
#     ## Avoid empty area
#     "reasoning": [reasoning for moving to this position],
#     "need_to_see": [description of what should be observed at this position],
#   }}, 
#   ... 
# ] 
# ```

# Question: {Question}
# """


NAVIGATE_VQA_USER = """\
Please analyze this pathology image by creating a step-by-step viewing path that simulates how a professional pathologist would examine the specimen to answer the given question.
Focus specifically on areas relevant to the question and potential pathological changes.

## Image Viewing Mechanics
- The default view (1.0x) shows the complete image at 1/5 of its original resolution
- You can zoom to any level up to 5.0x (including decimal values like 2.3x)
- Image coordinates use relative values from 0 to 1
- At any zoom level Z:
  * Field of view: 1/Z² of total image area (1/Z of each side length)
  * Downscale ratio: 5/Z

## Zoom Level Examples
- 1.0x: Full image (downscaled 5x)
- 2.0x: Shows 1/4 area (downscaled 2.5x)
- 3.0x: Shows 1/9 area (downscaled 1.67x)
- 4.0x: Shows 1/16 area (downscaled 1.25x)
- 5.0x: Shows 1/25 area (original resolution)

## Viewing Path Guidelines
Your viewing path should follow these principles:
1. Begin with low-power scanning to assess tissue architecture and identify key patterns
2. Focus exclusively on areas relevant to the given question
3. Avoid empty or background areas lacking diagnostic value
4. Prioritize regions showing key diagnostic criteria for the specific question
5. Use appropriate magnification for each observation (only changing zoom when necessary)
6. Follow a logical diagnostic hierarchy (primary lesions before secondary changes)
7. Ensure minimal redundant overlap between examined areas
8. Demonstrate systematic movement and reasoning between viewing steps

Please provide the viewing path as a JSON list:

```json
[
  {{
    "step": [step number],
    "magnification": [zoom level from 2.0x-5.0x],
    "region_coordinate": [x1, y1, x2, y2], # # Coordinates represent [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    ## Avoid empty area
    "reasoning": [reasoning for moving to this position],
    "need_to_see": [description of what should be observed at this position],
  }}, 
  ... 
] 
```

Question: {Question}
"""

# ── VQA variant of Stage 3 ────────────────────────────────────────────────────

REASONING_VQA_USER = """\
You are a pathology specialist , you are examining images from a pathologist's viewing path through a pathological region.
Based on these images, you need to provide detailed reasoning to answer a specific question or select the most appropriate option from multiple choices.

For each image in the viewing path, you will receive:
- Magnification: Zoom level from 1.0x to 5.0x (where 1.0x shows the entire image)
- Region coordinates: Position you're viewing within the image (normalized from 0-1)
- Need to see: Specific features or structures to observe at this position

When analyzing the images, follow this structured approach:
1. Observational Evidence
- Document key findings visible in each image relevant to the question
- Be specific about which image/region shows each finding
- Describe cellular morphology, tissue architecture, and special features
2. Evidence Integration- Connect and compare findings across different images in the viewing path
3. Option Analysis (for multiple-choice questions)
- Evaluate how the observed evidence supports or contradicts each option
- Explain why incorrect options don't align with the visual evidence
- Justify why the best option is supported by your findings
4. Question-Specific Reasoning
- Apply pathological principles directly to the question
- Explain how specific visual clues lead to your conclusion
- Address any ambiguities or limitations in the visible evidence
5. Multi-scale Observation  (1.0x-5.0x, Low magnification-High magnification):
- Consider how features appear across different magnifications (1.0x-5.0x)
- Note how your viewing window represents portions of the original image at different zoom levels
- Your viewing window is 1/5 of original resolution where:
  * 1.0x shows full image
  * 2.0x shows 1/2 image length
  * 3.0x shows 1/3 image length
6. Reasoning Process Requirements: Use "thought flow" expression:
- Think while observing- Record thinking process in real-time (give the box coordinate)
- Allow returning for re-observation
- Record thinking turning points

Present your analysis as a natural, flowing narrative from a first-person perspective.
Think and reason as you observe, like an experienced pathologist examining the images in real-time.
Record your thought process, including any moments when you need to revisit certain images or when your thinking changes based on new observations.
The description should flow as a cohesive narrative **several paragraphs** rather than a bulleted list of observations.
Conclude with a brief one-sentence summary of your answer, followed by "Therefore, the answer is (X).

Question: {question}
"""

# ── Step 1 (revised) : Patch-level description → cluster → ROI ───────────────

PATCH_DESC_USER = """\
You are an expert pathologist examining a single patch from a whole-slide image.
Describe the key pathological features in 1-2 sentences (10 words max).

Cover: tissue architecture, cellular density, nuclear features, and whether high-magnification follow-up is needed.
Be objective and concise. No speculation beyond what is visible."""

CLUSTER_USER = """\
You are an expert pathologist. Group the following {tissue_type} WSI patches by pathological similarity.

Rules:
- Every patch ID must appear in exactly one group.
- is_background: set to true ONLY if the group contains no tissue (empty slide, artifact, or pure background); false otherwise.
- Set require_high_magnification=true for groups needing closer examination.
- diagnostic_priority: 1=examine first, 5=examine last.
- observation_points: list the specific morphological and structural features that a pathologist must examine when reviewing this group at high magnification.

Patch descriptions:
{patch_descriptions}

Reply with ONLY the JSON below — no other text before or after:
{{"groups":[{{"name":"<name>","id_list":[<ids>],"is_background":<true|false>,"require_high_magnification":<true|false>,"diagnostic_priority":<1-5>,"observation_points":["<key feature 1>","<key feature 2>"]}}]}}"""

# ── Stage 2 (revised) : 16×16 grid sub-patch description → LLM plan ─────────

NAV_SUBPATCH_DESC_USER = """\
You are an expert pathologist examining a small sub-region patch from a high-resolution pathology image.
Respond in 1 sentence (40 words max).

State: empty/background or tissue present; if tissue, note architecture, cell density, and any feature needing closer examination."""

NAV_GRID_PLAN_USER = """\
You are an expert pathologist planning a detailed examination of a high-resolution pathology region.
The region has been divided into a {grid_size}×{grid_size} grid ({total_patches} sub-patches, IDs 0–{max_id}).
Sub-patch ID = row × {grid_size} + column  (row 0 = top row, column 0 = left column).
Each sub-patch covers 1/{grid_size} of the region width and height.

The global screening stage identified the following key features to look for in this region:
{observation_points}

Sub-patch descriptions:
{subpatch_descriptions}

Your task:
- Select sub-patches where the above observation points can be assessed (skip empty / background / featureless areas).
- Assign a magnification level 2.0×–5.0× for each selected sub-patch based on diagnostic importance.
- Limit total selected sub-patches to at most {max_steps}.
- Prioritise sub-patches showing transition zones, high cellularity, or architectural disruption relevant to the observation points above.

Output ONLY valid JSON (do NOT include coordinates — they are computed automatically from the sub-patch ID):
```json
{{
  "plan": [
    {{
      "subpatch_id": <integer 0–{max_id}>,
      "magnification": <float 2.0–5.0>,
      "reasoning": "<why this sub-patch is relevant to the observation points>",
      "need_to_see": "<specific features to observe, linked to the observation points>"
    }}
  ]
}}
```
"""

NAV_GRID_PLAN_VQA_USER = """\
You are an expert pathologist planning a targeted examination of a high-resolution pathology region
to answer the following question:

Question: {question}

The global screening stage identified the following key features to look for in this region:
{observation_points}

The region has been divided into a {grid_size}×{grid_size} grid ({total_patches} sub-patches, IDs 0–{max_id}).
Sub-patch ID = row × {grid_size} + column  (row 0 = top row, column 0 = left column).
Each sub-patch covers 1/{grid_size} of the region width and height.

Sub-patch descriptions:
{subpatch_descriptions}

Your task:
- Select sub-patches relevant to answering the question and the observation points above (skip empty / background / unrelated areas).
- Assign magnification 2.0×–5.0× based on how important each sub-patch is for the question.
- Limit total selected sub-patches to at most {max_steps}.

Output ONLY valid JSON (do NOT include coordinates — they are computed automatically from the sub-patch ID):
```json
{{
  "plan": [
    {{
      "subpatch_id": <integer 0–{max_id}>,
      "magnification": <float 2.0–5.0>,
      "reasoning": "<why this sub-patch is relevant to the question and observation points>",
      "need_to_see": "<specific features to observe>"
    }}
  ]
}}
```
"""

# ── Stage 3 (revised) : Per-patch VLM description ────────────────────────────
# VLM is called once per cropped view; results are collected for LLM synthesis.

STEP3_PATCH_DESC_USER = """\
You are an expert pathologist performing a systematic multi-scale examination of a pathology specimen.

Viewing context: {step_info}

The global screening stage identified the following key features to examine in this region:
{observation_points}

Examine this field of view and describe your findings in 2-3 sentences (80 words max).
For each observation point listed above, state whether the feature is present, absent, or not assessable at this magnification.
Additionally report: tissue architecture, cellular density, nuclear morphology (size, pleomorphism, chromatin), and any architecturally or cytologically abnormal pattern.
Be objective and specific. Do not speculate beyond what is visible."""

# ── Stage 3 (revised) : LLM synthesis of all per-patch descriptions ──────────
# Text LLM receives all collected descriptions and produces a unified report.

STEP3_SYNTHESIS_USER = """\
You are an expert pathologist synthesizing a multi-scale examination of a {tissue_type} pathology specimen.

The following observations were collected sequentially from views at different magnifications:

{combined_observations}

Based on these multi-scale observations:
1. Systematically integrate findings across all views.
2. Identify the dominant pathological pattern.
3. Reason through the differential diagnosis, noting supporting and contradicting evidence.
4. Arrive at a definitive diagnostic conclusion.

Write as a flowing first-person narrative (not bullet points).

End your response with:

**Pathological Report:**
[One concise paragraph summarising your diagnosis and supporting evidence]

**Diagnosis:** [single most likely diagnosis term, e.g. IDC, DCIS, adenocarcinoma, normal]"""

STEP3_SYNTHESIS_VQA_USER = """\
You are an expert pathologist synthesizing a multi-scale examination of a pathology specimen to answer a specific question.

Question: {question}

The following observations were collected sequentially from views at different magnifications:

{combined_observations}

Based on these multi-scale observations:
1. Integrate findings across all views.
2. Evaluate how the evidence supports or contradicts each option (for multiple-choice questions).
3. Apply pathological principles to reach a conclusion.

Write as a flowing first-person narrative.

End with a brief one-sentence summary followed by:
"Therefore, the answer is (X)."

**Pathological Report:**
[Brief paragraph summarising key findings that support your answer]"""

# ── Stage 3 : Short diagnosis extraction (Text LLM) ──────────────────────────

STEP3_DIAGNOSIS_EXTRACT_PROMPT = """\
Extract the final diagnosis from the pathology report below.
Return ONLY the diagnosis term (e.g. "IDC", "DCIS", "adenocarcinoma", "normal tissue").
Do not include any explanation, punctuation beyond the term itself, or extra words.

Report:
{report_text}"""

# ── Text-only JSON extraction helper (Qwen3 4B) ───────────────────────────────

JSON_EXTRACT_PROMPT = """\
Extract the JSON from the following text and return ONLY the raw JSON with no extra text.
If the JSON is malformed, repair it so it is valid.

Text:
{raw_text}"""
