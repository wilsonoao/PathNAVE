#!/usr/bin/env python3
"""
Think-style Lymph Node Classification with Pre-extracted ROI Images
"""
import transformers
print(transformers.__version__)
print(transformers.__file__)

import json
import os
import base64
import time
import argparse
from dotenv import load_dotenv
load_dotenv(override=False)  # .env values only fill in missing env vars
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import openai
from langchain_community.chat_models import ChatOllama
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None  # WSI thumbnails can exceed PIL's default decompression bomb limit
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import pandas as pd
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class InferenceTracker:
    """Counts VLM/LLM calls and accumulated wall-clock time per QA."""

    def __init__(self):
        self.llm_calls = 0
        self.vlm_calls = 0
        self.llm_time = 0.0
        self.vlm_time = 0.0

    def record(self, is_vlm: bool, elapsed: float):
        if is_vlm:
            self.vlm_calls += 1
            self.vlm_time += elapsed
        else:
            self.llm_calls += 1
            self.llm_time += elapsed

    def to_dict(self, total_elapsed: float = None) -> dict:
        d = {
            "llm_calls": self.llm_calls,
            "vlm_calls": self.vlm_calls,
            "llm_time_s": round(self.llm_time, 3),
            "vlm_time_s": round(self.vlm_time, 3),
            "total_inference_time_s": round(self.llm_time + self.vlm_time, 3),
        }
        if total_elapsed is not None:
            d["total_elapsed_s"] = round(total_elapsed, 3)
        return d


def _has_image(messages: list) -> bool:
    """Return True if any message contains an image_url content item."""
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True
    return False


def move_to_cpu(model):
    import torch

    # 🔥 如果是 dict（你的 hf client）
    if isinstance(model, dict):
        model = model.get("model", model)

    if hasattr(model, "to"):
        model.to("cpu")

    torch.cuda.empty_cache()


def move_to_gpu(model):
    # 🔥 如果是 dict
    if isinstance(model, dict):
        model = model.get("model", model)

    if hasattr(model, "to"):
        model.to("cuda")


# --- Configuration ---
def get_api_server(model_name: str) -> str:
    """Determine API server based on model name."""
    model_name = model_name.lower()

    # local ollama
    if model_name.startswith("ollama:"):
        return "ollama"

    # explicit openai: prefix (e.g. openai:o3-mini, openai:o4-mini)
    if model_name.startswith("openai:"):
        return "oai"

    # OpenAI: gpt-* and o-series (o1, o1-mini, o3, o3-mini, o4-mini, etc.)
    if model_name.startswith('gpt'):
        return 'oai'
    if model_name.startswith('o1') or model_name.startswith('o3') or model_name.startswith('o4'):
        return 'oai'

    # xAI
    if model_name.startswith('grok'):
        return 'xai'

    # Gemini
    if model_name.startswith('gemini'):
        return 'gemini'

    # Patho-R1 / other OpenRouter-hosted models
    # 你可以直接傳：patho_r1、patho-r1、wenchuanzhang/patho-r1-7b
    # if (
    #     "patho_r1" in model_name
    #     or "patho-r1" in model_name
    #     or "wenchuanzhang/patho-r1-7b" in model_name
    # ):
    #     return "openrouter"

    # # 你原本特判的 qwen
    # if model_name == 'qwen/qwen2.5-vl-72b-instruct:free':
    #     return 'openrouter'

    # default
    return 'hf'

def initialize_client(api_server: str, model_name: str):
    """Initialize the client based on the API server."""

    if api_server == 'ollama':
        ollama_model = model_name.split("ollama:", 1)[1] if model_name.lower().startswith("ollama:") else model_name
        return ChatOllama(
            model=ollama_model,
            temperature=0.2,
            base_url="http://192.168.63.184:11434",
        )

    elif api_server == 'oai':
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        if not OPENAI_API_KEY:
            raise ValueError('OPENAI_API_KEY environment variable not set')
        # strip explicit prefix before storing client (actual model name resolved at call time)
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://api.openai.com/v1"
        )

    elif api_server == 'sbb':
        OPENAI_API_KEY = os.environ.get('SBB_API_KEY')
        if not OPENAI_API_KEY:
            raise ValueError('SBB_API_KEY environment variable not set')
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://api.shubiaobiao.com/v1"
        )

    elif api_server == 'xai':
        OPENAI_API_KEY = os.environ.get('XAI_API_KEY')
        if not OPENAI_API_KEY:
            raise ValueError('XAI_API_KEY environment variable not set')
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://api.x.ai/v1"
        )

    elif api_server == 'gemini':
        OPENAI_API_KEY = os.environ.get('GEMINI_API_KEY')
        if not OPENAI_API_KEY:
            raise ValueError('GEMINI_API_KEY environment variable not set')
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    elif api_server == 'openrouter':
        OPENAI_API_KEY = os.environ.get('OPENROUTER_API_KEY')
        if not OPENAI_API_KEY:
            raise ValueError('OPENROUTER_API_KEY environment variable not set')
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
    elif api_server == 'hf':
        model_name_l = model_name.lower()

        # ===== VLM branch =====
        if any(k in model_name_l for k in ["qwen2.5-vl", "qwen-vl", "llava", "patho-r1"]):
            import torch
            from transformers import AutoProcessor

            # Qwen2.5-VL

            from transformers import Qwen2_5_VLForConditionalGeneration

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )


            print(f"[HF] Loaded as VLM: {model_name}")
            return {
                "type": "hf_vlm",
                "model": model,
                "processor": processor,
            }

        # ===== LLM branch =====
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False,   # 避免某些 tokenizer.json 格式炸掉
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )

            print(f"[HF] Loaded as LLM: {model_name}")
            return {
                "type": "hf",
                "model": model,
                "tokenizer": tokenizer,
            }

    else:
        raise ValueError(f'Invalid API server configuration: {api_server}')


def call_model(client, model_name, messages, tracker: InferenceTracker = None):
    is_vlm = _has_image(messages)
    is_hf = isinstance(client, dict) and client.get("type") in ("hf", "hf_vlm")
    if is_hf:
        move_to_gpu(client)

    t0 = time.time()
    try:
        result = _call_model_impl(client, model_name, messages)
    finally:
        elapsed = time.time() - t0
        if tracker is not None:
            tracker.record(is_vlm, elapsed)
        if is_hf:
            move_to_cpu(client)
            torch.cuda.empty_cache()

    return result


def _call_model_impl(client, model_name, messages):

    # --- Ollama ---
    if isinstance(client, ChatOllama):
        response = client.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    # --- HuggingFace VLM ---
    if isinstance(client, dict) and client.get("type") == "hf_vlm":
        processor = client["processor"]
        model = client["model"]

        from PIL import Image

        image = None
        text_prompt = ""

        for m in messages:
            if isinstance(m["content"], list):
                for item in m["content"]:
                    if item["type"] == "image_url":
                        import base64
                        from io import BytesIO

                        img_data = base64.b64decode(item["image_url"]["url"].split(",")[1])
                        image = Image.open(BytesIO(img_data)).convert("RGB")

                    elif item["type"] == "text":
                        text_prompt += item["text"]

            elif isinstance(m["content"], str):
                text_prompt += m["content"]

        # ⚠️ 沒有 image → fallback 當 LLM
        if image is None:
            inputs = processor(text=[text_prompt], return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=256)
            return processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # 正常 VLM
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=256)

        return processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # --- HuggingFace LLM ---
    if isinstance(client, dict) and client.get("type") == "hf":
        tokenizer = client["tokenizer"]
        model = client["model"]

        prompt = ""
        for m in messages:
            if isinstance(m["content"], str):
                prompt += m["content"] + "\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=256)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- OpenAI / OpenRouter ---
    if hasattr(client, "chat"):
        # strip explicit provider prefix (e.g. "openai:o3-mini" → "o3-mini")
        if ":" in model_name and model_name.split(":")[0].lower() in ("openai", "ollama"):
            model_name = model_name.split(":", 1)[1]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content

    raise ValueError("Unsupported client type")

    
# Prompts
DIAGNOSTIC_CRITERIA = '''
Diagnostic Criteria for Lymph Node Metastasis:

POSITIVE indicators:
- Clear clusters of atypical epithelial cells with enlarged, irregular nuclei
- Obvious architectural disruption by tumor cells
- Glandular structures typical of adenocarcinoma
- Sheets or nests of malignant cells replacing lymphoid tissue

NEGATIVE indicators:
- Predominantly normal lymphoid tissue with small, uniform lymphocytes
- Preserved lymph node architecture with intact sinuses
- No clear atypical cell clusters
- Processing artifacts without viable tumor cells

TUMOR DEPOSIT (not positive lymph node):
- Entire node replaced by necrotic/dead tumor tissue
- No remaining viable lymph node architecture
'''

THUMB_PROMPT = '''
This is an H&E whole-slide image (WSI) thumbnail.

You are given a pathology question about this slide.

Your task:
1. Understand the question
2. Provide an initial impression of the slide
3. Identify what kind of evidence is needed to answer the question
4. Suggest regions or features that should be examined

Question:
{question}

Format your response as:
<impression>...</impression>
<analysis_plan>...</analysis_plan>
'''

ZOOM_IN_PROMPT = '''
You are examining a region of interest from a whole-slide image.

Your goal is to extract evidence relevant to the question.

Question:
{question}

Please:
1. Describe what you observe in this region
2. Determine whether this region provides evidence relevant to the question
3. Explain how this region supports or does not support a possible answer

Format your response as:
<observation>...</observation>
<relevance>relevant / not relevant</relevance>
<reasoning>...</reasoning>
'''

SUMMARY_PROMPT = """
You are answering a pathology question based on the analysis of a whole-slide image.

Question:
{question}

You are given:
- Initial impression of the slide
- Observations from multiple regions

Your task:
1. Synthesize all the evidence
2. Answer the question
3. Only use evidence that is clearly supported by the observations
4. If evidence is insufficient, say so

Format your response EXACTLY as follows:

<final_answer>...</final_answer>
<concise_answer>...</concise_answer>
<confidence>high / medium / low</confidence>
"""


def image_to_base64(image_path: str):
    """Convert image file to base64 string."""
    with open(image_path, 'rb') as image_file:
        img_data = image_file.read()
    img_str = base64.b64encode(img_data).decode()
    return f"data:image/jpeg;base64,{img_str}"

def load_case_data(case_dir: str):
    """Load thumbnail and ROI images for a case."""
    thumbnail_path = os.path.join(case_dir, "thumbnail.jpg")
    if not os.path.exists(thumbnail_path):
        raise FileNotFoundError(f"Thumbnail not found: {thumbnail_path}")
    
    # Find all ROI images
    roi_images = []
    roi_idx = 1
    while True:
        roi_path = os.path.join(case_dir, f"roi_{roi_idx}.jpg")
        if os.path.exists(roi_path):
            roi_images.append(roi_path)
            roi_idx += 1
        else:
            break
    
    if not roi_images:
        raise FileNotFoundError(f"No ROI images found in {case_dir}")
    
    return thumbnail_path, roi_images

def save_image_with_boxes(thumbnail_path: str, num_rois: int, output_path: str):
    """Create thumbnail with numbered boxes indicating ROI locations."""
    # Since we don't have exact bbox coordinates, we'll create a simple numbered overlay
    thumbnail = Image.open(thumbnail_path)
    img_with_boxes = thumbnail.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    try:
        font = ImageFont.truetype("Arial", 36)
    except:
        font = ImageFont.load_default()
    
    # Add text indicating number of ROIs analyzed
    text = f"ROIs analyzed: {num_rois}"
    draw.text((10, 10), text, fill="red", font=font)
    
    img_with_boxes.save(output_path, "JPEG", quality=95)
    return output_path

def analyze_roi_regions(roi_images: list, output_dir: str, initial_messages: list = None, model_name: str = "gpt-4o",
                       image_history_mode: str = "current_only", client: openai.OpenAI = None, question = None,
                       tracker: InferenceTracker = None):
    """
    Analyze each ROI region.
    
    image_history_mode options:
    - "current_only": Only send current ROI image (default)
    - "full_history": Send all previous images in conversation history
    """
    responses = []
    messages_history = initial_messages.copy() if initial_messages else []
    
    for i, roi_path in enumerate(roi_images):
        # Copy ROI image to output directory
        roi_filename = os.path.join(output_dir, f"roi_{i+1}.jpg")
        roi_image = Image.open(roi_path)
        roi_image.save(roi_filename, "JPEG", quality=95)
        
        roi_base64 = image_to_base64(roi_path)
        
        prompt = f"""
        This is the {i+1}th region of interest. {ZOOM_IN_PROMPT.format(question=question)}
        """
        
        user_message = {
            "role": "user", 
            "content": [
                {"type": "image_url", "image_url": {"url": roi_base64}},
                {"type": "text", "text": prompt},
            ]
        }
        
        if image_history_mode == "full_history":
            # Include full conversation history
            current_messages = messages_history + [user_message]
        else:
            # Only current ROI image and prompt
            current_messages = [user_message]
        
        region_response = call_model(client, model_name, current_messages, tracker=tracker)
        responses.append({
            'roi_index': i + 1,
            'response': region_response
        })
        
        # Add to history for next round if using full history mode
        if image_history_mode == "full_history":
            messages_history.append(user_message)
            messages_history.append({"role": "assistant", "content": region_response})
        
        # print(f"ROI {i+1}: {region_response}\n")
    
    return responses

def get_final_summary(initial_response: str, region_responses: list, model_name: str = "gpt-4o", client: openai.OpenAI = None, question = None,
                      tracker: InferenceTracker = None):
    """Generate final diagnostic summary."""
    summary_text = f"""
Question:
{question}

INITIAL IMPRESSION:
{initial_response}

REGION ANALYSIS:
"""
    for region_data in region_responses:
        roi_idx = region_data['roi_index']
        response = region_data['response']
        summary_text += f"\nROI {roi_idx}:\n{response}\n"
    
    summary_text += SUMMARY_PROMPT
    
    return call_model(
        client,
        model_name,
        [{"role": "user", "content": summary_text}],
        tracker=tracker,
    )

def create_pdf_report(chat_history: list, output_dir: str, output_filename: str = "pathology_report.pdf"):
    """Create PDF report from chat history."""
    output_path = os.path.join(output_dir, output_filename)
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='UserStyle', parent=styles['Normal'], textColor=colors.darkred, spaceAfter=12))
    styles.add(ParagraphStyle(name='AssistantStyle', parent=styles['Normal'], textColor=colors.black, spaceAfter=12))
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, spaceAfter=30)
    
    story = [Paragraph("Pathology Analysis Report", title_style), Spacer(1, 20)]
    
    for message in chat_history:
        role = message['role']
        content = message['content']
        if isinstance(content, str):
            style = styles['UserStyle'] if role == 'user' else styles['AssistantStyle']
            story.append(Paragraph(content.replace("\n", "<br/>"), style))
        elif isinstance(content, list):
            for item in content:
                if item['type'] == 'text':
                    style = styles['UserStyle'] if role == 'user' else styles['AssistantStyle']
                    story.append(Paragraph(item['text'].replace("\n", "<br/>"), style))
                elif item['type'] == 'image':
                    img_path = item['image']
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        img_width, img_height = img.size
                        max_width, max_height = 6 * inch, 8 * inch
                        scale_factor = min(max_width / img_width, max_height / img_height)
                        new_width, new_height = img_width * scale_factor, img_height * scale_factor
                        img_rl = RLImage(img_path, width=new_width, height=new_height)
                        story.append(img_rl)
                        story.append(Spacer(1, 12))
        story.append(Spacer(1, 12))
    
    doc.build(story)
    print(f"PDF report saved as {output_path}")

def analyze_case(case_dir: str, output_folder: str, model_name: str = "gpt-4o",
                image_history_mode: str = "current_only", client: openai.OpenAI = None, client_summary = None, question = None,
                model_name_summary: str = None, tracker: InferenceTracker = None) -> dict:
    """Analyze a single case with pre-extracted ROI images."""

    # 1. Setup output directory
    case_id = os.path.basename(case_dir)
    output_dir = os.path.join(output_folder, f"{case_id}_{question}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load case data
    try:
        thumbnail_path, roi_images = load_case_data(case_dir)
    except FileNotFoundError as e:
        print(f"Error loading case data: {e}")
        return None
    
    print(f"Analyzing case: {case_id}")
    print(f"Found {len(roi_images)} ROI images")
    print(f"Image history mode: {image_history_mode}")
    
    # 3. Copy thumbnail and create version with ROI indicators
    thumbnail_output = os.path.join(output_dir, "thumbnail.jpg")
    thumbnail_image = Image.open(thumbnail_path)
    thumbnail_image.save(thumbnail_output, "JPEG", quality=95)
    
    thumbnail_with_indicators = os.path.join(output_dir, "thumbnail_with_indicators.jpg")
    save_image_with_boxes(thumbnail_path, len(roi_images), thumbnail_with_indicators)
    
    # 4. Initial analysis of thumbnail
    print("Sending initial thumbnail to VLM...")
    thumbnail_base64 = image_to_base64(thumbnail_with_indicators)
    
    initial_user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": THUMB_PROMPT.format(question=question)},
            {"type": "image_url", "image_url": {"url": thumbnail_base64}}
        ]
    }
    
    initial_response = call_model(client, model_name, [initial_user_message], tracker=tracker)
    # print(f"Initial VLM Impression: {initial_response}\n")
    
    # 5. Analyze each ROI
    print("Analyzing each ROI region...")
    initial_messages = [initial_user_message, {"role": "assistant", "content": initial_response}]
    region_responses = analyze_roi_regions(roi_images, output_dir, initial_messages, model_name, image_history_mode, client, question, tracker=tracker)

    # 6. Get final summary
    print("Getting final summary...")
    summary_model = model_name_summary if model_name_summary else model_name
    final_summary = get_final_summary(initial_response, region_responses, summary_model, client_summary, question, tracker=tracker)
    print(f"Final Summary: done\n")
    
    # 7. Structure chat history for output
    clean_messages = [
        {"role": "user", "content": [
            {"type": "text", "text": THUMB_PROMPT.format(question=question)}, 
            {"type": "image", "image": thumbnail_with_indicators}
        ]},
        {"role": "assistant", "content": initial_response}
    ]
    
    for i, region_data in enumerate(region_responses):
        roi_path = os.path.join(output_dir, f"roi_{i+1}.jpg")
        clean_messages.append({
            "role": "user", 
            "content": [{"type": "image", "image": roi_path}]
        })
        clean_messages.append({
            "role": "assistant", 
            "content": region_data['response']
        })
    
    clean_messages.append({"role": "assistant", "content": final_summary})
    
    # 8. Extract diagnostic info
    import re
    diagnostic_info_match = re.search(r'<final_answer>(.*?)</final_answer>', final_summary, re.DOTALL)
    # if diagnostic_info_match:
    #     diagnostic_info = diagnostic_info_match.group(1)
    #     # pt_or_ln = re.search(r'PT_or_LN:\s*"([^"]+)"', diagnostic_info).group(1) if re.search(r'PT_or_LN:\s*"([^"]+)"', diagnostic_info) else "Unknown"
    #     # t_stage = int(re.search(r't_stage:\s*(\d+)', diagnostic_info).group(1)) if re.search(r't_stage:\s*(\d+)', diagnostic_info) else 0
    #     # lymph_node_positive = re.search(r'lymph_node_positive:\s*(true|false)', diagnostic_info).group(1).lower() == "true" if re.search(r'lymph_node_positive:\s*(true|false)', diagnostic_info) else False
    # else:
    #     pt_or_ln, t_stage, lymph_node_positive = "Unknown", 0, False
    
    # 9. Create final output structure
    output = {
        "case_id": case_id,
        "chat_history": clean_messages,
        "roi_count": len(roi_images),
        # "PT_or_LN": pt_or_ln,
        # "t_stage": t_stage,
        # "lymph_node_positive": lymph_node_positive,
        "model_name": model_name,
        "image_history_mode": image_history_mode,
        "timestamp": datetime.now().isoformat(),
        "anwser": diagnostic_info_match.group() if diagnostic_info_match else None,
        "final_summary": final_summary,
        "initial_response": initial_response,
    }
    print(type(output))
    
    # 10. Save JSON and PDF report
    json_path = os.path.join(output_dir, "analysis_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # create_pdf_report(clean_messages, output_dir)
    print(f"All outputs saved to directory: {output_dir}")
    
    return output


import os
import json

import os
import json

def load_qa_data(json_file: str, data_dir: str):
    """Load QA dataset from a single JSON file and match with case directories."""
    
    if not os.path.exists(json_file):
        print(f"Warning: JSON file {json_file} not found")
        return []
    
    # 🔥 Step 1: 建立 case_dir mapping
    case_dirs = {}
    for d in os.listdir(data_dir):
        full_path = os.path.join(data_dir, d)
        if os.path.isdir(full_path):
            case_dirs[d.lower()] = d  # 保留原名稱
    
    qa_list = []
    
    # 🔥 Step 2: 讀單一 JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 支援 dict / list
    if isinstance(data, dict):
        data = [data]
    
    # 🔥 Step 3: matching
    for item in data:
        raw_id = item.get("Id")
        if raw_id is None:
            continue
        
        raw_id_lower = raw_id.lower()
        matched_case = None
        
        # ✅ 完全 match
        if raw_id_lower in case_dirs:
            matched_case = case_dirs[raw_id_lower]
        else:
            # ✅ 模糊 match
            for case_name_lower, case_name in case_dirs.items():
                if raw_id_lower in case_name_lower or case_name_lower in raw_id_lower:
                    matched_case = case_name
                    break
        
        if matched_case is None:
            print(f"[Warning] No matching case dir for {raw_id}")
            continue
        

        qa_list.append({
            "case_id": matched_case,
            "case_root": os.path.join(data_dir, matched_case),
            "question": item.get("Question"),
            "choices": item.get("Choice", []),
            "answer": item.get("Answer"),
            "type": item.get("type"),
            "original_id": raw_id
        })
    
    return qa_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Think-style analysis of lymph node cases with pre-extracted ROI images.')
    parser.add_argument('-d', '--data-dir', default="LNCO2", help='Data directory containing case folders (default: LNCO2)')
    parser.add_argument('-o', '--output', required=True, help='Output folder path')
    parser.add_argument('-m', '--model', default="gpt-4o", help='Model name to use (default: gpt-4o)')
    parser.add_argument('-m_s', '--model_summary', default="gpt-4o", help='Model name to use (default: gpt-4o)')
    parser.add_argument('-n', '--num-cases', type=int, default=5, help='Number of cases to test (default: 5)')
    parser.add_argument('--qa_file', default="annotation_summary_merged_remove_unknown_exclude.csv", 
                       help='Path to annotation CSV file')
    parser.add_argument('--image-history', choices=['current_only', 'full_history'], default='current_only',
                       help='Image history mode: current_only (default) or full_history')
    parser.add_argument('--start', type=int, default=0,
                       help='Start index (inclusive) of qa_datas slice (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='End index (exclusive) of qa_datas slice (default: all)')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Initialize API client
    api_server = get_api_server(args.model)
    client = initialize_client(api_server, args.model)
    move_to_cpu(client)

    api_server = get_api_server(args.model_summary)
    client_summary = initialize_client(api_server, args.model_summary)
    move_to_cpu(client_summary)

    # Load annotation data for ground truth
    qa_datas = load_qa_data(json_file=args.qa_file, data_dir=args.data_dir)
    qa_datas = qa_datas[args.start:args.end]
    print(f"[Slice] Processing {len(qa_datas)} cases (start={args.start}, end={args.end})")

    # Find available cases
    # case_dirs = []
    # if os.path.exists(args.data_dir):
    #     for item in os.listdir(args.data_dir):
    #         case_path = os.path.join(args.data_dir, item)
    #         if os.path.isdir(case_path):
    #             # Check if it has both thumbnail and at least one ROI
    #             thumbnail_path = os.path.join(case_path, "thumbnail.jpg")
    #             roi_1_path = os.path.join(case_path, "roi_1.jpg")
    #             if os.path.exists(thumbnail_path) and os.path.exists(roi_1_path):
    #                 case_dirs.append(case_path)

    # if not case_dirs:
    #     print(f"No valid cases found in {args.data_dir}")
    #     exit(1)

    # print(f"Found {len(case_dirs)} valid cases")
    
    # Limit to requested number of cases
    # case_dirs = case_dirs[:args.num_cases]
    # print(case_dirs)
    
    import re as _re

    def _attach_ground_truth(result: dict, qa_data: dict) -> dict:
        """Add ground truth, predicted answer, and correctness to a result dict."""
        gt = qa_data.get("answer")
        result["ground_truth"] = gt
        result["choices"] = qa_data.get("choices", [])

        # Extract the concise answer produced by the model
        final_summary = result.get("final_summary", "") or ""
        concise_match = _re.search(r"<concise_answer>(.*?)</concise_answer>", final_summary, _re.DOTALL)
        if concise_match:
            predicted_text = concise_match.group(1).strip()
        else:
            final_match = _re.search(r"<final_answer>(.*?)</final_answer>", final_summary, _re.DOTALL)
            predicted_text = final_match.group(1).strip() if final_match else ""

        result["predicted"] = predicted_text

        # Correctness: match full ground truth OR its abbreviation (text inside parentheses).
        # e.g. "Invasive Ductal Carcinoma (IDC)" also matches if predicted contains "IDC".
        if gt and predicted_text:
            gt_lower = gt.strip().lower()
            pred_lower = predicted_text.lower()
            abbrev_match = _re.search(r'\(([^)]+)\)', gt)
            abbrev = abbrev_match.group(1).strip().lower() if abbrev_match else None
            result["correct"] = gt_lower in pred_lower or bool(abbrev and abbrev in pred_lower)
        else:
            result["correct"] = False

        return result

    test_total_dir = os.path.join(args.output, "test_total")
    test_total_cases = set()
    if os.path.isdir(test_total_dir):
        for fname in os.listdir(test_total_dir):
            case_id_tt = fname.split("_")[0]
            if case_id_tt:
                test_total_cases.add(case_id_tt)

    results = []
    for i, qa_data in enumerate(qa_datas):
        try:
            case_id = os.path.basename(qa_data["case_root"])
            if case_id in test_total_cases:
                print(f"[Skip] Already in test_total: {case_id}")
                continue
            output_dir = os.path.join(args.output, f"{case_id}_{qa_data['question']}")
            existing_json = os.path.join(output_dir, "analysis_results.json")
            if os.path.exists(existing_json):
                print(f"[Skip] Already processed: {case_id} | {qa_data['question']}")
                with open(existing_json, "r") as f:
                    result = json.load(f)
                # Back-fill ground truth if the saved JSON is missing it
                if "ground_truth" not in result:
                    result = _attach_ground_truth(result, qa_data)
                    with open(existing_json, "w") as f:
                        json.dump(result, f, indent=2)
                results.append(result)
                continue

            result = None
            for attempt in range(1, 11):
                try:
                    tracker = InferenceTracker()
                    qa_start = time.time()
                    result = analyze_case(qa_data["case_root"], args.output, args.model, args.image_history, client, client_summary, qa_data["question"], model_name_summary=args.model_summary, tracker=tracker)
                    qa_elapsed = time.time() - qa_start
                    break
                except (RuntimeError, torch.cuda.OutOfMemoryError) as oom_e:
                    if "out of memory" in str(oom_e).lower() or isinstance(oom_e, torch.cuda.OutOfMemoryError):
                        print(f"[OOM] attempt {attempt}/10 for case {qa_data['question']}, waiting 15s... ({oom_e})")
                        torch.cuda.empty_cache()
                        time.sleep(15)
                    else:
                        raise
            if result:
                stats = tracker.to_dict(total_elapsed=qa_elapsed)
                print(
                    f"  [Stats] LLM calls={stats['llm_calls']}  VLM calls={stats['vlm_calls']}"
                    f"  inference={stats['total_inference_time_s']}s"
                    f"  elapsed={stats['total_elapsed_s']}s"
                )
                result["inference_stats"] = stats
                result = _attach_ground_truth(result, qa_data)
                json_path = os.path.join(output_dir, "analysis_results.json")
                with open(json_path, "w") as f:
                    json.dump(result, f, indent=2)
                results.append(result)
        except Exception as e:
            print(f"Error processing case {qa_data['question']}: {e}")
            continue
    
    # Save summary results
    summary_path = os.path.join(args.output, f"think_analysis_summary_{args.model}_{args.image_history}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate performance metrics if ground truth available
    correct_predictions = sum(1 for r in results if r.get('correct', False))
    total_with_gt = sum(1 for r in results if 'ground_truth' in r)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total cases processed: {len(results)}")
    print(f"Image history mode: {args.image_history}")
    if total_with_gt > 0:
        accuracy = correct_predictions / total_with_gt
        print(f"Cases with ground truth: {total_with_gt}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2%}")
    
    print(f"Summary saved to: {summary_path}")