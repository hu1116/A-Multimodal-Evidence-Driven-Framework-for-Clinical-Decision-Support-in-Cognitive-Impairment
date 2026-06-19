import json
import os
import re
import glob
import logging
import shutil
from typing import Any, Dict, List, Optional, TypedDict, Literal, Union
import time
import random
import pandas as pd
from functools import partial
from multiprocessing import Pool, cpu_count
from poe_client import PoeClient
from openai import OpenAI 
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ValidationError, validator
import httpx 

try:
    import config
except ImportError:
    config = None


SUMMARY_CACHE_FILENAME = "similar_case_summaries.csv"
RESULTS_ROOT = "./adRAG/mask0"
DATA_ROOT = "./adRAG/data"
CLASS_LABELS = ["NC", "MCI", "DE"]
DEFAULT_PREDICTION_COLUMNS = [
    "ID",
    "primary_label",
    "secondary_labels",
    "combined_label",
    "prediction_label",
]

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

class DiagnosisResult(BaseModel):
    primary_label: Literal['NC', 'MCI', 'DE'] = Field(..., description="The primary diagnosis label.")
    secondary_labels: List[str] = Field(default=[], description="List of subtypes (e.g., AD, LBD). Use ['NONE'] if NC.")
    confidence: int = Field(..., ge=0, le=100, description="Confidence score (0-100).")
    clinical_reasoning: str = Field(..., description="Summary of the reasoning process.")
    
    @validator('secondary_labels', pre=True)
    def handle_none_secondary(cls, v):
        if isinstance(v, str):
            if v.upper() == 'NONE': return ['NONE']
            return [x.strip() for x in v.split(',')]
        return v

def parse_json_from_llm(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError("No valid JSON found in response")

def clean_llm_response(content: str) -> str:
    if not content:
        return ""
    content = content.strip()
    return content.strip()

def set_global_model(model_name):
    global POE_MODEL_NAME
    if model_name:
        POE_MODEL_NAME = model_name

def get_global_model():
    return POE_MODEL_NAME

SECTION_PATTERN = re.compile(r"\*\*(.+?)\*\*:\s*", re.MULTILINE)

def parse_structured_sections(text):
    if not isinstance(text, str):
        return {}

    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return {}

    sections = {}
    for idx, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        sections[section_name] = section_text
    return sections

def extract_section_items(section_text):
    if not section_text:
        return []

    items = []
    for line in section_text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned[0] in {"-", "*", "•"}:
            cleaned = cleaned[1:].strip()
        items.append(cleaned)
    return items

def load_ml_prediction_data(csv_path: Optional[str]) -> Dict[str, List[str]]:
    if not csv_path or not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Warning: Failed to read ML prediction file {csv_path}: {exc}")
        return {}

    if 'ID' not in df.columns:
        return {}

    predictions: Dict[str, List[str]] = {}
    base_labels = ['NC', 'MCI', 'DE', 'CI', 'MCI_A', 'MCI_AM', 'MCI_Na', 'MCI_NaM', 'AD', 'LBD', 'VD', 'FTD', 'EXC', 'PSY', 'ODE']

    for _, row in df.iterrows():
        patient_id = row.get('ID')
        if pd.isna(patient_id): continue
        patient_key = str(patient_id).strip()
        if not patient_key: continue

        labels: List[str] = []
        for base_name in base_labels:
            col_name = f"{base_name}_label"
            if col_name in df.columns:
                val = row[col_name]
                if pd.isna(val): continue

                match = False
                try:
                    match = float(val) == 1.0
                except:
                    match = str(val).strip().lower() in ['1', '1.0', 'true']

                if match:
                    labels.append(base_name)
        
        predictions[patient_key] = labels
    return predictions

def build_ml_prediction_section(patient_id: Any, predictions: Dict[str, List[str]]) -> str:
    if not predictions: return ""
    patient_key = str(patient_id).strip()
    labels = predictions.get(patient_key, [])
    return ", ".join(labels) if labels else "None"

def load_summary_cache(cache_path: str) -> Dict[str, str]:
    if not cache_path or not os.path.exists(cache_path): return {}
    try:
        df = pd.read_csv(cache_path)
        if "ID" not in df.columns or "summary" not in df.columns: return {}
        cache: Dict[str, str] = {}
        for _, row in df.iterrows():
            patient_id = str(row.get("ID", "")).strip()
            summary = row.get("summary")
            if patient_id and isinstance(summary, str):
                cache[patient_id] = summary
        return cache
    except:
        return {}

def save_summary_cache(cache_path: str, cache: Dict[str, str]) -> None:
    if not cache_path: return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    rows = sorted(((pid, summary) for pid, summary in cache.items()), key=lambda item: item[0])
    df = pd.DataFrame(rows, columns=["ID", "summary"])
    df.to_csv(cache_path, index=False, encoding="utf-8")

def sanitize_patient_id(patient_id: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(patient_id))

def load_primary_label_map(csv_path: str) -> Dict[str, str]:
    if not csv_path or not os.path.exists(csv_path): return {}
    try:
        df = pd.read_csv(csv_path)
        if "ID" not in df.columns: return {}
        df["ID"] = df["ID"].astype(str).str.strip()
        label_map: Dict[str, str] = {}
        for _, row in df.iterrows():
            patient_id = row.get("ID")
            if not isinstance(patient_id, str) or not patient_id: continue
            positives: List[str] = []
            for label in CLASS_LABELS:
                col_name = f"{label}_label"
                if col_name not in df.columns: continue
                val = row.get(col_name)
                if pd.isna(val): continue
                
                is_pos = False
                try: 
                    is_pos = float(val) == 1.0
                except: 
                    is_pos = str(val).strip().upper() in ['1', '1.0', label, f"{label}_LABEL"]
                
                if is_pos: positives.append(label)
            label_map[patient_id] = positives[0] if positives else "UNKNOWN"
        return label_map
    except:
        return {}

def load_prediction_dataframe(model_name: str, fold_index: int) -> pd.DataFrame:
    csv_path = os.path.join(RESULTS_ROOT, f"model-fold{fold_index}", model_name, "prediction_labels.csv")
    if not os.path.exists(csv_path): return pd.DataFrame(columns=DEFAULT_PREDICTION_COLUMNS)
    try:
        df = pd.read_csv(csv_path)
        if "ID" in df.columns: df["ID"] = df["ID"].astype(str).str.strip()
        return df
    except:
        return pd.DataFrame(columns=DEFAULT_PREDICTION_COLUMNS)

def load_summary_cache_map(model_name: str, fold_index: int) -> Dict[str, str]:
    cache_path = os.path.join(RESULTS_ROOT, f"model-fold{fold_index}", model_name, SUMMARY_CACHE_FILENAME)
    if not os.path.exists(cache_path): return {}
    try:
        df = pd.read_csv(cache_path)
        if "ID" not in df.columns or "summary" not in df.columns: return {}
        df["ID"] = df["ID"].astype(str).str.strip()
        return dict(zip(df["ID"], df["summary"]))
    except:
        return {}

def copy_report_for_patient(source_dir: str, dest_dir: str, patient_id: str) -> bool:
    if not os.path.isdir(source_dir): return False
    safe_id = sanitize_patient_id(patient_id)
    pattern = os.path.join(source_dir, f"{safe_id}_*.txt")
    candidates = glob.glob(pattern)
    if not candidates: return False
    latest_source = max(candidates, key=os.path.getmtime)
    for existing in glob.glob(os.path.join(dest_dir, f"{safe_id}_*.txt")):
        try: os.remove(existing)
        except OSError: pass
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(latest_source, os.path.join(dest_dir, os.path.basename(latest_source)))
    return True

def prepare_base_outputs_for_fold(
    model_name: str,
    fold_index: int,
    rerun_ids: set[str],
    current_ml_labels: Dict[str, str],
    resume: bool = True,         
) -> Dict[str, Any]:
    current_model_dir = os.path.join(RESULTS_ROOT, f"model-fold{fold_index}", model_name)
    os.makedirs(current_model_dir, exist_ok=True)

    if not resume:
        for entry in os.listdir(current_model_dir):
            entry_path = os.path.join(current_model_dir, entry)
            try:
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
                else:
                    os.remove(entry_path)
            except OSError as exc:
                print(f"Warning: failed to remove {entry_path}: {exc}")

    predictions_to_copy: List[pd.DataFrame] = []
    summary_entries: List[Dict[str, str]] = []
    copied_ids: set[str] = set()

    ml_label_cache: Dict[int, Dict[str, str]] = {}
    prediction_cache: Dict[int, pd.DataFrame] = {}
    summary_cache: Dict[int, Dict[str, str]] = {}

    for patient_id, current_label in current_ml_labels.items():
        if patient_id in rerun_ids:
            continue

        matching_found = False
        for previous_fold in range(fold_index - 1, -1, -1):
            if previous_fold not in ml_label_cache:
                label_path = os.path.join(RESULTS_ROOT, f"temp_y_fold{previous_fold}.csv")
                ml_label_cache[previous_fold] = load_primary_label_map(label_path)
            previous_labels = ml_label_cache.get(previous_fold, {})
            if previous_labels.get(patient_id) != current_label:
                continue

            source_model_dir = os.path.join(RESULTS_ROOT, f"model-fold{previous_fold}", model_name)
            report_copied = copy_report_for_patient(source_model_dir, current_model_dir, patient_id)
            if not report_copied:
                continue

            if previous_fold not in prediction_cache:
                prediction_cache[previous_fold] = load_prediction_dataframe(model_name, previous_fold)
            previous_predictions = prediction_cache[previous_fold]
            if not previous_predictions.empty and "ID" in previous_predictions.columns:
                matched_rows = previous_predictions[previous_predictions["ID"] == patient_id]
                if not matched_rows.empty:
                    predictions_to_copy.append(matched_rows.copy())

            if previous_fold not in summary_cache:
                summary_cache[previous_fold] = load_summary_cache_map(model_name, previous_fold)
            summary_map = summary_cache[previous_fold]
            summary_text = summary_map.get(patient_id)
            if summary_text:
                summary_entries.append({"ID": patient_id, "summary": summary_text})

            copied_ids.add(patient_id)
            rerun_ids.discard(patient_id)
            matching_found = True
            break

        if not matching_found:
            print(
                f"Fold {fold_index}, model {model_name}: no matching historical report "
                f"for ID {patient_id} with label {current_label}. It will be regenerated."
            )
            rerun_ids.add(patient_id)

    if predictions_to_copy:
        base_predictions_df = pd.concat(predictions_to_copy, ignore_index=True)
        if "ID" in base_predictions_df.columns:
            base_predictions_df = base_predictions_df.drop_duplicates("ID", keep="last")
            base_predictions_df = base_predictions_df.sort_values("ID")
    else:
        base_predictions_df = pd.DataFrame(columns=DEFAULT_PREDICTION_COLUMNS)

    base_summary_df = (
        pd.DataFrame(summary_entries) if summary_entries
        else pd.DataFrame(columns=["ID", "summary"])
    )

    return {
        "prediction_df": base_predictions_df,
        "summary_df": base_summary_df,
        "copied_ids": copied_ids,
        "updated_rerun_ids": rerun_ids,
    }


class PatientState(TypedDict):
    patient_id: str
    patient_text: str
    ml_section: str
    retrieved_docs: list
    similar_cases_summary: str
    patient_contradictions: list
    raw_response: str
    parsed_result: Optional[DiagnosisResult]
    critique: Optional[str]
    attempt_count: int
    error: Optional[str]

def generate_node(state: PatientState):
    print(f"--- [Generate] Attempt {state['attempt_count'] + 1} for {state['patient_id']} ---")
    prompt = generate_diagnosis_report(
        state['patient_text'],
        state['retrieved_docs'],
        state['similar_cases_summary'],
        state['ml_section'],
        state['critique'],
    )
    response = call_poe(prompt) 
    try:
        clean_text = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL).strip()
        json_data = parse_json_from_llm(clean_text)
        result = DiagnosisResult(**json_data)
        return {"raw_response": response, "parsed_result": result, "error": None}
    except Exception as e:
        return {"raw_response": response, "error": str(e)}

def critic_node(state: PatientState):
    print(f"--- [Critic] Reviewing {state['patient_id']} ---")
    if state.get("error"):
        return {
            "critique": f"Format Error: The output was not valid JSON matching the schema. Error: {state['error']}",
            "attempt_count": state["attempt_count"] + 1
        }
    result = state["parsed_result"]
    if result.primary_label == 'NC' and (len(result.secondary_labels) > 0 and 'NONE' not in result.secondary_labels):
        return {
            "critique": "Logic Error: Primary label is NC, but secondary labels are provided. NC patients cannot have subtypes.",
            "attempt_count": state["attempt_count"] + 1
        }
    return {"critique": None}

def should_continue(state: PatientState):
    if state["critique"] is None: return "end"
    if state["attempt_count"] >= 3: return "end" 
    return "retry"

try:
    from rank_bm25 import BM25Okapi
    import numpy as np
    BM25_AVAILABLE = True
except ImportError:
    print("Warning: rank_bm25 or numpy not installed, retrieval function may not be available")
    BM25_AVAILABLE = False

class SimilarCaseRetriever:
    def __init__(self, case_db_path="./adRAG/case_reference/feature_descriptions_optimized.csv", 
                 ground_truth_path="data/all_samples_x_final.csv"):
        self.case_db = self._load_case_database(case_db_path)
        self.ground_truth = self._load_ground_truth(ground_truth_path)

    def _load_case_database(self, path):
        if not os.path.exists(path): return pd.DataFrame()
        df = pd.read_csv(path)
        return df

    def _load_ground_truth(self, path):
        if not os.path.exists(path): return pd.DataFrame()
        df = pd.read_csv(path)
        return df

    def extract_contradictions(self, text):
        contradictions = []
        block = self._extract_contradiction_block(text)
        if block:
            contradictions.extend(extract_section_items(block))
            if not contradictions:
                contradictions.extend([line.strip() for line in block.splitlines() if line.strip()])
        else:
            sections = parse_structured_sections(text)
            if sections:
                section_text = sections.get("Contradictions")
                if section_text:
                    contradictions.extend(extract_section_items(section_text))

        if contradictions:
            seen = set()
            normalized = []
            for item in contradictions:
                norm = re.sub(r"\s+", " ", item.strip().lower())
                if norm and norm not in seen:
                    seen.add(norm)
                    normalized.append(item.strip())
            return normalized[:10]

        fallback = []
        lowered = text.lower()
        if any(token in lowered for token in ["contradiction", "conflict", "discrep", "inconsist"]):
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['contradiction', 'contrast', 'conflict', 'discrepancy', 'inconsistent']):
                    cleaned = sentence.strip()
                    if cleaned: fallback.append(cleaned)
        return fallback[:5]

    def _extract_contradiction_block(self, text: str) -> str:
        if not isinstance(text, str): return ""
        pattern = re.compile(r"\*\*\s*Contradictions", re.IGNORECASE)
        match = pattern.search(text)
        if not match: return ""
        start = match.end()
        remainder = text[start:].lstrip()
        if not remainder: return ""
        double_newline = re.search(r"\n\s*\n", remainder)
        if double_newline: return remainder[:double_newline.start()].strip()
        return remainder.strip()

    def _extract_bold_keywords(self, contradictions):
        keywords: set[str] = set()
        for contradiction in contradictions:
            bold_texts = re.findall(r'\*\*(.*?)\*\*', contradiction)
            for text in bold_texts:
                clean_text = re.sub(r'\s+', ' ', text.strip().lower())
                if clean_text: keywords.update(clean_text.split())
        return keywords

    def calculate_simple_similarity(self, patient_contradictions, case_contradictions):
        patient_keywords = self._extract_bold_keywords(patient_contradictions)
        case_keywords = self._extract_bold_keywords(case_contradictions)
        if not patient_keywords and not case_keywords: return 0.0
        intersection = patient_keywords & case_keywords
        union = patient_keywords | case_keywords
        return len(intersection) / len(union) if union else 0.0

    def find_similar_cases(self, patient_description, top_k=3, min_similarity=0.01):
        if self.case_db.empty or self.ground_truth.empty: return []
        patient_contradictions = self.extract_contradictions(patient_description)
        if not patient_contradictions: return []
        patient_keywords = self._extract_bold_keywords(patient_contradictions)

        scored_cases = []
        for _, case_row in self.case_db.iterrows():
            case_id = case_row.get('ID', '')
            case_description = case_row.get('optimized_description', '')
            if not case_id or not case_description: continue
            case_contradictions = self.extract_contradictions(case_description)
            if not case_contradictions: continue
            similarity = self.calculate_simple_similarity(patient_contradictions, case_contradictions)
            if similarity >= min_similarity:
                gt_row = self.ground_truth[self.ground_truth['ID'] == case_id]
                gt_labels = self._get_positive_labels(gt_row.iloc[0]) if not gt_row.empty else []
                case_keywords = self._extract_bold_keywords(case_contradictions)
                shared_keywords = patient_keywords & case_keywords
                scored_cases.append({
                    'case_id': case_id,
                    'similarity': similarity,
                    'patient_keywords': sorted(patient_keywords),
                    'case_keywords': sorted(case_keywords),
                    'shared_keywords': sorted(shared_keywords),
                    'contradictions': case_contradictions,
                    'ground_truth': gt_labels,
                    'description_preview': case_description[:200] + "..." if len(case_description) > 200 else case_description
                })
        scored_cases.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_cases[:top_k]

    def _get_positive_labels(self, gt_row):
        positive_labels = []
        label_columns = ['NC_label', 'MCI_label', 'DE_label', 'CI_label', 'MCI_A_label', 'MCI_AM_label', 'MCI_Na_label', 'MCI_NaM_label', 'AD_label', 'LBD_label', 'VD_label', 'FTD_label', 'EXC_label', 'PSY_label', 'ODE_label']
        for col in label_columns:
            if col in gt_row and pd.notna(gt_row[col]) and float(gt_row[col]) == 1.0:
                positive_labels.append(col)
        return positive_labels

def generate_similar_cases_summary(similar_cases, patient_contradictions=None):
    if not similar_cases: return "No similar cases found with matching contradictions."
    case_ids = [case['case_id'] for case in similar_cases]
    patient_contradictions = patient_contradictions or []
    patient_contradictions_block = "\n".join(f"- {item}" for item in patient_contradictions) if patient_contradictions else "- None provided"

    formatted_cases = []
    for case in similar_cases:
        contradictions = "\n    - ".join(case.get('contradictions', []) or ["None provided"])
        ground_truth = ", ".join(case.get('ground_truth', []) or ["Unknown"])
        formatted_cases.append(
            f"Case ID: {case.get('case_id', 'UNKNOWN')}\n"
            f"Similarity: {case.get('similarity', 0):.3f}\n"
            f"Ground Truth Labels: {ground_truth}\n"
            f"Contradictions:\n    - {contradictions}"
        )

    cases_block = "\n\n".join(formatted_cases)
    prompt = f"""
As a medical expert, analyze the following similar cases that share contradictions with the current patient. 
Provide a concise summary (max 300 words) of the diagnostic patterns observed in these cases.

Similar Case References:
{cases_block}

Patient Contradictions:
{patient_contradictions_block}

Please provide:
1. Common diagnostic patterns among cases with similar contradictions
2. How these contradictions were reconciled relative to the reported ground truths
3. Key insights or cautions for the current case
4. Explicitly list the case IDs referenced in your summary

Keep the summary focused and clinically relevant.
"""
    try:
        response = call_poe(prompt)
        if isinstance(response, str):
            prefix = f"Case IDs: {', '.join(case_ids)}\n"
            if not response.strip().lower().startswith("case ids"):
                response = prefix + response
        return response
    except Exception as e:
        return f"Summary of {len(similar_cases)} similar cases with matching contradictions. Case IDs: {', '.join(case_ids)}. Ground truth patterns: {[case['ground_truth'] for case in similar_cases]}"

class MedicalRetriever:
    def __init__(self, corpus_path="AD_less/5w_txt_13,385"):
        if not BM25_AVAILABLE:
            raise ImportError("BM25Okapi not installed. Please install: pip install rank_bm25")
        self.corpus = self._load_corpus(corpus_path)
        self.bm25 = BM25Okapi([doc["text"] for doc in self.corpus])
    
    def _load_corpus(self, path):
        corpus = []
        for file in os.listdir(path):
            with open(os.path.join(path, file)) as f:
                corpus.append({"id": file, "text": f.read()})
        return corpus
    
    def retrieve(self, query, top_k=5):
        if not BM25_AVAILABLE: return []
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.corpus[i] for i in top_indices]

def extract_key_information(patient_text):
    sections = parse_structured_sections(patient_text)
    if not sections: return patient_text[:500]

    priority_sections = [
        "Key Anomalies & Flags", "Contradictions", "Cognitive & Behavioral Status",
        "Neurological & Motor Findings", "Functional Status & ADLs", "Medical History", "Medications",
    ]
    snippets = []
    for name in priority_sections:
        section_body = sections.get(name)
        if not section_body: continue
        items = extract_section_items(section_body)
        if not items: continue
        snippets.append(f"{name}: " + "; ".join(items))

    if snippets: return " | ".join(snippets)[:1000]
    return patient_text[:500]

def multi_hop_retrieval(patient_text, retriever):
    key_info = extract_key_information(patient_text)
    hop1_query = f"cognitive impairment {key_info}"
    results1 = retriever.retrieve(hop1_query)
    contradictions_query = "contradiction inconsistent discrepancy " + key_info
    results2 = retriever.retrieve(contradictions_query)
    all_docs = {doc["id"]: doc for doc in results1 + results2}
    return list(all_docs.values())[:5]

def generate_diagnosis_report(patient_text, retrieved_docs, similar_cases_summary, ml_section="", critique=None):
    docs_str = "\n\n".join([f"## DOC-{doc['id']}\n{doc['text'][:1500]}" for doc in retrieved_docs])
    ml_reference = ml_section or "NONE"
    critique_instruction = ""
    if critique:
        critique_instruction = f"""
        \n\n!!! PREVIOUS ATTEMPT REJECTED !!!
        Your previous output was rejected by the auditor.
        Feedback: {critique}
        You must fix these issues in this new attempt.
        """
    prompt = f"""
You are a senior neurologist conducting cognitive impairment diagnosis. Integrate multiple sources of information and provide a comprehensive diagnostic report in English.

### 1.Multimodal(mHC) prediction
{ml_reference}

### 2. STRICT Verification Protocol (Read Carefully)
**The AI prediction is highly reliable.** Do NOT change it unless you find a fatal error.

**Your Rule of Thumb:** "In dubio pro reo" (When in doubt, stick to the AI prediction).

**MODE A: Verification (If Anchor is NOT "NONE")**
1.  **Default Position**: You intend to **CONFIRM** the AI prediction.
2.  **Burden of Proof**: To override the AI, you must find **EXPLICIT, TEXTUAL EVIDENCE** that makes the prediction impossible.
    - *Example of Valid Override*: AI says "NC" (Normal), but text says "MoCA score: 12/30" (Severe impairment).
    - *Example of INVALID Override*: AI says "MCI", text is vague about ADLs -> **KEEP MCI**. Do not assume impairment if not stated.
3.  **Ambiguity Rule**: If the patient text is missing details (e.g., ADL status is unclear), **YOU MUST RETAIN THE AI PREDICTION**.

**MODE B: Generation (If Anchor Hypothesis IS "NONE")**
*   **Directive**: Diagnose from scratch based solely on the patient data.
*   **Process**: Analyze symptoms -> Assess ADLs -> Determine Label step-by-step.

### 3. Patient Data (Evidence)
### Patient Description
{patient_text}

### Similar Cases Summary
{similar_cases_summary}


Use these validation metrics as confidence context when balancing the AI prior against patient-specific evidence.

### 4. Classification Rules
### Diagnostic Classification Tree
```
Cognitive Impairment Diagnosis
├── NC (Normal Control)
├── MCI (Mild Cognitive Impairment)
│   ├── MCI_A (Amnestic)
│   ├── MCI_AM (Amnestic Multi-domain)
│   ├── CI (pre-MCI)
│   ├── MCI_Na (Non-amnestic)
│   └── MCI_NaM (Non-amnestic Multi-domain)
└── DE (Dementia)
    ├── AD (Alzheimer's Disease)
    ├── LBD (Lewy Body)
    ├── VD (Vascular)
    ├── FTD (Frontotemporal)
    ├── EXC (movement disorder,reversible)
    ├── PSY (psychiatric)
    └── ODE (Other)
```
### 5. Analysis Checkpoints
**Instruction**: 
- In **Verification Mode**: Use these checkpoints to *challenge* the prediction (look for counter-evidence).
- In **Generation Mode**: Use these checkpoints to *derive* the correct label.

**Checkpoint A: The NC vs. Impairment Boundary**
*   *If Prediction is NC*: Does the patient have objective cognitive decline (low MoCA/MMSE) or confirmed biomarkers? (If yes -> Contradiction).
*   *If Prediction is MCI/DE*: Is the cognition objectively normal? (If yes -> Contradiction).

**Checkpoint B: The MCI vs. Dementia (DE) Boundary**
*   *Crucial Test*: **Activities of Daily Living (ADLs)**.
*   *If Prediction is MCI*: Are ADLs significantly impaired due to cognition? (If yes -> Override to DE).
*   *If Prediction is DE*: Are ADLs preserved/independent? (If yes -> Override to MCI).

**Checkpoint C: Subtype Consistency**
*   *If Prediction specifies a subtype (e.g., AD, LBD)*: Does the clinical text mention specific features (e.g., hallucinations for LBD, hippocampal atrophy for AD) that support or refute this?
*   *Mixed Etiology*: If the prediction misses a co-pathology (e.g., prediction is AD, but MRI shows significant vascular damage), add the secondary label.

**Labeling Rules:**
1. **Primary Label**: Must be one of [NC, MCI, DE].
2. **Secondary Labels**: Must be specific subtypes. Use 'NONE' if the primary label is NC.
3. **Override Rule**: You may only change the label if the clinical note contains factual evidence that makes the prior prediction impossible.
{critique_instruction}

### 6. Output Format (STRICT)
You must output a JSON object. 
Before the JSON, you must include a `<thinking>` block to show your step-by-step logic.

Format structure:
<thinking>
1. Symptom Analysis: ...
2. ADL Assessment: ...
3. Consistency Check with ML ({ml_section}): ...
4. Final Conclusion Logic: ...
</thinking>

```json
{{
    "primary_label": "NC" | "MCI" | "DE",
    "secondary_labels": ["subtype1", "subtype2"] or ["NONE"],
    "mHC_prediction": {ml_reference},
    "confidence": 0-100 (Integer),
    "clinical_reasoning": "Concise summary of why this label was chosen..."
    "clinical_recommendations": "Recommended next steps or investigations..."
}}
"""
    return prompt 

def process_single_patient(patient_row, retriever, case_retriever, output_dir="reports", ml_predictions=None, force_overwrite=False, summary_cache=None):
    patient_id = str(patient_row["ID"]).strip()
    safe_id = re.sub(r'[^A-Za-z0-9_-]', '_', patient_id)
    
    if output_dir:
        pattern = os.path.join(output_dir, f"{safe_id}_*.txt")
        existing_files = glob.glob(pattern)
        if existing_files and not force_overwrite:
            valid_files = [f for f in existing_files if "_ERROR_" not in os.path.basename(f)]
            if valid_files:
                print(f"Skipping {patient_id}: valid report exists.")
                return (patient_id, "SKIPPED", "SKIPPED", "SKIPPED", True, "SKIPPED", None)


    workflow = StateGraph(PatientState)
    workflow.add_node("generate", generate_node)
    workflow.add_node("critic", critic_node)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "critic")
    workflow.add_conditional_edges("critic", should_continue, {"retry": "generate", "end": END})
    
    app = workflow.compile()
    
    initial_state = {
        "patient_id": patient_id,
        "patient_text": patient_row["optimized_description"],
        "ml_section": build_ml_prediction_section(patient_id, ml_predictions or {}),
        "attempt_count": 0,
        "critique": None,
        "error": None
    }
    
    try:
        final_state = app.invoke(initial_state)
        result = final_state.get("parsed_result") 
        raw_text = final_state.get("raw_response", "")
        
        if result:
            p_label = result.primary_label 
            if isinstance(result.secondary_labels, list):
                s_labels_list = [l for l in result.secondary_labels if l.upper() != 'NONE']
                s_label = ",".join(s_labels_list) if s_labels_list else "NONE"
            else:
                s_label = str(result.secondary_labels)
            final_report_text = f"Patient ID: {patient_id}\n\n{raw_text}"
        else:
            p_label = "ERROR"
            s_label = "ERROR"
            final_report_text = f"Patient ID: {patient_id}\n\nFAILED.\nError: {final_state.get('error')}\n{raw_text}"

        os.makedirs(output_dir, exist_ok=True)
        
        safe_primary = p_label
        safe_secondary = s_label.replace(',', '+').replace(' ', '')
        safe_secondary = re.sub(r'[^A-Za-z0-9_+\-]', '_', safe_secondary)
        if not safe_secondary: safe_secondary = "NONE"
        
        filename = f"{safe_id}_{safe_primary}_{safe_secondary}.txt"
        
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(final_report_text)
            
        combined = f"{p_label},{s_label}" if s_label != 'NONE' else p_label
        return (patient_id, p_label, s_label, combined, True, None, final_state.get('similar_cases_summary'))

    except Exception as e:
        return (patient_id, 'ERROR', 'ERROR', 'ERROR', False, str(e), None)

def run_report_generation(input_csv=None, output_dir=None, max_samples=None, three_class_mode=None, ml_prediction_csv=None, force_overwrite=False, existing_prediction_csv_path=None):
    if input_csv is None: input_csv = "Feature2Txt/output/feature_descriptions_optimized.csv"
    if output_dir is None: output_dir = "reports"

    print("Loading patient data...")
    df = pd.read_csv(input_csv)

    ml_predictions = load_ml_prediction_data(ml_prediction_csv)
    if ml_predictions: print(f"Loaded ML predictions for {len(ml_predictions)} patients")
    
    if not force_overwrite:
        print("🔍 Scanning for existing valid reports to skip...")
        ids_to_process = []
        skipped_count = 0
        for _, row in df.iterrows():
            patient_id = row.get('ID')
            if pd.isna(patient_id): continue
            safe_id = re.sub(r'[^A-Za-z0-9_-]', '_', str(patient_id))
            pattern = os.path.join(output_dir, f"{safe_id}_*.txt")
            existing_files = glob.glob(pattern)
            has_valid_report = False
            for fpath in existing_files:
                fname = os.path.basename(fpath)
                if "_UNKNOWN_" not in fname and "_ERROR_" not in fname:
                    has_valid_report = True
                    break
            if not has_valid_report: ids_to_process.append(patient_id)
            else: skipped_count += 1
        
        df = df[df['ID'].isin(ids_to_process)]
        print(f"✅ Skipped {skipped_count} valid cases. Remaining to process: {len(df)} (Missing or UNKNOWN)")

    if df.empty:
        print("🎉 All cases have valid reports! Nothing to do.")
        return existing_prediction_csv_path    

    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
        print(f"Limiting processing to {max_samples} samples")
    
    required_cols = ['ID', 'optimized_description']
    for col in required_cols:
        if col not in df.columns:
            if col == 'ID': df[col] = [f'Patient_{i}' for i in range(len(df))]
            elif col == 'optimized_description':
                alt_cols = ['natural_language_description', 'original_description', 'description']
                for alt_col in alt_cols:
                    if alt_col in df.columns:
                        df[col] = df[alt_col]
                        break
                if col not in df.columns: df[col] = "No description available."
            print(f"Warning: Column '{col}' not found, created with default values")
    
    df['ID'] = df['ID'].fillna('Unknown')
    df['optimized_description'] = df['optimized_description'].fillna("No description available.")
    
    print(f"Successfully loaded {len(df)} patient records")
    print("Initializing retrievers...")
    retriever = MedicalRetriever()
    case_retriever = SimilarCaseRetriever()

    summary_cache_path = os.path.join(output_dir, SUMMARY_CACHE_FILENAME)
    summary_cache = load_summary_cache(summary_cache_path)
    summary_updates: Dict[str, str] = {}
    
    num_cores = 30
    print(f"Using {num_cores} processes for parallel processing...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with Pool(num_cores) as pool:
        process_func = partial(
            process_single_patient,
            retriever=retriever,
            case_retriever=case_retriever,
            output_dir=output_dir,
            ml_predictions=ml_predictions,
            force_overwrite=force_overwrite,
            summary_cache=summary_cache,
        )
        
        results = pool.imap(process_func, [row for _, row in df.iterrows()])
        
        success_count = 0
        error_log = []
        prediction_results = []
        
        for i, (patient_id, primary_label, secondary_labels, combined_label, status, error_msg, summary_update) in enumerate(results, 1):
            if status:
                success_count += 1
                prediction_results.append({
                    'ID': patient_id,
                    'primary_label': primary_label,
                    'secondary_labels': secondary_labels,
                    'combined_label': combined_label,
                    'prediction_label': combined_label,
                })
                logging.info(f"Progress: {i}/{len(df)} | Success: {success_count} | Failed: {len(error_log)}")
                if summary_update is not None:
                    summary_updates[str(patient_id).strip()] = summary_update
            else:
                error_log.append((patient_id, error_msg))
                prediction_results.append({
                    'ID': patient_id,
                    'primary_label': 'ERROR',
                    'secondary_labels': 'ERROR',
                    'combined_label': 'ERROR',
                    'prediction_label': 'ERROR',
                })
                print(f"\nError - Patient {patient_id}: {error_msg}")
                if summary_update is not None:
                    summary_updates[str(patient_id).strip()] = summary_update

    if summary_updates:
        summary_cache.update(summary_updates)
        save_summary_cache(summary_cache_path, summary_cache)
    
    prediction_df = pd.DataFrame(prediction_results)
    prediction_ids = pd.Series([], dtype=str)
    if not prediction_df.empty and 'ID' in prediction_df.columns:
        prediction_df['ID'] = prediction_df['ID'].astype(str).str.strip()
        prediction_ids = prediction_df['ID']

    prediction_csv_path = existing_prediction_csv_path or os.path.join(output_dir, "prediction_labels.csv")

    combined_df = prediction_df
    if existing_prediction_csv_path and os.path.exists(prediction_csv_path):
        try:
            existing_df = pd.read_csv(prediction_csv_path)
            if 'ID' in existing_df.columns:
                existing_df['ID'] = existing_df['ID'].astype(str).str.strip()
                remaining_df = existing_df[~existing_df['ID'].isin(prediction_ids)]
                combined_df = pd.concat([remaining_df, prediction_df], ignore_index=True)
            else:
                print(f"Warning: Existing prediction file {prediction_csv_path} missing 'ID' column; replacing file.")
        except Exception as exc:
            print(f"Warning: Failed to read existing prediction file {prediction_csv_path}: {exc}. Replacing file with latest results.")

    combined_df = combined_df.drop_duplicates(subset='ID', keep='last') if 'ID' in combined_df.columns else combined_df
    if 'ID' in combined_df.columns:
        combined_df = combined_df.sort_values(by='ID')

    combined_df.to_csv(prediction_csv_path, index=False, encoding='utf-8')
    print(f"\nPrediction labels updated at: {prediction_csv_path}")
    
    if error_log:
        error_log_path = os.path.join(output_dir, "processing_errors.log")
        with open(error_log_path, "w") as f:
            f.write("PatientID,Error\n")
            for patient_id, error_msg in error_log:
                f.write(f"{patient_id},{error_msg}\n")
    
    print(f"\nProcessing completed! Success: {success_count}/{len(df)}, Failed: {len(error_log)}")
    return prediction_csv_path

def regenerate_bad_cases_for_model(model_name, base_output_root="./adRAG/mask/model", ml_prediction_csv=None):
    bad_case_csv = os.path.join(base_output_root, f"{model_name}_bad_case_descriptions.csv")
    if not os.path.exists(bad_case_csv): return None
    model_output_dir = os.path.join(base_output_root, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    set_global_model(model_name)
    return run_report_generation(
        input_csv=bad_case_csv,
        output_dir=model_output_dir,
        max_samples=None,
        three_class_mode=None,
        ml_prediction_csv=ml_prediction_csv,
        force_overwrite=True,
        existing_prediction_csv_path=os.path.join(model_output_dir, "prediction_labels.csv"),
    )

def process_fold_for_model(model_name: str, fold_index: int, rerun_csv_path: str, ml_prediction_csv_path: str) -> None:
    fold_root = os.path.join(RESULTS_ROOT, f"model-fold{fold_index}")
    model_output_dir = os.path.join(fold_root, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    set_global_model(model_name)
    prediction_csv_path = os.path.join(model_output_dir, "prediction_labels.csv")
    summary_cache_path = os.path.join(model_output_dir, SUMMARY_CACHE_FILENAME)

    current_labels = load_primary_label_map(ml_prediction_csv_path)

    rerun_ids: set[str] = set()
    rerun_df: Optional[pd.DataFrame] = None

    if os.path.exists(rerun_csv_path):
        try:
            rerun_df = pd.read_csv(rerun_csv_path)
            if "ID" in rerun_df.columns:
                rerun_df["ID"] = rerun_df["ID"].astype(str).str.strip()
                rerun_ids = set(rerun_df["ID"].tolist())
        except Exception as exc:
            print(f"Fold {fold_index}: failed to read rerun CSV {rerun_csv_path}: {exc}")
    
    base_outputs = prepare_base_outputs_for_fold(
        model_name=model_name, fold_index=fold_index, rerun_ids=rerun_ids, current_ml_labels=current_labels,resume=True, 
    )

    base_prediction_df: pd.DataFrame = base_outputs["prediction_df"]
    base_summary_df: pd.DataFrame = base_outputs["summary_df"]
    rerun_ids = base_outputs["updated_rerun_ids"]

    if not base_prediction_df.empty:
        base_prediction_df.to_csv(prediction_csv_path, index=False, encoding="utf-8")
        print(f"Fold {fold_index}, model {model_name}: reused predictions for {len(base_prediction_df)} patients")

    if not base_summary_df.empty:
        summary_map = {str(row["ID"]).strip(): row["summary"] for _, row in base_summary_df.iterrows() if isinstance(row.get("summary"), str)}
        if summary_map:
            save_summary_cache(summary_cache_path, summary_map)
            print(f"Fold {fold_index}, model {model_name}: cached {len(summary_map)} similar-case summaries")

    if not rerun_ids:
        print(f"Fold {fold_index}: model {model_name} had no patients requiring regeneration.")
        return

    if rerun_df is None: return

    rerun_df_filtered = rerun_df[rerun_df["ID"].isin(rerun_ids)].copy()
    if rerun_df_filtered.empty: return

    temp_rerun_path = rerun_csv_path
    if len(rerun_df_filtered) != len(rerun_df):
        temp_rerun_path = os.path.join(model_output_dir, f"rerun_fold{fold_index}_{model_name}.csv")
        rerun_df_filtered.to_csv(temp_rerun_path, index=False, encoding="utf-8")

    set_global_model(model_name)
    print(f"\n=== Fold {fold_index}: regenerating {len(rerun_df_filtered)} cases for model {model_name} ===")

    run_report_generation(
        input_csv=temp_rerun_path,
        output_dir=model_output_dir,
        max_samples=None,
        three_class_mode=None,
        ml_prediction_csv=ml_prediction_csv_path,
        force_overwrite=False,
        existing_prediction_csv_path=prediction_csv_path,
    )

    if temp_rerun_path != rerun_csv_path:
        try: os.remove(temp_rerun_path)
        except OSError: pass

def main(input_csv=None, output_dir=None, max_samples=None, three_class_mode=None, ml_prediction_csv=None, force_overwrite=False):
    return run_report_generation(input_csv, output_dir, max_samples, three_class_mode, ml_prediction_csv, force_overwrite)

if __name__ == "__main__":
    test_models = [
    ]

    merged_csv_path = "./adRAG/data/val_ALL_folds_merged_ratio0.csv"
    desc_csv_path = "./adRAG/mask0/feature_descriptions_optimized.csv"
    
    print(f"Loading merged validation data from {merged_csv_path}")
    merged_df = pd.read_csv(merged_csv_path)
    
    print(f"Loading optimized descriptions from {desc_csv_path}")
    desc_df = pd.read_csv(desc_csv_path)

    
    if 'Fold_ID' not in merged_df.columns:
        raise ValueError("The merged CSV must contain a 'Fold_ID' column!")
        
    fold_indices = sorted(merged_df['Fold_ID'].dropna().unique().astype(int))
    
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    for fold_index in fold_indices:
        print(f"\n==========================================")
        print(f"===== Preparing Data for Fold {fold_index} =====")
        print(f"==========================================")
        
        fold_df = merged_df[merged_df['Fold_ID'] == fold_index]
        
        fold_desc_df = desc_df[desc_df['ID'].isin(fold_df['ID'])]
        
        temp_desc_path = os.path.join(RESULTS_ROOT, f"temp_descriptions_fold{fold_index}.csv")
        temp_y_path = os.path.join(RESULTS_ROOT, f"temp_y_fold{fold_index}.csv")
        
        fold_desc_df.to_csv(temp_desc_path, index=False)
        fold_df.to_csv(temp_y_path, index=False)
        
        print(f"Fold {fold_index}: Saved {len(fold_desc_df)} descriptions and {len(fold_df)} labels to temporary files.")

        for model_name in test_models:
            process_fold_for_model(
                model_name=model_name,
                fold_index=fold_index,
                rerun_csv_path=temp_desc_path,
                ml_prediction_csv_path=temp_y_path,
            )
