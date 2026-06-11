
import pandas as pd
import json
import os
import logging
import sys
import time
import argparse
import re
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import httpx
from openai import OpenAI
from poe_client import PoeClient


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'UniBrain-master'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureToTextConverter:
    
    def __init__(self, 
                 conversion_rules_path: str,
                 sample_data_path: str,
                 output_dir: str,
                 enable_mri_diagnosis: bool = True):
        
        self.conversion_rules_path = conversion_rules_path
        self.sample_data_path = sample_data_path
        self.output_dir = Path(output_dir)
        self.enable_mri_diagnosis = enable_mri_diagnosis
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.conversion_rules = {}
        self.sample_data = None
        self.mri_diagnosis_results = {}
        self.data_loaded = False
        self._mri_load_attempted = False
        
    def load_data(self):
        if self.data_loaded: return
        logger.info("Loading data...")
        
        with open(self.conversion_rules_path, 'r', encoding='utf-8') as f:
            self.conversion_rules = json.load(f)
        
        self.sample_data = pd.read_csv(self.sample_data_path)

        logger.info("Preprocessing data: Converting '.' values to NA...")
        dot_count = 0
        for col in self.sample_data.columns:
            if col != 'ID':
                mask = self.sample_data[col] == '.'
                dot_count += mask.sum()
                self.sample_data.loc[mask, col] = pd.NA
        
        if self.enable_mri_diagnosis:
            self._load_mri_diagnosis()
        
        self.data_loaded = True
    
    def _load_mri_diagnosis(self):
        if self._mri_load_attempted: return
        self._mri_load_attempted = True
        try:
            logger.info("Loading MRI diagnosis results from UniBrain...")
            original_cwd = os.getcwd()
            unibrain_dir = os.path.join(os.path.dirname(__file__), '..', 'UniBrain-master')
            os.chdir(unibrain_dir)
            
            import mri_diagnosis
            self.mri_diagnosis_results = mri_diagnosis.main(csv_file=self.sample_data_path)
            logger.info(f"Loaded MRI diagnosis for {len(self.mri_diagnosis_results)} patients")
        except Exception as e:
            logger.warning(f"Failed to load MRI diagnosis: {e}")
            self.mri_diagnosis_results = {}
        finally:
            os.chdir(original_cwd)
    
    def _normalize_value_for_mapping(self, value: Any, rule: Dict) -> str:
        value_mappings = rule.get("value_mappings", {})
        if not value_mappings: return str(value)
        try:
            if isinstance(value, (int, float)):
                if float(value).is_integer():
                    int_str = str(int(float(value)))
                    if int_str in value_mappings: return int_str
                    float_str = str(float(value))
                    if float_str in value_mappings: return float_str
                    return int_str
                else:
                    return str(value)
            else:
                try:
                    num_value = float(value)
                    if num_value.is_integer():
                        int_str = str(int(num_value))
                        if int_str in value_mappings: return int_str
                        if str(value) in value_mappings: return str(value)
                        return int_str
                    else:
                        return str(value)
                except (ValueError, TypeError):
                    return str(value)
        except (ValueError, TypeError):
            return str(value)
    
    def convert_samples(self, mode: str = "default"):
        logger.info(f"Converting samples to natural language (mode: {mode})...")
        results = []
        for idx, row in self.sample_data.iterrows():
            sample_id = row.get('ID', f'Sample_{idx}')
            description = self._convert_single_sample(row, mode)
            results.append({
                'ID': sample_id,
                'natural_language_description': description
            })
        
        results_df = pd.DataFrame(results)
        output_path = self.output_dir / f"feature_descriptions_{mode}.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Intermediate raw descriptions saved to {output_path}")
        return results_df
    
    def _convert_single_sample(self, sample_row: pd.Series, mode: str = "default") -> str:
        descriptions = []
        priority_vars = ['his_SEX', 'his_RACE', 'his_EDUC', 'his_MARISTAT', 'ph_NACCBMI', 'ph_WEIGHT', 'ph_HEIGHT', 'bat_NACCMMSE']

        processed_vars = set()
        
        for var_name in priority_vars:
            if var_name in self.conversion_rules:
                desc = self._convert_variable(sample_row, var_name, mode)
                if desc:
                    descriptions.append(desc)
                    processed_vars.add(var_name)
        
        for var_name in self.conversion_rules.keys():
            if var_name not in processed_vars:
                desc = self._convert_variable(sample_row, var_name, mode)
                if desc:
                    descriptions.append(desc)
        
        if self.enable_mri_diagnosis and self.mri_diagnosis_results:
            patient_id = sample_row.get('ID', '')
            mri_desc = self._get_mri_diagnosis_description(patient_id)
            if mri_desc:
                descriptions.append(mri_desc)
        
        return ". ".join(descriptions) + "." if descriptions else ""
    
    def _get_mri_diagnosis_description(self, patient_id: str) -> str:
        if patient_id not in self.mri_diagnosis_results: return ""
        diagnosis_list = self.mri_diagnosis_results[patient_id]
        if not diagnosis_list: return "该患者的MRI图像经UniBrain模型的诊断结果缺失"
        return f"该患者的MRI图像经UniBrain模型的诊断结果为：{'、'.join(diagnosis_list)}"
    
    def _convert_variable(self, sample_row: pd.Series, var_name: str, mode: str = "default") -> str:
        if var_name not in self.conversion_rules: return ""
        
        rule = self.conversion_rules[var_name]
        english_name = rule.get('english_name', var_name).lower()
        template = rule.get("template", "The patient has {VALUE}")
        
        actual_col_name = var_name
        if var_name not in sample_row.index:
            stripped_name = var_name.split('_')[-1] if '_' in var_name else var_name
            if stripped_name in sample_row.index:
                actual_col_name = stripped_name
            else:
                if mode == "complete":
                    return f"The status of {english_name} is unknown."
                return ""
                
        value = sample_row.get(actual_col_name)
        
        if pd.isna(value) or str(value).strip() == '':
            if mode == "default": 
                return ""
            elif mode == "complete":
                return f"The status of {english_name} is unknown."
                
        value_str = self._normalize_value_for_mapping(value, rule)
        value_mappings = rule.get("value_mappings", {})
        
        if value_str in value_mappings:
            mapped_value = value_mappings[value_str]
        elif "default" in value_mappings:
            default_template = value_mappings["default"]
            if "{VALUE}" in default_template:
                mapped_value = default_template.replace("{VALUE}", str(value))
            else:
                mapped_value = default_template
        else:
            mapped_value = str(value)
            
        result = template.replace("{VALUE}", str(mapped_value))
        
        if "{VALUE}" in result: 
            result = result.replace("{VALUE}", str(value))
            
        return result
    
    def process(self, mode: str = "default"):
        self.load_data()
        return self.convert_samples(mode)


def clean_llm_output(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("thinking") or stripped.startswith(">"):
            continue
        cleaned_lines.append(line)
    cleaned_text = "\n".join(cleaned_lines).strip()
    return cleaned_text.replace("Thinking...", "").replace("Thinking…", "")

def get_optimization_prompt():
    return """Act as a clinical data summarizer. Process the following patient characteristics into a structured medical summary. Apply these rules strictly:

1.  **Categorization:** Group findings into these EXACT sections:
    *   **Demographics & Baseline:** Sex, Race, Education, Marital Status, Living Situation, Language (if valid), Handedness.
    *   **Anthropometrics & Vitals:** BMI, Weight, Height, Systolic BP, Diastolic BP, Resting Heart Rate.
    *   **Medical History:** Explicitly stated active/inactive diagnoses (e.g., Arthritis, Thyroid Disease, Atrial Fibrillation - state presence/absence clearly), History (e.g., Stroke, TIA, MI, Seizures, TBI, Alcohol Abuse), Smoking Status, Alcohol Use (Frequency).
    *   **Medications:** List all medication classes *reported as used* at the visit. Group logically (e.g., Antihypertensives, Anticoagulants, Antidepressants). Explicitly state "No Use Reported" for classes mentioned but not taken.
    *   **Family History:** Focus on cognitive impairment, dementia types (AD, FTLD), and specific mutations (e.g., APP).
    *   **Neurological & Motor Findings:** Abnormal exam findings (e.g., Bradykinesia, Postural Instability, Gait Abnormality, Slowing of fine motor movements), Parkinsonian signs, tremor, rigidity, specific syndromes ruled in/out. Include sensory/motor deficits if present.
    *   **Cognitive & Behavioral Status:** Global cognitive status (e.g., Normal for age), MoCA score & key subscores (highlighting *severe impairment in naming*), other cognitive test results (e.g., Trails, Fluency, Recall), Behavioral symptoms (Presence/Absence of depression, anxiety, hallucinations, apathy, etc. - note contradictions like "active depression" vs. GDS=0).
    *   **Functional Status & ADLs:** Independence level, difficulties with specific tasks (Financial, Cooking, Travel, etc.), Incontinence (Urinary/Bowel), Vision/Hearing (with/without aids).
    *   **MRI Imaging Findings:** Include MRI diagnostic results from UniBrain or other imaging models, listing specific findings (e.g., focal ischemia, meningioma, normal findings) and their clinical significance.
    *   **Contradictions:** Enumerate findings that conflict with typical cognitive impairment patterns or contradict each other (e.g., acute delirium features with preserved daily function, severe MoCA impairment but intact instrumental ADLs, depression reported but mood scale negative). Provide brief rationale for why each item is contradictory.
    *   **Key Anomalies & Flags:** Concise list of the MOST clinically significant unexpected, contradictory, or critical findings (e.g., APP mutation + no cognitive impairment, Active depression + GDS=0, Severe naming deficit + normal MoCA, Arthritis type/region, Gait description, Gradual onset motor symptoms, Falls, conflicts between objective testing and reported function).

2.  **Optimization & Merging:**
    *   **Remove Redundancy:** Merge duplicate entries.
    *   **Correct Errors:** Fix nonsensical phrases.
    *   **Eliminate Meaningless Data:** Omit features with clearly invalid/nonsensical values (e.g., "primary language is 0.0").
    *   **Clarify Negatives & Positives:** Convert vague statements to concise medical facts.
    *   **Resolve Contradictions:** Explicitly analyse and surface contradictions.
    *   **Precision:** Report specific numerical values where clinically relevant.

3.  **Focus on Anomalies:** Emphasize findings that are abnormal or unexpected.

4.  **Conciseness:** Use clear, concise medical language. Use bullet points.

**Input Data:**
{description}

Please provide the optimized structured medical summary following the exact format above. Do **not** include any reasoning narrative, chain-of-thought; output only the structured sections requested."""

def process_single_description(args):
    patient_id, description = args
    try:
        optimized_description = call_poe_api(description)
        time.sleep(0.1)
        return patient_id, optimized_description
    except Exception as e:
        return patient_id, f"Error: {str(e)}"


def run_pipeline(args):
    logger.info("="*50)
    logger.info("🚀 Starting End-to-End Clinical Data Pipeline")
    logger.info("="*50)

    logger.info(f"\n[Phase 1/2] Converting Tabular Features to Raw Text (mode: {args.mode}) ...")
    converter = FeatureToTextConverter(
        conversion_rules_path=args.rules,
        sample_data_path=args.samples,
        output_dir=args.output_dir,
        enable_mri_diagnosis=args.enable_mri
    )
    
    raw_df = converter.process(mode=args.mode)

    if raw_df.empty:
        logger.error("Phase 1 yielded no data. Exiting.")
        return

    logger.info(f"\n[Phase 2/2] Optimizing {len(raw_df)} Descriptions via LLM ({POE_MODEL_NAME}) ...")
    
    descriptions_to_process = []
    for _, row in raw_df.iterrows():
        pid = row.get('ID')
        desc = row.get('natural_language_description', "No description available.")
        descriptions_to_process.append((pid, desc))

    optimized_results = []
    max_workers = args.max_workers or max(1, os.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(process_single_description, data): data[0] for data in descriptions_to_process}
        
        for future in tqdm(future_to_id, desc="Optimizing"):
            patient_id = future_to_id[future]
            try:
                _, optimized_desc = future.result()
                orig_desc = raw_df[raw_df['ID'] == patient_id]['natural_language_description'].iloc[0]
                optimized_results.append({
                    'ID': patient_id,
                    'original_description': orig_desc,
                    'optimized_description': optimized_desc
                })
            except Exception as e:
                logger.error(f"Failed on patient {patient_id}: {e}")
                orig_desc = raw_df[raw_df['ID'] == patient_id]['natural_language_description'].iloc[0]
                optimized_results.append({
                    'ID': patient_id,
                    'original_description': orig_desc,
                    'optimized_description': f"Error: {str(e)}"
                })

    final_df = pd.DataFrame(optimized_results)
    final_output_path = os.path.join(args.output_dir, "feature_descriptions_optimized.csv")
    final_df.to_csv(final_output_path, index=False, encoding='utf-8')
    
    success_count = len([r for r in optimized_results if not r['optimized_description'].startswith('Error:')])
    
    logger.info("\n" + "="*50)
    logger.info("✅ Pipeline Completed Successfully!")
    logger.info(f"Total processed : {len(final_df)}")
    logger.info(f"Success rate    : {success_count/len(final_df)*100:.1f}%")
    logger.info(f"Final output    : {final_output_path}")
    logger.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='End-to-End Clinical Data Pipeline')
    parser.add_argument('--rules', default='./adRAG/Feature2Txt/data/conversion_rules_1225.json', help='Path to JSON rules')
    parser.add_argument('--samples', default='adRAG/data/val_ALL_folds_merged_ratio0.csv', help='Input CSV')
    parser.add_argument('--output-dir', default='./adRAG/mask0', help='Output directory')
    parser.add_argument('--disable-mri', dest='enable_mri', action='store_false', help='Disable MRI diagnosis')
    parser.add_argument('--max-workers', type=int, default=20, help='Max concurrent LLM API calls')
    parser.add_argument('--mode', choices=['default', 'complete'], default='default', help='Conversion mode')    
    parser.set_defaults(enable_mri=True)
    
    args = parser.parse_args()
    run_pipeline(args)
