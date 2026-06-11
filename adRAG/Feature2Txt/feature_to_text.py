

import pandas as pd
import json
import os
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'UniBrain-master'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureToTextConverter:
    
    def __init__(self, 
                 conversion_rules_path: str = "./adRAG/Feature2Txt/data/conversion_rules.json",
                 sample_data_path: str = "./adRAG/data/selected_samples_cleaned.csv",
                 output_dir: str = "./adRAG/output",
                 enable_mri_diagnosis: bool = True,
                 patient_data_file: Optional[str] = None):
        
        self.conversion_rules_path = conversion_rules_path
        self.sample_data_path = sample_data_path
        self.patient_data_file = patient_data_file or sample_data_path
        self.output_dir = Path(output_dir)
        self.enable_mri_diagnosis = enable_mri_diagnosis
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.conversion_rules = {}
        self.sample_data = None
        self.mri_diagnosis_results = {}
        self.data_loaded = False
        self._mri_load_attempted = False
        
    def load_data(self):
        if self.data_loaded:
            logger.debug("Data already loaded, skipping reload")
            return

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
        
        logger.info(f"Converted {dot_count} '.' values to NA")
        logger.info(f"Loaded {len(self.conversion_rules)} conversion rules")
        logger.info(f"Loaded {len(self.sample_data)} samples")
        
        if self.enable_mri_diagnosis:
            self._load_mri_diagnosis()
        
        self.data_loaded = True
    
    def _load_mri_diagnosis(self):
        if self._mri_load_attempted:
            logger.debug("MRI diagnosis already loaded or attempted, skipping")
            return

        self._mri_load_attempted = True
        try:
            logger.info("Loading MRI diagnosis results...")
            
            original_cwd = os.getcwd()
            
            unibrain_dir = os.path.join(os.path.dirname(__file__), '..', 'UniBrain-master')
            os.chdir(unibrain_dir)
            
            import mri_diagnosis
            
            self.mri_diagnosis_results = mri_diagnosis.main(csv_file=self.sample_data_path)
            logger.info(f"Loaded MRI diagnosis for {len(self.mri_diagnosis_results)} patients")
            
        except Exception as e:
            logger.warning(f"Failed to load MRI diagnosis: {e}")
            logger.warning("Continuing without MRI diagnosis information")
            self.mri_diagnosis_results = {}
        finally:
            os.chdir(original_cwd)
    
    def _normalize_value_for_mapping(self, value: Any, rule: Dict) -> str:
        value_mappings = rule.get("value_mappings", {})
        
        if not value_mappings:
            return str(value)
        
        try:
            if isinstance(value, (int, float)):
                if float(value).is_integer():
                    int_value = int(float(value))
                    int_str = str(int_value)
                    
                    if int_str in value_mappings:
                        return int_str
                    
                    float_str = str(float(value))
                    if float_str in value_mappings:
                        return float_str
                    
                    return int_str
                else:
                    return str(value)
            else:
                try:
                    num_value = float(value)
                    if num_value.is_integer():
                        int_value = int(num_value)
                        int_str = str(int_value)
                        
                        if int_str in value_mappings:
                            return int_str
                        
                        if str(value) in value_mappings:
                            return str(value)
                        
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
            
            if idx % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(self.sample_data)} samples")
        
        results_df = pd.DataFrame(results)
        output_filename = f"feature_descriptions_{mode}.csv"
        output_path = self.output_dir / output_filename
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Feature descriptions saved to {output_path}")
        return results_df
    
    def _convert_single_sample(self, sample_row: pd.Series, mode: str = "default") -> str:
        descriptions = []
        
        priority_vars = [
            'SEX', 'RACE', 'EDUC', 'MARISTAT',
            'NACCBMI', 'WEIGHT', 'HEIGHT', 'NACCMMSE'
        ]
        
        processed_vars = set()
        for var_name in priority_vars:
            if var_name in self.conversion_rules and var_name in sample_row.index:
                desc = self._convert_variable(sample_row, var_name, mode)
                if desc:
                    descriptions.append(desc)
                    processed_vars.add(var_name)
        
        for var_name in self.conversion_rules.keys():
            if var_name not in processed_vars and var_name in sample_row.index:
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
        if patient_id not in self.mri_diagnosis_results:
            return ""
        
        diagnosis_list = self.mri_diagnosis_results[patient_id]
        
        if not diagnosis_list:
            return "该患者的MRI图像经UniBrain模型的诊断结果缺失"
        
        diagnosis_str = "、".join(diagnosis_list)
        return f"该患者的MRI图像经UniBrain模型的诊断结果为：{diagnosis_str}"
    
    def _convert_variable(self, sample_row: pd.Series, var_name: str, mode: str = "default") -> str:
        if var_name not in self.conversion_rules:
            return ""
        
        if var_name not in sample_row.index:
            if mode == "complete":
                rule = self.conversion_rules[var_name]
                value_mappings = rule.get("value_mappings", {})
                default_desc = value_mappings.get("default", f"unknown {rule.get('english_name', var_name).lower()}")
                template = rule.get("template", "The patient has {VALUE}")
                return template.replace("{VALUE}", default_desc)
            else:
                return ""
        
        rule = self.conversion_rules[var_name]
        value = sample_row.get(var_name)
        
        if pd.isna(value):
            if mode == "default":
                return ""
            elif mode == "complete":
                value_mappings = rule.get("value_mappings", {})
                default_desc = value_mappings.get("default", f"unknown {rule.get('english_name', var_name).lower()}")
                template = rule.get("template", "The patient has {VALUE}")
                return template.replace("{VALUE}", default_desc)
        
        value_str = self._normalize_value_for_mapping(value, rule)
        
        template = rule.get("template", "The patient has {VALUE}")
        value_mappings = rule.get("value_mappings", {})
        
        if value_str in value_mappings:
            mapped_value = value_mappings[value_str]
        else:
            if "default" in value_mappings:
                default_template = value_mappings["default"]
                if "{VALUE}" in default_template:
                    mapped_value = default_template.replace("{VALUE}", str(value))
                else:
                    mapped_value = default_template
                logger.debug(f"Variable {var_name}: value '{value_str}' not found in mapping, using default")
            else:
                mapped_value = str(value)
                logger.debug(f"Variable {var_name}: value '{value_str}' not found in mapping and no default available")
        
        result = template.replace("{VALUE}", mapped_value)
        
        if "{VALUE}" in result:
            result = result.replace("{VALUE}", str(value))
            
        return result
    
    def process_patient_features(self, patient_data: Dict[str, Any], patient_id: str = None, mode: str = "default") -> str:
        if not self.conversion_rules:
            self.load_data()
        
        if patient_id:
            patient_data['ID'] = patient_id
        patient_series = pd.Series(patient_data)
        
        description = self._convert_single_sample(patient_series, mode)
        
        return description
    
    def process(self, mode: str = "default"):
        logger.info(f"Starting feature to text conversion (mode: {mode})...")
        
        self.load_data()
        
        results = self.convert_samples(mode)
        
        logger.info("Feature to text conversion completed!")
        return results
    
    def batch_convert(self, modes: List[str] = None):
        if modes is None:
            modes = ["default", "complete"]
        
        results = {}
        for mode in modes:
            logger.info(f"Processing mode: {mode}")
            result = self.process(mode)
            results[mode] = result
        
        return results


def run_feature_to_text(sample_csv=None, rules_json=None, output_dir=None, modes=None, 
                        enable_mri_diagnosis=True):

    if sample_csv is None:
        sample_csv = "./adRAG/data/selected_samples_cleaned.csv"
    if rules_json is None:
        rules_json = "./adRAG/output/conversion_rules.json"
    if output_dir is None:
        output_dir = "./adRAG/output"
    if modes is None:
        modes = ["default", "complete"]
    
    converter = FeatureToTextConverter(
        conversion_rules_path=rules_json,
        sample_data_path=sample_csv,
        output_dir=output_dir,
        enable_mri_diagnosis=enable_mri_diagnosis
    )
    
    results = converter.batch_convert(modes)
    
    logger.info(f"Feature to text conversion completed!")
    for mode, result_df in results.items():
        logger.info(f"Mode '{mode}': {len(result_df)} samples processed")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Convert medical feature vectors to natural language text')
    parser.add_argument('--rules', default='./adRAG/Feature2Txt/data/conversion_rules.json',
                       help='Path to conversion rules JSON file')
    parser.add_argument('--samples', default='./adRAG/data/外部队列/adni_merged_probs_fold0_final.csv',
                       help='Path to sample data CSV file')
    parser.add_argument('--output', default='./adRAG/ADNI_results',
                       help='Output directory')
    parser.add_argument('--mode', choices=['default', 'complete', 'both'], default='both',
                       help='Conversion mode: default (ignore NA), complete (use default for NA), or both')
    parser.add_argument('--enable-mri', action='store_true', default=True,
                       help='Enable MRI diagnosis results (default: True)')
    parser.add_argument('--disable-mri', dest='enable_mri', action='store_false',
                       help='Disable MRI diagnosis results')
    
    args = parser.parse_args()
    
    converter = FeatureToTextConverter(
        conversion_rules_path=args.rules,
        sample_data_path=args.samples,
        output_dir=args.output,
        enable_mri_diagnosis=args.enable_mri,
    )
    
    try:
        if args.mode == 'both':
            results = converter.batch_convert(['default', 'complete'])
            print(f"\nFeature to text conversion completed!")
            for mode, result_df in results.items():
                print(f"Mode '{mode}': {len(result_df)} samples processed")
        else:
            results = converter.process(args.mode)
            print(f"\nFeature to text conversion completed!")
            print(f"Mode '{args.mode}': {len(results)} samples processed")
        
        print(f"Results saved to: {args.output}")
        
        if args.mode == 'both':
            example_df = results['default']
        else:
            example_df = results
            
        if len(example_df) > 0:
            print(f"\nExample output:")
            print(f"ID: {example_df.iloc[0]['ID']}")
            desc = example_df.iloc[0]['natural_language_description']
            print(f"Description: {desc[:300]}..." if len(desc) > 300 else f"Description: {desc}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        converter = FeatureToTextConverter()
        
        try:
            results = converter.batch_convert(['default', 'complete'])
            print(f"\nFeature to text conversion completed!")
            for mode, result_df in results.items():
                print(f"Mode '{mode}': {len(result_df)} samples processed")
            print(f"Results saved to: ./adRAG/output/")
            
            example_df = results['default']
            if len(example_df) > 0:
                print(f"\nExample output (default mode):")
                print(f"ID: {example_df.iloc[0]['ID']}")
                desc = example_df.iloc[0]['natural_language_description']
                print(f"Description: {desc[:300]}..." if len(desc) > 300 else f"Description: {desc}")
                
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
