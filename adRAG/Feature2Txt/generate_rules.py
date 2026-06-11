
import pandas as pd
try:
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
except ImportError:
    print("警告: 无法导入配置文件config.py，请确保文件存在且包含API配置")
    DEEPSEEK_API_KEY = DEEPSEEK_API_KEY
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
import json
import os
import requests
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")
    TQDM_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GLOBAL_API_KEY = ""
GLOBAL_BASE_URL = ""

def process_single_variable(args: Tuple[str, Dict, List, str, str]) -> Tuple[str, Optional[Dict]]:
    var_name, var_desc, sample_values, api_key, base_url = args
    
    llm_client = LLMClient(api_key, base_url)
    
    var_info = f"""
Variable Name: {var_name}
Variable Type: {var_desc.get('variable_type', '')}
Description: {var_desc.get('short_descriptor', '')}
Data Type: {var_desc.get('data_type', '')}
Allowable Codes: {var_desc.get('allowable_codes', '')}
Detailed Description: {var_desc.get('description_or_derivation', '')}
Sample Values: {sample_values}
"""
    
    prompt = f"""
Please design an English natural language conversion template for the following medical variable:

{var_info}

Please return the result in JSON format:
{{
  "english_name": "English name of the variable",
  "value_mappings": {{
    "1": "English description for value 1",
    "2": "English description for value 2",
    "default": "'s [attribute] is {{VALUE}}"
  }},
  "template": "The patient {{VALUE}}",
  "description": "Brief English description of the variable"
}}

Requirements:
1. Use {{VALUE}} as placeholder for the value in the template
2. Include medical standard English terminology
3. For the "default" key in value_mappings, ALWAYS use "{{VALUE}}" so unmapped values will show their original value
4. Template should be a complete sentence describing patient status
5. Replace [attribute] in template with appropriate attribute name (e.g., "age", "sex", "race", etc.)
6. Return only JSON, no other explanations

Examples of good templates:
- "The patient is {{VALUE}} years old"
- "The patient's sex is {{VALUE}}"
- "The patient's race is {{VALUE}}"
- "The patient has {{VALUE}}"

Important: The "default" value in value_mappings should ALWAYS be exactly "{{VALUE}}" (not "unknown something").
"""

    try:
        response = llm_client.chat_completion([
            {"role": "user", "content": prompt}
        ])
        
        cleaned_response = extract_json_from_response(response)
        result = json.loads(cleaned_response)
        
        if "default" not in result.get("value_mappings", {}):
            result["value_mappings"]["default"] = f"unknown {result.get('english_name', var_name).lower()}"
        
        
        return (var_name, result)
        
    except Exception as e:
        print(f"[{var_name}] Failed to generate rule: {e}")
        basic_rule = {
            "english_name": var_name.replace('_', ' ').title(),
            "value_mappings": {"default": f"unknown {var_name.replace('_', ' ').lower()}"},
            "template": "The patient has {VALUE}",
            "description": f"Information about {var_name.replace('_', ' ').lower()}"
        }
        print(f"[{var_name}] Using basic rule as fallback")
        return (var_name, basic_rule)


def extract_json_from_response(response: str) -> str:
    response = response.strip()
    if response.startswith('```json'):
        response = response[7:]
    if response.startswith('```'):
        response = response[3:]
    if response.endswith('```'):
        response = response[:-3]
    
    start_idx = response.find('{')
    end_idx = response.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return response[start_idx:end_idx + 1]
    
    return response.strip()


class LLMClient:
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = "deepseek-chat", 
                       max_tokens: int = 2000,
                       temperature: float = 0.1) -> str:
        try:
            url = f"{self.base_url}/v1/chat/completions"
            data = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return ""


class RuleGenerator:
    
    def __init__(self, 
                 variable_descriptions_path: str = "data/variable_descriptions_updated.csv",
                 sample_data_path: str = "data/selected_samples_cleaned.csv",
                 output_dir: str = "output",
                 max_workers: int = None):
        
        self.variable_descriptions_path = variable_descriptions_path
        self.sample_data_path = sample_data_path
        self.output_dir = Path(output_dir)
        
        if max_workers is None:
            self.max_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.max_workers = max_workers
            
        print(f"Using {self.max_workers} worker processes (CPU cores: {multiprocessing.cpu_count()})")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_client = LLMClient(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL)
        
        self.variable_descriptions = {}
        self.sample_data = None
        self.conversion_rules = {}
        
    def load_data(self):
        logger.info("Loading data...")
        
        var_df = pd.read_csv(self.variable_descriptions_path)
        for _, row in var_df.iterrows():
            self.variable_descriptions[row['variable']] = {
                'variable_type': row['variable_type'],
                'short_descriptor': row['short_descriptor'],
                'data_type': row['data_type'],
                'allowable_codes': str(row['allowable_codes']) if pd.notna(row['allowable_codes']) else "",
                'description_or_derivation': str(row['description_or_derivation']) if pd.notna(row['description_or_derivation']) else ""
            }
        
        self.sample_data = pd.read_csv(self.sample_data_path)
        
        logger.info(f"Loaded {len(self.variable_descriptions)} variable descriptions")
        logger.info(f"Loaded {len(self.sample_data)} samples")
    
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
    
    def generate_rules(self):
        logger.info("Generating conversion rules with parallel processing...")
        
        variable_samples = {}
        variables_to_process = []
        
        for var in self.variable_descriptions.keys():
            if var in self.sample_data.columns:
                unique_values = self.sample_data[var].dropna().unique()
                sample_values = list(unique_values)[:10]
                variable_samples[var] = sample_values
                
                variables_to_process.append((
                    var, 
                    self.variable_descriptions[var], 
                    sample_values,
                    DEEPSEEK_API_KEY,
                    DEEPSEEK_BASE_URL
                ))
        
        logger.info(f"Processing {len(variables_to_process)} variables with {self.max_workers} workers...")
        
        completed_count = 0
        total_count = len(variables_to_process)
        
        if TQDM_AVAILABLE:
            progress_bar = tqdm(total=total_count, desc="Generating rules", unit="vars")
        else:
            print(f"Processing {total_count} variables...")
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_var = {
                executor.submit(process_single_variable, args): args[0] 
                for args in variables_to_process
            }
            
            for future in as_completed(future_to_var):
                var_name = future_to_var[future]
                try:
                    result_var_name, rule = future.result()
                    if rule:
                        self.conversion_rules[result_var_name] = rule
                        completed_count += 1
                    
                    if TQDM_AVAILABLE:
                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            completed=completed_count,
                            success_rate=f"{completed_count/max(1, completed_count + len([f for f in future_to_var if not f.done()])):.1%}"
                        )
                    else:
                        print(f"Progress: {completed_count}/{total_count} ({completed_count/total_count*100:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"Error processing variable {var_name}: {e}")
                    if TQDM_AVAILABLE:
                        progress_bar.update(1)
        
        if TQDM_AVAILABLE:
            progress_bar.close()
        
        self._save_conversion_rules()
        
        logger.info(f"Successfully generated rules for {len(self.conversion_rules)} variables")
        print(f"\nRule generation summary:")
        print(f"  Total variables processed: {total_count}")
        print(f"  Successful rules generated: {len(self.conversion_rules)}")
        print(f"  Success rate: {len(self.conversion_rules)/total_count*100:.1f}%")
    
    def _save_conversion_rules(self):
        output_path = self.output_dir / "conversion_rules.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversion_rules, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Conversion rules saved to {output_path}")
    
    def process(self):
        logger.info("Starting rule generation process...")
        
        self.load_data()
        
        self.generate_rules()
        
        logger.info("Rule generation completed!")
        return self.conversion_rules


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate conversion rules for medical variables')
    parser.add_argument('--var_desc', default='data/variable_descriptions_updated.csv',
                       help='Path to variable descriptions CSV file')
    parser.add_argument('--samples', default='data/selected_samples_cleaned.csv',
                       help='Path to sample data CSV file')
    parser.add_argument('--output', default='output',
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=None,
                       help=f'Number of worker processes (default: {max(1, multiprocessing.cpu_count() - 1)} = CPU cores - 1)')
    
    args = parser.parse_args()
    
    total_cores = multiprocessing.cpu_count()
    if args.workers is None:
        workers = max(1, total_cores - 1)
        print(f"Auto-detected {total_cores} CPU cores, using {workers} workers (cores - 1)")
    else:
        workers = args.workers
        print(f"Using {workers} workers (total cores: {total_cores})")
    
    generator = RuleGenerator(
        variable_descriptions_path=args.var_desc,
        sample_data_path=args.samples,
        output_dir=args.output,
        max_workers=workers
    )
    
    try:
        start_time = time.time()
        rules = generator.process()
        end_time = time.time()
        
        print(f"\nRule generation completed!")
        print(f"Generated rules for {len(rules)} variables")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print(f"Average time per variable: {(end_time - start_time) / len(rules):.2f} seconds")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        total_cores = multiprocessing.cpu_count()
        workers = max(1, total_cores - 1)
        print(f"Auto-detected {total_cores} CPU cores, using {workers} workers (cores - 1)")
        
        generator = RuleGenerator(max_workers=workers)
        
        try:
            start_time = time.time()
            rules = generator.process()
            end_time = time.time()
            
            print(f"\nRule generation completed!")
            print(f"Generated rules for {len(rules)} variables")
            print(f"Processing time: {end_time - start_time:.2f} seconds")
            print(f"Average time per variable: {(end_time - start_time) / len(rules):.2f} seconds")
            print(f"Results saved to: output/")
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
