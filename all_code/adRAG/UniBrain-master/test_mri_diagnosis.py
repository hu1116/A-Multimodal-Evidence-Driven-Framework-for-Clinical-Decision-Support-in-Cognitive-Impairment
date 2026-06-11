
import os
import json
import pandas as pd
import sys
import time
from pathlib import Path


current_script_path = os.path.dirname(os.path.abspath(__file__))
if current_script_path not in sys.path:
    sys.path.append(current_script_path)


try:
    from Brain_MRI import inferenceSdk as InferenceSdk
except ImportError as e:
    print(f"Error importing Brain_MRI: {e}")
    print(f"Please ensure the 'Brain_MRI' folder is located in: {current_script_path}")
    sys.exit(1)

def generate_description_with_llm(diagnosis_list):
    
    if not diagnosis_list or diagnosis_list == ["normal"]:
        return "Based on the MRI analysis showing normal findings, this patient does not exhibit signs of cognitive impairment disorders. No significant pathological features associated with dementia or other cognitive disorders are detected."
    
    cognitive_indicators = ["focal ischemia", "atrophy", "white matter", "hippocampal", "cortical", "subcortical", "vascular"]
    pathological_indicators = ["meningioma", "hemangioma", "tumor", "lesion", "infarct"]
    
    has_cognitive_indicators = any(indicator in ' '.join(diagnosis_list).lower() for indicator in cognitive_indicators)
    has_pathological = any(indicator in ' '.join(diagnosis_list).lower() for indicator in pathological_indicators)
    
    if has_cognitive_indicators:
        return f"The MRI findings including {', '.join(diagnosis_list)} suggest potential cognitive impairment with key diagnostic indicators being {diagnosis_list[0]} and related features. The presence of these findings warrants further clinical correlation for cognitive disorder subtype determination, particularly focusing on {diagnosis_list[0]} which may indicate specific patterns of neurodegeneration."
    elif has_pathological:
        return f"The identified findings ({', '.join(diagnosis_list)}) indicate structural abnormalities including {diagnosis_list[0]}. While these may not directly indicate cognitive impairment disorders, their location and impact on surrounding brain tissue should be evaluated for potential cognitive effects, particularly given the prominence of {diagnosis_list[0]} in the differential diagnosis."
    else:
        return f"The MRI analysis revealed {', '.join(diagnosis_list)} with {diagnosis_list[0]} being the primary finding. Clinical correlation is needed to determine if these findings represent normal variants or early pathological changes that could impact cognitive function."


def create_input_json(patient_id, t1_path, t2_path, t1_exists, t2_exists):
    input_data = [[]]
    
    if t1_exists and t1_path and str(t1_path).strip() and str(t1_path) != 'nan':
        input_data[0].append({
            "data": str(t1_path),
            "aux": "t1"
        })
    
    if t2_exists and t2_path and str(t2_path).strip() and str(t2_path) != 'nan':
        input_data[0].append({
            "data": str(t2_path), 
            "aux": "t2"
        })
    
    return input_data


def process_single_patient(patient_row, model, input_dir, output_dir):
    
    patient_id = patient_row['IDage']
    t1_path = patient_row.get('t1_path', '')
    t2_path = patient_row.get('t2_path', '')
    t1_exists = patient_row.get('t1_path_exists', False)
    t2_exists = patient_row.get('t2_path_exists', False)
    
    input_file = os.path.join(input_dir, f"{patient_id}_input.json")
    output_file = os.path.join(output_dir, f"{patient_id}_output.json")
    
    print(f"Processing patient: {patient_id}")
    print(f"  T1 exists: {t1_exists}, T2 exists: {t2_exists}")
    
    if not (t1_exists or t2_exists):
        print(f"  Warning: Patient {patient_id} missing both T1 and T2 images, creating default output")
        
        default_output = [{
            "diagnosis": [],
            "description": "Unable to perform MRI analysis due to missing image data. Both T1 and T2 weighted images are required for comprehensive brain MRI diagnosis."
        }]
        
        with open(output_file, 'w') as f:
            json.dump(default_output, f, indent=4)
        
        return {"patient_id": patient_id, "status": "skipped", "reason": "missing_images"}
    
    try:
        input_data = create_input_json(patient_id, t1_path, t2_path, t1_exists, t2_exists)
        
        if not input_data[0]:
            print(f"  Warning: No valid image paths for patient {patient_id}")
            default_output = [{
                "diagnosis": [],
                "description": "No valid MRI image paths available for analysis."
            }]
            
            with open(output_file, 'w') as f:
                json.dump(default_output, f, indent=4)
            
            return {"patient_id": patient_id, "status": "skipped", "reason": "invalid_paths"}
        
        with open(input_file, 'w') as f:
            json.dump(input_data, f, indent=4)
        
        print(f"  Created input file: {input_file}")
        print(f"  Input data: {len(input_data[0])} images")
        
        print(f"  Running model inference...")
        results = model.diagRG(input_data)
        
        diagnosis_list = []
        if results and isinstance(results, list) and len(results) > 0:
            if 'diagnosis' in results[0]:
                diagnosis_list = results[0]['diagnosis']
        
        print(f"  Diagnosis: {diagnosis_list}")
        
        print(f"  Generating description...")
        description = generate_description_with_llm(diagnosis_list)
        
        enhanced_output = [{
            "diagnosis": diagnosis_list,
            "description": description
        }]
        
        with open(output_file, 'w') as f:
            json.dump(enhanced_output, f, indent=4)
        
        print(f"  Saved output file: {output_file}")
        
        return {"patient_id": patient_id, "status": "success", "diagnosis_count": len(diagnosis_list)}
        
    except Exception as e:
        print(f"  Error processing patient {patient_id}: {str(e)}")
        
        error_output = [{
            "diagnosis": [],
            "description": f"Error occurred during MRI analysis: {str(e)}. Unable to complete diagnosis."
        }]
        
        with open(output_file, 'w') as f:
            json.dump(error_output, f, indent=4)
        
        return {"patient_id": patient_id, "status": "error", "error": str(e)}


def main():
    
    execution_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Working Directory: {execution_dir}")
    print(f"Script Directory:  {script_dir}")

    csv_file = os.path.join(execution_dir, "UniBrain-master/selected_samples_cleaned.csv")
    input_dir = os.path.join(execution_dir, "UniBrain-master/input")
    output_dir = os.path.join(execution_dir, "UniBrain-master/output")
    
    config_file = os.path.join(script_dir, "Brain_MRI", "configs", "config_UniBrain.yaml")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("MRI Diagnosis Processing Pipeline - Test Version")
    print("="*60)
    
    print(f"Loading patient data from: {csv_file}")
    try:
        if not os.path.exists(csv_file):
            print(f"Error: CSV file not found at {csv_file}")
            print(f"Please make sure you are running this script from the project root directory.")
            return

        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} patients")
        
        df = df.head(3)
        print(f"Testing with first {len(df)} patients")
        
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    print("\nInitializing UniBrain model...")
    print(f"Using config file: {config_file}")
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        return

    try:
        start_time = time.time()
        model = InferenceSdk.RatiocinationSdk(gpu_id=[0], inference_cfg=config_file)
        init_time = time.time() - start_time
        print(f"Model initialized successfully in {init_time:.2f} seconds")
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nProcessing {len(df)} test patients...")
    print("-"*60)
    
    results_summary = {
        "success": 0,
        "skipped": 0, 
        "error": 0,
        "total": len(df)
    }
    
    for idx, row in df.iterrows():
        result = process_single_patient(row, model, input_dir, output_dir)
        results_summary[result["status"]] += 1
        print()
    
    print("="*60)
    print("Processing Summary:")
    print("="*60)
    print(f"Total patients: {results_summary['total']}")
    print(f"Successfully processed: {results_summary['success']}")
    print(f"Skipped (missing images): {results_summary['skipped']}")
    print(f"Errors: {results_summary['error']}")
    if results_summary['total'] > 0:
        print(f"Success rate: {results_summary['success']/results_summary['total']*100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
