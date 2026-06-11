import os
import json
import pandas as pd
import sys
from pathlib import Path
from Brain_MRI import inferenceSdk as InferenceSdk
import time
from tqdm import tqdm


def create_input_json(patient_id, t1_path, t2_path, t1_exists, t2_exists):
    input_data = [[]]
    
    def get_first_valid_path(path_str):
        if not path_str or str(path_str).strip() == '' or str(path_str) == 'nan':
            return None
        
        paths = str(path_str).split(',')
        for path in paths:
            path = path.strip()
            if path and path != 'nan' and os.path.exists(path):
                return path
        
        first_path = paths[0].strip() if paths else None
        return first_path if first_path and first_path != 'nan' else None
    
    if t1_exists and t1_path:
        valid_t1_path = get_first_valid_path(t1_path)
        if valid_t1_path:
            input_data[0].append({
                "data": valid_t1_path,
                "aux": "t1"
            })
    
    if t2_exists and t2_path:
        valid_t2_path = get_first_valid_path(t2_path)
        if valid_t2_path:
            input_data[0].append({
                "data": valid_t2_path,
                "aux": "t2"
            })
    
    return input_data


def process_single_patient_unibrain(patient_row_dict, input_dir, config_file, model):

    patient_id = patient_row_dict.get('ID', 'Unknown_Patient')
    
    t1_path = patient_row_dict.get('t1_path', '')
    t2_path = patient_row_dict.get('t2_path', '')
    t1_exists = patient_row_dict.get('t1_path_exists', False)
    t2_exists = patient_row_dict.get('t2_path_exists', False)

    if 't1_path_exists' not in patient_row_dict and t1_path:
        t1_exists = check_path_exists_single(t1_path)
    if 't2_path_exists' not in patient_row_dict and t2_path:
        t2_exists = check_path_exists_single(t2_path)

    if not (t1_exists or t2_exists):
        print(f"\u26a0 Patient {patient_id}: Missing both T1 and T2 images")
        return patient_id, [], False
    
    try:
        input_data = create_input_json(patient_id, t1_path, t2_path, t1_exists, t2_exists)
        
        if not input_data[0]: 
            print(f"\u26a0 Patient {patient_id}: No valid image paths")
            return patient_id, [], False
        
        results = model.diagRG(input_data)
        
        diagnosis_list = []
        if results and isinstance(results, list) and len(results) > 0:
            if 'diagnosis' in results[0]:
                diagnosis_list = results[0]['diagnosis']
        
        return patient_id, diagnosis_list, True
        
    except Exception as e:
        print(f"\u274c Patient {patient_id}: UniBrain Error - {str(e)}")
        return patient_id, [], False


def check_path_exists_single(path_str):
    if pd.isna(path_str) or str(path_str).strip() == '' or str(path_str) == 'nan':
        return False
    
    paths = str(path_str).split(',')
    for path in paths:
        path = path.strip()
        if path and path != 'nan' and os.path.exists(path):
            return True
    return False


def main(csv_file=None, input_dir=None, output_dir=None, max_workers=None):
    base_dir = "./adRAG/UniBrain-master"
    
    if csv_file is None:
        raise FileNotFoundError("CSV file path is required but not provided")
        
    config_file = os.path.join(base_dir, "Brain_MRI/configs/config_UniBrain.yaml")
    
    print("="*60)
    print("MRI Diagnosis Processing Pipeline (Modified Single-Step Approach)")
    print("="*60)
    print(f"Step 1: UniBrain model inference (single process)")
    print("="*60)
    
    print(f"Loading patient data from: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} patients")
        
        required_cols = ['ID', 't1_path', 't2_path']
        for col in required_cols:
            if col not in df.columns:
                if col == 'ID':
                    df[col] = [f'Patient_{i}' for i in range(len(df))]
                elif col in ['t1_path', 't2_path']:
                    df[col] = ''
                print(f"Warning: Column '{col}' not found, created with default values")
        
        for i, val in df['ID'].items():
            if pd.isna(val):
                df.at[i, 'ID'] = f'Patient_{i}'
        df['t1_path'] = df['t1_path'].fillna('')
        df['t2_path'] = df['t2_path'].fillna('')
        
        def check_path_exists(path_str):
            if pd.isna(path_str) or str(path_str).strip() == '' or str(path_str) == 'nan':
                return False
            
            paths = str(path_str).split(',')
            for path in paths:
                path = path.strip()
                if path and path != 'nan' and os.path.exists(path):
                    return True
            return False
        
        df['t1_path_exists'] = df['t1_path'].apply(check_path_exists)
        df['t2_path_exists'] = df['t2_path'].apply(check_path_exists)
        
        print(f"Generated path existence flags:")
        print(f"T1 paths available: {df['t1_path_exists'].sum()}/{len(df)} ({df['t1_path_exists'].sum()/len(df)*100:.1f}%)")
        print(f"T2 paths available: {df['t2_path_exists'].sum()}/{len(df)} ({df['t2_path_exists'].sum()/len(df)*100:.1f}%)")
        print(f"Patients with at least one image: {(df['t1_path_exists'] | df['t2_path_exists']).sum()}/{len(df)}")
        
        print(f"\nSample T1 paths:")
        for i, row in df.head(3).iterrows():
            print(f"  {row['ID']}: {row['t1_path']} (exists: {row['t1_path_exists']})")
        print(f"Sample T2 paths:")
        for i, row in df.head(3).iterrows():
            print(f"  {row['ID']}: {row['t2_path']} (exists: {row['t2_path_exists']})")
        
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return {}
    
    print(f"\n{'='*60}")
    print("STEP 1: UniBrain Model Processing (Single Process)")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        print("Initializing UniBrain model...")
        model = InferenceSdk.RatiocinationSdk(gpu_id=[0], inference_cfg=config_file)
        init_time = time.time() - start_time
        print(f"\u2705 Model initialized successfully in {init_time:.2f} seconds")
    except Exception as e:
        print(f"\u274c Error initializing model: {e}")
        return {}
    
    diagnosis_results = {}
    
    print(f"\nProcessing {len(df)} patients with UniBrain model...")
    print("-"*60)
    
    step1_summary = {
        "success": 0,
        "skipped": 0,
        "error": 0,
        "total": len(df)
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="UniBrain Processing"):
        patient_id, diagnosis_list, success = process_single_patient_unibrain(
            row.to_dict(), None, config_file, model
        )
        
        if success:
            diagnosis_results[patient_id] = diagnosis_list
            step1_summary["success"] += 1
        elif not diagnosis_list:
            diagnosis_results[patient_id] = []
            step1_summary["skipped"] += 1
        else:
            diagnosis_results[patient_id] = []
            step1_summary["error"] += 1
    
    print(f"\n{'='*60}")
    print("STEP 1 Summary:")
    print(f"{'='*60}")
    print(f"Total patients: {step1_summary['total']}")
    print(f"Successfully processed: {step1_summary['success']}")
    print(f"Skipped (missing images): {step1_summary['skipped']}")
    print(f"Errors: {step1_summary['error']}")
    print(f"Success rate: {step1_summary['success']/step1_summary['total']*100:.1f}%")
    
    total_success = step1_summary['success']
    total_patients = len(df)
    
    print(f"\n{'='*60}")
    print("OVERALL Processing Summary:")
    print(f"{'='*60}")
    print(f"Total patients: {total_patients}")
    print(f"Successfully processed: {total_success}")
    print(f"Step 1 failures: {step1_summary['error'] + step1_summary['skipped']}")
    print(f"Overall success rate: {total_success/total_patients*100:.1f}%")
    print("="*60)
    
    print("Processing completed!")
    
    return diagnosis_results


if __name__ == "__main__":
    main()