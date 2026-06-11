import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from deepseek_api import DeepSeekClient
import re
from typing import Optional

def extract_pure_csv(mixed_text: str) -> Optional[str]:

    clean_text = re.sub(r'\s+', ' ', mixed_text.strip())

    patterns = [
        r'([A-Za-z0-9]+\s*,\s*\d+(?:\.\d+)?\s*(?:,\s*[01]){15})',
        r'"?([A-Za-z0-9]+)"?\s*,\s*"?(\d+(?:\.\d+)?)"?\s*((?:,\s*"?[01]"?){15})',
        r'([A-Za-z0-9]+)\s*,\s*(\d+(?:\.\d+)?)\s*((?:,\s*[01]){15})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, clean_text)
        if match:
            if len(match.groups()) == 1:
                csv_line = match.group(1)
            else:
                csv_line = f"{match.group(1)},{match.group(2)}{match.group(3)}"
            
            csv_line = re.sub(r'\s*,\s*', ',', csv_line.strip())
            csv_line = csv_line.replace('"', '')
            
            values = csv_line.split(',')
            if len(values) == 17:  
                if re.match(r'^[A-Za-z0-9]+$', values[0]):
                    try:
                        float(values[1])
                    except ValueError:
                        continue

                    if all(val in ['0', '1'] for val in values[2:]):
                        return csv_line

    lines = clean_text.split('\n')
    for line in lines:
        if ',' in line and re.search(r'[01]', line):
            csv_candidate = re.sub(r'[^\w,.\s]', '', line).strip()
            if csv_candidate:
                values = [v.strip() for v in csv_candidate.split(',') if v.strip()]
                if len(values) == 17:
                    try:
                        if (re.match(r'^[A-Za-z0-9]+$', values[0]) and
                            isinstance(float(values[1]), float) and
                            all(v in ['0', '1'] for v in values[2:])):
                            return ','.join(values)
                    except (ValueError, IndexError):
                        continue
    
    return None

try:
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
except ImportError:
    print("警告: 无法导入配置文件config.py，请确保文件存在且包含API配置")
    DEEPSEEK_API_KEY = None
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

if DEEPSEEK_API_KEY:
    client = DeepSeekClient(api_key=DEEPSEEK_API_KEY)
else:
    print("错误: 无法获取API KEY，请检查config.py文件")
    client = None

CLASS_COLUMNS = [
    "ID", "NACCAGE", 'NC', 'MCI', 'DE', 'CI', 'MCI_A', 'MCI_AM', 'MCI_Na', 'MCI_NaM', 'AD', 'LBD', 'VD', 'FTD', 'EXC', 'PSY', 'ODE'
]

def get_diagnosis_prompt(three_class_mode=False):
    if three_class_mode:
        diagnosis_hierarchy = """
### 三分类模式诊断标准
```
graph TD
    A[认知障碍诊断] --> B[正常对照 NC]
    A --> C[轻度认知障碍 MCI] 
    A --> D[痴呆 DE]
```
**重要**: 在三分类模式下，只能输出NC、MCI、DE三个类别中的一个。
其他所有亚型分类（CI、MCI_A、MCI_AM等）都设为0。
        """
        output_instruction = """
### 任务要求：
1. 仅使用三分类：NC、MCI、DE
2. 输出必须包含以下16个值（ID+年龄+15个0/1标签）：
   ID,NACCAGE,NC,MCI,DE,CI,MCI_A,MCI_AM,MCI_Na,MCI_NaM,AD,LBD,VD,FTD,EXC,PSY,ODE
3. 在三分类模式下，只有NC、MCI、DE中的一个为1，其余全为0
4. 必须为纯CSV格式的一行数据
5. 不要包含任何额外解释
6. NACCAGE保留一位小数
        """
    else:
        diagnosis_hierarchy = """
### 完整诊断分类标准
```
graph TD
    A[认知障碍诊断] --> B[正常对照 NC]
    A --> E[轻度认知障碍 MCI]
    A --> D[痴呆 DE]
    E --> F[其他轻度认知障碍 CI]
    E --> G[MCI_A<br>遗忘型轻度认知障碍]
    E --> H[MCI_AM<br>遗忘型多领域MCI]
    E --> I[MCI_Na<br>非遗忘型MCI]
    E --> J[MCI_NaM<br>非遗忘型多领域MCI]
    D --> K[痴呆具体类型]
    K --> L[阿尔茨海默病 AD]
    K --> M[路易体痴呆 LBD] 
    K --> N[血管性痴呆 VD]
    K --> O[额颞叶痴呆 FTD]
    K --> P[其他痴呆 ODE]
    D --> Q[排除标准 EXC]
    D --> R[精神病诊断 PSY]
```
        """
        output_instruction = """
### 任务要求：
1. 严格遵循分类树层级结构
2. 输出必须包含以下16个值（ID+年龄+15个0/1标签）：
   ID,NACCAGE,NC,MCI,DE,CI,MCI_A,MCI_AM,MCI_Na,MCI_NaM,AD,LBD,VD,FTD,EXC,PSY,ODE
3. 必须为纯CSV格式的一行数据
4. 不要包含任何额外解释
5. NACCAGE保留一位小数
        """
    
    return f"""
请根据以下诊断分类标准，判断报告中患者的诊断分类：

{diagnosis_hierarchy}

{output_instruction}

### 报告内容：
{{report_text}}
"""

def extract_id_from_filename(filename):
    name_without_ext = filename.replace('.txt', '').replace('.json', '')

    parts = name_without_ext.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[:-1])
    
    return None

def process_single_report(filepath, three_class_mode=False):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            report_text = f.read()
        filename = os.path.basename(filepath)
 
        name_without_ext = filename.replace('.txt', '').replace('.json', '')
        parts = name_without_ext.split('_')
        
        if len(parts) >= 3:
            try:
                patient_age = float(parts[-2])
                patient_id = '_'.join(parts[:-2])
            except ValueError:
                patient_id = '_'.join(parts[:-1])
                patient_age = 0.0
        elif len(parts) >= 2:
            try:
                patient_age = float(parts[-1])
                patient_id = '_'.join(parts[:-1])
            except ValueError:
                patient_id = '_'.join(parts[:-1])
                patient_age = 0.0
        else:
            patient_id = extract_id_from_filename(filename)
            if not patient_id:
                return None
            patient_age = 0.0  
        
        prompt = get_diagnosis_prompt(three_class_mode).format(report_text=report_text)
        
        if client is None:
            print(f"错误: DeepSeek客户端未初始化，跳过文件 {filepath}")
            return None
        
        response = client.chat(
            prompt,
            model="deepseek-chat",
            temperature=0,
            max_tokens=200
        )
        
        csv_line = extract_pure_csv(response)
        if csv_line:
            parts = csv_line.split(',')
            if len(parts) >= 17:
                parts[0] = patient_id
                try:
                    float(parts[1])
                except (ValueError, IndexError):
                    parts[1] = f"{patient_age:.1f}"
                
                if three_class_mode:
                    for i in range(2, len(parts)):
                        parts[i] = '0'
                    
                    original_csv = csv_line.split(',')
                    if len(original_csv) >= 17:
                        if original_csv[2] == '1':
                            parts[2] = '1'
                        elif original_csv[3] == '1':
                            parts[3] = '1' 
                        elif original_csv[4] == '1':
                            parts[4] = '1'
                        else:
                            if any(original_csv[i] == '1' for i in [5, 6, 7, 8, 9]):
                                parts[3] = '1'
                            elif any(original_csv[i] == '1' for i in [10, 11, 12, 13, 16]):
                                parts[4] = '1'
                            else:
                                parts[2] = '1'
                
                return ','.join(parts)
        
        print(f"无效响应格式: {response[:200]}...")
        return None
    
    except Exception as e:
        print(f"处理文件 {filepath} 时出错: {str(e)}")
        return None

def generate_classification_csv(reports_dir, output_csv="diagnosis_classification.csv", three_class_mode=False):
    report_files = [
        os.path.join(reports_dir, f) 
        for f in os.listdir(reports_dir) 
        if f.endswith("_report.txt")
    ]
    
    print(f"找到 {len(report_files)} 个报告文件")
    print(f"三分类模式: {'启用' if three_class_mode else '禁用'}")
    
    num_cores = max(1, cpu_count() - 1)
    print(f"使用 {num_cores} 个进程并行处理...")
    
    results = []
    with Pool(num_cores) as pool:
        from functools import partial
        process_func = partial(process_single_report, three_class_mode=three_class_mode)
        
        for i, result in enumerate(pool.imap(process_func, report_files), 1):
            if result:
                results.append(result)
            print(f"\r处理进度: {i}/{len(report_files)} | 成功: {len(results)}", end="")
    
    if results:
        with open(output_csv, 'w', encoding='utf-8') as f:
            f.write(",".join(CLASS_COLUMNS) + "\n")
            for line in results:
                f.write(line + "\n")
        print(f"\n结果已保存到 {output_csv}")
        
        if three_class_mode:
            three_class_csv = output_csv.replace('.csv', '_three_class.csv')
            with open(three_class_csv, 'w', encoding='utf-8') as f:
                f.write(",".join(CLASS_COLUMNS) + "\n")
                for line in results:
                    f.write(line + "\n")
            print(f"三分类版本已保存到 {three_class_csv}")
    else:
        print("\n未生成有效结果")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成诊断分类CSV")
    parser.add_argument("--reports-dir", default="reports", help="报告目录")
    parser.add_argument("--output-csv", default="diagnosis_classification.csv", help="输出CSV文件")
    parser.add_argument("--three-class-mode", action="store_true", help="启用三分类模式（仅NC、MCI、DE）")
    
    args = parser.parse_args()
    generate_classification_csv(args.reports_dir, args.output_csv, args.three_class_mode)