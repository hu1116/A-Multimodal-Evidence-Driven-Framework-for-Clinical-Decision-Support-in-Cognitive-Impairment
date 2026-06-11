# 诊断报告生成与评估流水线 (Pipeline)

## 概述

这是一个完整的一站式诊断报告生成与评估流水线，包含以下6个步骤：

1. **特征文本化** - 将CSV格式的特征矩阵转换为自然语言描述
2. **文本优化** - 优化生成的文本描述
3. **MRI图像诊断** - 处理MRI图像并生成诊断结果
4. **报告生成** - 生成个性化的诊断报告
5. **报告转CSV** - 将报告结果转换为结构化CSV格式
6. **预测评估** - 评估预测结果并生成可视化图表

## 环境要求



## 目录结构

```
adRAG/
├── pipeline.py                    # 主流水线文件
├── main.py                       # 诊断报告生成模块
├── ReportToCsv.py               # 报告转CSV模块
├── data/
│   ├── selected_samples.csv     # 输入数据（默认）
│   └── all_samples_x.csv        # 真实标签数据
├── Feature2Txt/
│   ├── feature_to_text.py       # 特征文本化
│   ├── optimize_descriptions.py  # 文本优化
│   └── output/                  # 输出目录
├── UniBrain-master/
│   └── mri_diagnosis.py         # MRI诊断
├── reports/                     # 报告输出目录
└── figures/                     # 图表输出目录
```

## 使用方法

### 1. 基本使用
```bash
python pipeline.py
```

### 2. 指定输入文件
```bash
python pipeline.py --input data/my_samples.csv
```

### 3. 指定时间戳
```bash
python pipeline.py --input data/my_samples.csv --timestamp 20240120_1030
```

### 4. 检查环境
```bash
python pipeline_example.py
```

## 输入数据格式

输入CSV文件必须包含以下列：
- `ID`: 患者唯一标识符
- `NACCAGE`: 患者年龄
- 其他列: 患者的各种特征数据

## 输出说明

### 报告目录结构：
```
reports/{timestamp}/
├── {ID}_{Age}_report.txt         # 个人诊断报告
├── classification_results.csv    # CSV格式结果
├── all_classes_evaluation.csv    # 所有类别详细评估
├── visualization_statistics.txt   # 可视化统计信息
└── processing_errors.log         # 错误日志（如有）
```



## 评估指标

流水线会生成以下评估指标：

### 三大类评估 (NC, MCI, DE)：
- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- F1-score（F1分数）
- AUROC（ROC曲线下面积）
- AUPR（PR曲线下面积）
- MCC（马修斯相关系数）
- Specificity（特异性）

### 所有类别评估：
包含15个诊断类别：NC, MCI, DE, CI, MCI_A, MCI_AM, MCI_Na, MCI_NaM, AD, LBD, VD, FTD, EXC, PSY, ODE


## 故障排除

### 1. 导入错误
如果遇到模块导入错误，请确保：
- 所有必需的Python文件存在于正确位置
- Python路径设置正确
- 依赖包已正确安装

### 2. 数据格式错误
确保输入CSV文件：
- 包含必需的ID和NACCAGE列
- 数据类型正确
- 没有缺失的关键值

### 3. 权限错误
确保有以下目录的写入权限：
- `reports/`
- `figures/`
- `Feature2Txt/output/`
- `UniBrain-master/output_nc/`


## 技术支持

如有问题，请检查：
1. 错误日志：`reports/{timestamp}/processing_errors.log`
2. 终端输出信息
3. 文件权限和路径设置
