from Brain_MRI import inferenceSdk as InferenceSdk
import json

import time

model = InferenceSdk.RatiocinationSdk(gpu_id=[0], inference_cfg='Brain_MRI/configs/config_UniBrain.yaml')


input_case_dict = json.load(open('./input/input.json','r'))

results = model.diagRG(input_case_dict)

with open('./output/output.json', 'w') as f:
    json.dump(results, f, indent=4)