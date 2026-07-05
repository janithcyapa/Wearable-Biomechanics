import json, re

with open('/home/jazz/Projects/Wearable-Biomechanics/Device_Performances/device_performance.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        if any('def get_sensor_data' in l for l in source):
            # Fix 8-space indent to 4-space indent
            new_source = []
            for line in source:
                if line.startswith('        '):
                    new_source.append(line.replace('        ', '    ', 1))
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open('/home/jazz/Projects/Wearable-Biomechanics/Device_Performances/device_performance.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

