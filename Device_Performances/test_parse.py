import json, ast

with open('/home/jazz/Projects/Wearable-Biomechanics/Device_Performances/device_performance.ipynb') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if source.startswith('!'):
            continue
        try:
            ast.parse(source)
        except SyntaxError as e:
            print(f"Cell index {i} error: {e.__class__.__name__}: {e}")
            for j, line in enumerate(source.split('\n')):
                print(f"Line {j+1}: {repr(line)}")
