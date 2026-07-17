import json
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('消融.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code = ''.join(nb['cells'][1]['source'])
exec(code)