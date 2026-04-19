import os
import subprocess
from pathlib import Path
os.environ['KAGGLE_USERNAME'] = 'alej0909'
os.environ['KAGGLE_KEY'] = 'KGAT_55c43d8a96170bc499971e7337c36a50'
dest = Path('/workspace/moe_medical_vision/data/raw/osteoporosis')
dest.mkdir(parents=True, exist_ok=True)
print('Descargando Osteoartritis...')
subprocess.run(['kaggle', 'datasets', 'download', '-d', 'dhruvacube/osteoarthritis', '-p', str(dest), '--unzip'], capture_output=True)
print('Listo.')
