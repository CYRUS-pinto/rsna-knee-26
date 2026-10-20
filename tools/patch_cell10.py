import json
from pathlib import Path

for nb_path in [
    Path('submissions/v32_ablation_stride2_norad/rsna-knee-v32-ablation-stride2-norad.ipynb'),
    Path('submissions/v32_tri_engine_medical_master/rsna-knee-v32-tri-engine-medical-master.ipynb')
]:
    nb = json.load(open(nb_path, 'r', encoding='utf-8'))
    cell = nb['cells'][10]
    src = ''.join(cell['source'])
    
    header = (
        "import os, sys, time, json\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from pathlib import Path\n\n"
        "if 'WORK' not in locals():\n"
        "    WORK = Path('/kaggle/working')\n"
        "if 'DATA' not in locals():\n"
        "    DATA = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')\n"
        "    if not DATA.exists():\n"
        "        DATA = Path('/kaggle/input/rsna-knee-abnormality-detection')\n\n"
    )
    
    if not src.startswith("import os, sys"):
        src = header + src
        
    src = src.replace("time.time() - NB_T0", "time.time() - (T0 if 'T0' in locals() else time.time())")
    src = src.replace("display(sub.head())", "print(sub.head())")
    
    lines = src.splitlines(keepends=True)
    cell['source'] = lines
    json.dump(nb, open(nb_path, 'w', encoding='utf-8'), indent=1)
    print(f"Successfully patched {nb_path}")
