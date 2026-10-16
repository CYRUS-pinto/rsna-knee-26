import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def push_kernel(folder_path, max_retries=3):
    folder = str(Path(folder_path).resolve())
    print(f"Preparing to push kernel from: {folder}")
    for attempt in range(max_retries):
        try:
            kaggle.api.authenticate()
            print("Authenticated successfully with Kaggle API.")
            res = kaggle.api.kernels_push(folder)
            print("Successfully pushed kernel!")
            print("Response:", res)
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {type(e)} {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print("Failed to push kernel after retries.")
                return False

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'submissions/v18_grandmaster_calibrated'
    push_kernel(target)
