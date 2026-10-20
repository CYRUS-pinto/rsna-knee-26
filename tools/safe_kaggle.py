import os
import time
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def call_with_retry(fn, max_attempts=5, initial_delay=2):
    delay = initial_delay
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            kaggle.api.authenticate()
            return fn()
        except Exception as e:
            last_err = e
            print(f"API attempt {attempt}/{max_attempts} failed ({type(e).__name__}: {e}). Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 1.5
    raise last_err

def check_kernel(kernel_slug):
    return call_with_retry(lambda: kaggle.api.kernels_status(kernel_slug))

def check_submissions(comp='rsna-knee-abnormality-detection', top_n=5):
    return call_with_retry(lambda: kaggle.api.competition_submissions(comp)[:top_n])

def pull_kernel(kernel_slug, path='scratch'):
    import os
    os.makedirs(path, exist_ok=True)
    return call_with_retry(lambda: kaggle.api.kernels_pull(kernel_slug, path=path))

if __name__ == '__main__':
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if action == 'v24':
        st = check_kernel('cyruspinto/rsna-knee-v24-r7-uniform-multiscale-master')
        print("V24 Kernel Status:", st)
    elif action == 'detailed_subs':
        subs = check_submissions(top_n=10)
        for s in subs:
            attrs = {k: v for k, v in vars(s).items() if not k.startswith('_')}
            print(f"Ref: {s.ref} | Date: {getattr(s, 'date', None)} | Status: {s.status} | Score: {s.public_score} | Desc: {getattr(s, 'description', '')}")
