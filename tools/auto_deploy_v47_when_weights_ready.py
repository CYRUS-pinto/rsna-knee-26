"""
Automated Watcher & Deployer for Candidate V47 (Apex Tri-Arm Sovereign Ensemble):
1. Monitors Kaggle Dataset cyruspinto/rsna-knee-engine12-convnext-large-weights.
2. The instant the real checkpoint (>1MB) is uploaded from Colab Tab 2:
   - Pushes kernel submissions/v47_apex_tri_arm_ensemble to Kaggle.
   - Monitors interactive execution until COMPLETE.
   - Submits submission.csv to rsna-knee-abnormality-detection competition (Submission 3/5 today).
   - Automatically tracks scoring status and logs results.
"""
import os, sys, time, subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('.env')
import kaggle

DATASET_SLUG = 'cyruspinto/rsna-knee-engine12-convnext-large-weights'
KERNEL_PATH = Path('submissions/v47_apex_tri_arm_ensemble')
KERNEL_SLUG = 'cyruspinto/rsna-knee-v47-apex-tri-arm-ensemble'
COMPETITION = 'rsna-knee-abnormality-detection'

def main():
    kaggle.api.authenticate()
    print(f"[*] Starting Auto-Deploy Watcher for V47...", flush=True)
    print(f"[*] Monitoring Kaggle Dataset: {DATASET_SLUG}", flush=True)
    
    # 1. Poll dataset until real checkpoint (>1MB) appears
    weights_ready = False
    start_time = time.time()
    while time.time() - start_time < 3 * 3600: # 3 hour max wait
        try:
            files = kaggle.api.dataset_list_files(DATASET_SLUG).files
            for f in files:
                name = getattr(f, 'name', '')
                size = getattr(f, 'size', 0)
                # Filter out the 48-byte placeholder
                if 'engine12' in name.lower() or size > 1000000:
                    print(f"\\n[+] WEIGHTS DETECTED: {name} ({size / 2**20:.1f} MB)!", flush=True)
                    weights_ready = True
                    break
            if weights_ready:
                break
        except Exception as e:
            print(f"[!] Dataset poll error: {e}", flush=True)
            
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed}s] Waiting for Colab Tab 2 upload to {DATASET_SLUG}...", flush=True)
        time.sleep(30)
        
    if not weights_ready:
        print("[!] Timeout waiting for weights upload.", flush=True)
        return False
        
    # 2. Push V47 kernel to Kaggle
    print(f"[+] Pushing Candidate V47 to Kaggle: {KERNEL_SLUG}...", flush=True)
    cmd = ['kaggle', 'kernels', 'push', '-p', str(KERNEL_PATH)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout, res.stderr, flush=True)
    
    # 3. Monitor kernel commit
    print(f"[*] Monitoring commit execution for {KERNEL_SLUG}...", flush=True)
    kernel_start = time.time()
    while time.time() - kernel_start < 2 * 3600:
        try:
            status_res = kaggle.api.kernels_status(KERNEL_SLUG)
            status = getattr(status_res, 'status', None)
            k_elapsed = int(time.time() - kernel_start)
            print(f"[{k_elapsed}s] Kernel Status: {status}", flush=True)
            
            if 'COMPLETE' in str(status):
                print(f"[+] Kernel {KERNEL_SLUG} is COMPLETE! Submitting to competition...", flush=True)
                sub_res = kaggle.api.competition_submit_code(
                    file_name='submission.csv',
                    message='Candidate V47 Apex Tri-Arm Sovereign Ensemble (Engine 12 ConvNeXt-Large 384 + Raptor 384/224 + ConvNeXt R6)',
                    competition=COMPETITION,
                    kernel=KERNEL_SLUG,
                    kernel_version=1
                )
                print(f"[+] V47 SUBMISSION SUCCESSFUL! Response: {sub_res}", flush=True)
                
                # Update live state
                subprocess.run(['python', 'tools/state_snapshot.py'], check=False)
                return True
            elif 'ERROR' in str(status):
                fail = getattr(status_res, 'failure_message', 'No details')
                print(f"[!] Kernel failed with ERROR: {fail}", flush=True)
                return False
        except Exception as e:
            print(f"Kernel poll error: {e}", flush=True)
            
        time.sleep(30)
        
    return False

if __name__ == '__main__':
    main()
