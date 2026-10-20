import os
import sys
import time
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def watch_audit(kernel_slug='cyruspinto/rsna-knee-arm-audit-120-studies', poll_interval=20, timeout_minutes=45):
    print(f"[*] Monitoring audit kernel: {kernel_slug}...")
    start_time = time.time()
    out_dir = 'scratch/audit_results'
    os.makedirs(out_dir, exist_ok=True)
    
    while time.time() - start_time < timeout_minutes * 60:
        try:
            status_res = kaggle.api.kernels_status(kernel_slug)
            curr_status = getattr(status_res, 'status', None)
            elapsed = int(time.time() - start_time)
            print(f"[{elapsed}s] Kernel Status: {curr_status}")
            
            if 'COMPLETE' in str(curr_status):
                print("[+] Audit Kernel COMPLETE! Downloading outputs...")
                try:
                    kaggle.api.kernels_output(kernel_slug, out_dir)
                    print("[+] Outputs downloaded to:", out_dir)
                    print("Files:", os.listdir(out_dir))
                except Exception as e:
                    print("Error downloading outputs:", e)
                return True
            elif 'ERROR' in str(curr_status):
                failure_msg = getattr(status_res, 'failure_message', 'No details')
                print(f"[!] Kernel failed with ERROR: {failure_msg}")
                try:
                    kaggle.api.kernels_output(kernel_slug, out_dir)
                except Exception:
                    pass
                return False
        except Exception as e:
            print("Poll error:", type(e), e)
            
        time.sleep(poll_interval)
        
    print("[!] Timeout waiting for audit kernel.")
    return False

if __name__ == '__main__':
    watch_audit()
