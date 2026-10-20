import os
import sys
import time
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def monitor_kernel(kernel_slug='cyruspinto/rsna-knee-v23-r7-multiscale-decoupled-blend', poll_interval=15, timeout_minutes=30):
    """Notify-only kernel monitor. Does NOT submit. Leaves all submission decisions to the human."""
    print(f"[*] Starting Notify-Only Monitor for {kernel_slug}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout_minutes * 60:
        try:
            status_res = kaggle.api.kernels_status(kernel_slug)
            curr_status = getattr(status_res, 'status', None)
            elapsed = int(time.time() - start_time)
            print(f"[{elapsed}s] Kernel Status: {curr_status}")
            
            if 'COMPLETE' in str(curr_status):
                print(f"[+] Kernel {kernel_slug} is COMPLETE. Notification only - NO AUTO SUBMISSION.")
                return True
            elif 'ERROR' in str(curr_status):
                failure_msg = getattr(status_res, 'failure_message', 'No details')
                print(f"[!] Kernel {kernel_slug} failed with ERROR: {failure_msg}")
                return False
        except Exception as e:
            print("Poll error:", type(e), e)
            
        time.sleep(poll_interval)
        
    print("[!] Timeout waiting for kernel monitor.")
    return False

if __name__ == '__main__':
    slug = sys.argv[1] if len(sys.argv) > 1 else 'cyruspinto/rsna-knee-v23-r7-multiscale-decoupled-blend'
    monitor_kernel(slug)
