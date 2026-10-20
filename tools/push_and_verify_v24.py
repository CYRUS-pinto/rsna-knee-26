import os
import sys
import time
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def push_and_verify_v24():
    try:
        kaggle.api.authenticate()
        print("Authenticated with Kaggle API.")
        
        kernel_dir = 'submissions/v24_r7_uniform_multiscale_master'
        kernel_slug = 'cyruspinto/rsna-knee-v24-r7-uniform-multiscale-master'
        
        print(f"[*] Pushing Candidate V24 to Kaggle: {kernel_dir}...")
        kaggle.api.kernels_push(kernel_dir)
        print(f"[+] Pushed successfully. Monitoring kernel execution...")
        
        t0 = time.time()
        while time.time() - t0 < 3600:
            status = kaggle.api.kernels_status(kernel_slug)
            curr_status = getattr(status, 'status', None)
            elapsed = int(time.time() - t0)
            print(f"[{elapsed}s] V24 Status: {curr_status}")
            
            if 'COMPLETE' in str(curr_status):
                print("[+] Candidate V24 kernel run COMPLETE! Verified ready for submission.")
                return True
            elif 'ERROR' in str(curr_status):
                fail_msg = getattr(status, 'failure_message', 'No details')
                print(f"[!] Kernel failed with error: {fail_msg}")
                return False
                
            time.sleep(25)
            
        print("[!] Timeout waiting for V24 kernel run.")
        return False
        
    except Exception as e:
        print("Error during V24 push/verify:", type(e), e)
        return False

if __name__ == '__main__':
    push_and_verify_v24()
