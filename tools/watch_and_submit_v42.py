import os
import sys
import time
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def watch_and_submit_v42():
    try:
        kaggle.api.authenticate()
        print("[+] Authenticated with Kaggle API.")
        kernel_slug = 'cyruspinto/rsna-knee-v42-sovereign-live-master'
        
        status = kaggle.api.kernels_status(kernel_slug)
        curr_status = getattr(status, 'status', None)
        print(f"[+] Current kernel status for {kernel_slug}: {curr_status}")
        
        if 'COMPLETE' not in str(curr_status):
            print(f"[-] Kernel is currently: {curr_status}. Notebook commit is in progress on Kaggle GPU.")
            return False, curr_status
            
        print("[*] Submitting Candidate V42 Sovereign Live Master to Kaggle competition...")
        res = kaggle.api.competition_submit_code(
            file_name='submission.csv',
            message='Candidate V42 Sovereign Live Master (Engine 11 4-Fold 0.8712 + Engine 9 + Engine 10 + Super-CoAtNet + R7 + Raptor)',
            competition='rsna-knee-abnormality-detection',
            kernel=kernel_slug,
            kernel_version=1
        )
        print("[+] Submission Successful! Response:", res)
        return True, res
    except Exception as e:
        print("[-] Submission Attempt Error:", type(e), e)
        if hasattr(e, 'response') and e.response is not None:
            print("Response Body:", e.response.text)
        return False, str(e)

if __name__ == '__main__':
    watch_and_submit_v42()
