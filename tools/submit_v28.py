import os
import sys
import time
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def submit_v28():
    try:
        kaggle.api.authenticate()
        print("[+] Authenticated with Kaggle API.")
        kernel_slug = 'cyruspinto/rsna-knee-v28-calibrated-grandmaster-master'
        
        status = kaggle.api.kernels_status(kernel_slug)
        curr_status = getattr(status, 'status', None)
        print(f"[+] Current kernel status for {kernel_slug}: {curr_status}")
        
        if 'COMPLETE' not in str(curr_status):
            print(f"[-] Kernel is not yet COMPLETE (status: {curr_status}). Wait for commit to finish.")
            return False, curr_status
            
        print("[*] Submitting Candidate V28 Calibrated Grandmaster to Kaggle competition...")
        res = kaggle.api.competition_submit_code(
            file_name='submission.csv',
            message='Candidate V28 (384px R7 MultiScale + Raptor Dual-Crop + Calibrated Specialist Weights + Flip TTA)',
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
    submit_v28()
