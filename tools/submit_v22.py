import os
import sys
import time
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def submit_v22():
    try:
        kaggle.api.authenticate()
        print("Authenticated with Kaggle API.")
        kernel_slug = 'cyruspinto/rsna-knee-v22-quint-engine-blend'
        
        status = kaggle.api.kernels_status(kernel_slug)
        curr_status = getattr(status, 'status', None)
        print(f"Current kernel status for {kernel_slug}: {curr_status}")
        
        if 'COMPLETE' not in str(curr_status):
            print(f"Kernel is not yet COMPLETE (status: {curr_status}). Waiting before submission...")
            return False, curr_status
            
        res = kaggle.api.competition_submit_code(
            file_name='submission.csv',
            message='Candidate V22 Decoupled Compartmental Blend (ADR-022 Resolving Spearman Inversion)',
            competition='rsna-knee-abnormality-detection',
            kernel=kernel_slug,
            kernel_version=2
        )
        print("Submission Response:", res)
        return True, res
    except Exception as e:
        print("Submission Attempt Result:", type(e), e)
        if hasattr(e, 'response') and e.response is not None:
            print("Response Body:", e.response.text)
        return False, str(e)

if __name__ == '__main__':
    submit_v22()
