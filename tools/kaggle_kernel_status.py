import time
import os
import sys
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def check_kernel(kernel_id='cyruspinto/rsna-knee-engine5-decoupled-train', max_retries=3, delay=3):
    for attempt in range(max_retries):
        try:
            kaggle.api.authenticate()
            status = kaggle.api.kernels_status(kernel_id)
            print(f"Kernel: {kernel_id}")
            print(f"Status: {getattr(status, 'status', status)}")
            return status
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                print("Error querying kernel status after retries:", type(e), e)

if __name__ == '__main__':
    kernel = sys.argv[1] if len(sys.argv) > 1 else 'cyruspinto/rsna-knee-engine5-decoupled-train'
    check_kernel(kernel)
