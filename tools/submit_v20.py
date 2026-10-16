import os
import sys
import time
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def submit_v20():
    try:
        kaggle.api.authenticate()
        print("Authenticated with Kaggle API.")
        res = kaggle.api.competition_submit_code(
            file_name='submission.csv',
            message='V20 Generalized Power-Rank Master Blend (p=1.4)',
            competition='rsna-knee-abnormality-detection',
            kernel='cyruspinto/rsna-knee-v20-power-rank-blend'
        )
        print("Submission Response:", res)
        return True, res
    except Exception as e:
        print("Submission Attempt Result:", type(e), e)
        if hasattr(e, 'response') and e.response is not None:
            print("Response Body:", e.response.text)
        return False, str(e)

if __name__ == '__main__':
    submit_v20()
