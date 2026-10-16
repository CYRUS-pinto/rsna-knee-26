import os
import sys
import time
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def submit_v21():
    try:
        kaggle.api.authenticate()
        print("Authenticated with Kaggle API.")
        res = kaggle.api.competition_submit_code(
            file_name='submission.csv',
            message='V21 Quad-Engine Specialist (Raptor 40% + ConvNeXt 40% + Engine 5 2.5D 20%)',
            competition='rsna-knee-abnormality-detection',
            kernel='cyruspinto/rsna-knee-v21-quad-engine-blend'
        )
        print("Submission Response:", res)
        return True, res
    except Exception as e:
        print("Submission Attempt Result:", type(e), e)
        if hasattr(e, 'response') and e.response is not None:
            print("Response Body:", e.response.text)
        return False, str(e)

if __name__ == '__main__':
    submit_v21()
