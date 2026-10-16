import time
import os
import sys
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def query_with_retry(max_retries=3, delay=3):
    for attempt in range(max_retries):
        try:
            kaggle.api.authenticate()
            subs = kaggle.api.competition_submissions('rsna-knee-abnormality-detection')
            print(f"{'Ref':<10} {'Date':<20} {'Status':<25} {'PublicScore':<12} {'Description'}")
            print("-" * 100)
            for s in subs[:10]:
                ref = str(getattr(s, 'ref', ''))
                date = str(getattr(s, 'date', ''))[:19]
                status = str(getattr(s, 'status', ''))
                score = str(getattr(s, 'public_score', ''))
                desc = str(getattr(s, 'description', ''))[:45]
                print(f"{ref:<10} {date:<20} {status:<25} {score:<12} {desc}")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                print("Error querying Kaggle submissions after retries:", type(e), e)

if __name__ == '__main__':
    query_with_retry()
