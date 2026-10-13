"""
Automated Kaggle Competition Forum Intelligence Miner
Fetches competition topics, comments, and extracts critical alpha (scores, weights, datasets, team mergers).
"""
import os
import re
import json
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = 'rsna-knee-abnormality-detection'

def scrape_forum_intel(max_topics=25):
    api = KaggleApi()
    api.authenticate()
    
    print(f"[*] Querying topics for competition '{COMPETITION}'...")
    res = api.competition_list_topics(COMPETITION)
    topics = res.topics[:max_topics]
    print(f"[+] Found {len(topics)} topics. Extracting messages and community alpha...\n")
    
    intel_records = []
    
    for t in topics:
        topic_info = {
            'id': t.id,
            'title': t.title,
            'total_comments': getattr(t, 'total_comments', getattr(t, 'totalCount', 0)),
            'messages': []
        }
        
        try:
            msg_res = api.competition_list_topic_messages(COMPETITION, t.id)
            for m in msg_res.messages:
                raw = getattr(m, 'raw_markdown', '') or getattr(m, 'content', '')
                post_date = getattr(m, 'post_date', '')
                votes = getattr(m, 'votes', 0)
                topic_info['messages'].append({
                    'id': m.id,
                    'date': str(post_date),
                    'votes': votes,
                    'text': raw
                })
        except Exception as e:
            topic_info['error'] = str(e)
            
        intel_records.append(topic_info)
        
    return intel_records

if __name__ == '__main__':
    records = scrape_forum_intel(max_topics=20)
    out_path = os.path.join(os.path.dirname(__file__), 'forum_intel.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[✓] Saved intelligence to {out_path}")
