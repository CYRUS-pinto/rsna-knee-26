import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/forum_intel.json', encoding='utf-8') as f:
    topics = json.load(f)

target_ids = [748295, 747613, 748011]
for t in topics:
    if t.get('id') in target_ids or any(k in t.get('title', '').lower() for k in ['stuck', 'bypass', '0.960', 'core to reach']):
        print(f"==================================================")
        print(f"TOPIC: {t.get('id')} - {t.get('title')}")
        print(f"==================================================")
        for m in t.get('messages', []):
            txt = m.get('text', '').strip()
            print(f"--- [Votes: {m.get('votes')}] ---")
            print(txt)
            print()
