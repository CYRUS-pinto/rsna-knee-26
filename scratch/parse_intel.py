import json
import re

with open('scratch/forum_intel.json', encoding='utf-8') as f:
    topics = json.load(f)

print(f"=== FORUM INTELLIGENCE REPORT ({len(topics)} TOPICS SCANNED) ===\n")

links = []
team_mergers = []
top_strategies = []

for t in topics:
    title = t.get('title', '')
    tid = t.get('id', '')
    msgs = t.get('messages', [])
    print(f"[{tid}] {title} ({len(msgs)} messages)")
    
    for m in msgs:
        text = m.get('text', '')
        # Extract URLs
        found_urls = re.findall(r'https?://[^\s\)\]\>]+', text)
        for u in found_urls:
            links.append((tid, title, u))
            
        # Detect Team Merger mentions
        if any(w in text.lower() or w in title.lower() for w in ['team', 'merge', 'looking for', 'partner', 'invite']):
            if any(w in text.lower() for w in ['teammate', 'merge', 'looking for', 'invite']):
                team_mergers.append((tid, title, m.get('date'), text[:250]))

        # Detect High Score alpha
        if any(w in text.lower() for w in ['0.95', '0.96', 'gold', 'cv', 'private lb', 'blend', 'foundation', 'trick']):
            top_strategies.append((tid, title, m.get('votes'), text[:200]))

print(f"\n--- DISCOVERED EXTERNAL LINKS & ARTIFACTS ({len(links)}) ---")
unique_links = set()
for tid, title, u in links:
    if u not in unique_links and not u.endswith(('.png', '.jpg', '.jpeg', '.gif')):
        unique_links.add(u)
        print(f"  * [{title[:30]}]: {u}")

print(f"\n--- TEAM MERGER RADAR ({len(team_mergers)}) ---")
for tid, title, dt, snippet in team_mergers[:5]:
    clean = ' '.join(snippet.split())
    print(f"  * [{tid}] {title} ({dt}): {clean[:140]}...")

print("\n--- DONE ---")
