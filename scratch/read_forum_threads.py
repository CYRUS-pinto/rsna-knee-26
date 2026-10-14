import json

with open('scratch/forum_full.json', 'r', encoding='utf-8') as f:
    forum = json.load(f)

with open('scratch/read_forum_threads.txt', 'w', encoding='utf-8') as out:
    for item in forum:
        title = item.get('title', '')
        if any(k in title.lower() for k in ['plateau at 0.903', 'advice', 'competition data', 'external', 'single sub 0.960']):
            out.write('====================================\n')
            out.write(f"TITLE: {title}\n")
            for m in item.get('messages', []):
                out.write(f"--- Author: {m.get('author')} (Votes: {m.get('votes')}) ---\n")
                out.write(f"{m.get('text')}\n\n")

print("Done writing to scratch/read_forum_threads.txt")
