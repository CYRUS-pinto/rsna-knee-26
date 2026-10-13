import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')
data = json.load(open('scratch/forum_full.json', encoding='utf-8'))
out = open('scratch/winners_report.md', 'w', encoding='utf-8')
def clean(t): return ' '.join(re.sub(r'!\[[^\]]*\]\([^)]*\)', '[img]', t).split())

KEY = re.compile(r'\b(cv|oof|fold|label|slice|backbone|convnext|coatnet|efficientnet|dino|resnet|vit|swin|llm|qwen|'
                 r'gpt|claude|gemini|external|oai|pseudo|distill|ensemble|blend|resolution|crop|augment|epoch|loss|'
                 r'gold|weak|noisy|soft|sagittal|coronal|axial|mil|attention|2\.5d|3d)\b', re.I)
HIGH = re.compile(r'0\.9(5[5-9]|6\d)')

msgs = [(t['title'], m) for t in data for m in t['messages'] if m.get('text')]
author_known = sum(1 for _, m in msgs if m.get('author'))
out.write(f'# Forum mining: {len(data)} topics, {len(msgs)} messages (authors exposed: {author_known})\n\n')

out.write('## A. Posts claiming >=0.955 AND describing method\n\n')
for title, m in msgs:
    tx = m['text']
    if HIGH.search(tx) and len(KEY.findall(tx)) >= 3 and len(tx) > 120:
        out.write(f'- **{title}** [votes {m.get("votes")}] {clean(tx)[:1200]}\n\n')

out.write('\n## B. Top 60 most-upvoted technical posts\n\n')
tech = [(m.get('votes') or 0, title, m['text']) for title, m in msgs if len(m['text']) > 150 and len(KEY.findall(m['text'])) >= 3]
tech.sort(key=lambda x: -x[0])
for v, title, tx in tech[:60]:
    out.write(f'- [{v}] **{title}** {clean(tx)[:1000]}\n\n')

out.write('\n## C. All topic titles\n\n')
for t in data:
    out.write(f'- {t["id"]} {t["title"]} ({len(t["messages"])})\n')
out.close()
print('done')
