import json, sys, time
from kaggle.api.kaggle_api_extended import KaggleApi
sys.stdout.reconfigure(encoding='utf-8')
COMP = 'rsna-knee-abnormality-detection'
api = KaggleApi(); api.authenticate()

topics, page_token = [], None
seen = set()
for page in range(1, 20):
    res = api.competition_list_topics(COMP, page=page)
    new = [t for t in res.topics if t.id not in seen]
    if not new:
        break
    for t in new:
        seen.add(t.id); topics.append(t)
print('topics fetched:', len(topics))

def flat(msgs, depth=0):
    out = []
    for m in msgs or []:
        out.append({'author': getattr(m, 'author_name', ''), 'date': str(getattr(m, 'post_date', '')),
                    'votes': getattr(m, 'votes', 0), 'depth': depth,
                    'text': getattr(m, 'raw_markdown', '') or getattr(m, 'content', '')})
        out += flat(getattr(m, 'replies', None), depth + 1)
    return out

data = []
for t in topics:
    try:
        r = api.competition_list_topic_messages(COMP, t.id)
        msgs = flat(r.messages)
    except Exception as e:
        msgs = [{'error': str(e)}]
    data.append({'id': t.id, 'title': t.title, 'messages': msgs})
    time.sleep(0.3)

json.dump(data, open('scratch/forum_full.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved', len(data), 'topics,', sum(len(d['messages']) for d in data), 'messages')
