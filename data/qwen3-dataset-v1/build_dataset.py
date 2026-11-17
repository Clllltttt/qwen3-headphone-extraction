"""Build reproducible synthetic datasets; never loads or trains the model."""
import hashlib
import itertools
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASPECTS = ['音质', '舒适度', '续航', '连接', '物流', '客服', '价格']
SENTIMENTS = ['正面', '负面', '中性', '正负混合']
INTENTS = ['未明确', '愿意', '不愿意']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if c.isalnum())


def grams(text):
    value = normalized(text)
    return {value[i:i + 3] for i in range(max(1, len(value) - 2))}


def similarity(left, right):
    return len(left & right) / max(1, len(left | right))


def main():
    phrases = read(ROOT / 'phrase_bank.json')
    scenarios = read(ROOT / 'scenario_bank.json')
    system = read(ROOT.parent / 'qwen3-aspect-baseline/baseline-run/protocol.json')['system']
    (ROOT / 'task_prompt.txt').write_text(system, encoding='utf-8')
    old = []
    for rel in ['qwen3-aspect-baseline/cases.json', 'qwen3-aspect-baseline/fewshot_examples.json',
                'qwen3-aspect-training/training_samples.json']:
        data = read(ROOT.parent / rel)
        if isinstance(data, dict):
            data = data.get('cases', data.get('examples', []))
        old.extend(row['text'] for row in data)
    prior = [(normalized(text), grams(text)) for text in old]
    manifest = {'seed': 42, 'provenance': 'synthetic phrase composition, not real customer reviews',
                'previous_example_count': len(old), 'near_duplicate_threshold': 0.65,
                'system_sha256': hashlib.sha256(system.encode()).hexdigest(), 'splits': {}}
    atom_sets = {}
    for split in phrases:
        atom_sets[split] = {p for values in phrases[split].values() for group in values.values() for p in group}
        atom_sets[split].update(row['text'] for row in scenarios[split]['specials'])
    assert all(not atom_sets[a] & atom_sets[b] for a, b in itertools.combinations(atom_sets, 2))
    for split, size in [('train', 500), ('validation', 100), ('test', 100)]:
        rng = random.Random(42 + size + len(split))
        scene = scenarios[split]
        counts, features, cells = Counter(), Counter(), Counter()
        rows, seen, cores = [], set(), set()
        intents = ['未明确'] * (size * 6 // 10) + ['愿意'] * (size // 5) + ['不愿意'] * (size // 5)
        lengths = [0] * (size // 10) + [1] * (size // 5) + [2] * (size * 4 // 10) + [3] * (size * 3 // 10)
        rng.shuffle(intents)
        rng.shuffle(lengths)
        rejected = Counter()
        max_cross = 0.0
        for index, (intent, count) in enumerate(zip(intents, lengths), 1):
            for attempt in range(10000):
                atoms, tags = [], []
                if count and index % 4 == 0:
                    special = scene['specials'][(index // 4) % len(scene['specials'])]
                    atoms.append({'aspect': special['aspect'], 'sentiment': special['sentiment'],
                                  'text': special['text'], 'evidence': special['evidence'], 'source': special['id']})
                    tags.append(special['feature'])
                while len(atoms) < count:
                    available = [a for a in ASPECTS if a not in {x['aspect'] for x in atoms}]
                    if '转述实际使用者' in tags:
                        available = [a for a in available if a in ['物流', '客服', '价格']]
                    options = [(a, s) for a in available for s in SENTIMENTS]
                    rng.shuffle(options)
                    # Prefer underrepresented cells while keeping combinations variable.
                    a, s = min(options, key=lambda pair: cells[pair] + rng.random() * 3)
                    bank = phrases[split][a][s]
                    phrase_index = rng.randrange(len(bank))
                    phrase = bank[phrase_index]
                    atoms.append({'aspect': a, 'sentiment': s, 'text': phrase, 'evidence': phrase,
                                  'source': f'{split}/{a}/{s}/{phrase_index}'})
                rng.shuffle(atoms)
                parts = [x['text'] for x in atoms] if count else [rng.choice(scene['nulls'])]
                if not count:
                    tags.append('无目标方面评价')
                if count and index % 5 == 0:
                    # Avoid untested-feature disclaimers that contradict actual-use narratives.
                    missing = [x for x in scene['missing'] if x['aspect'] not in {a['aspect'] for a in atoms}
                               and x['aspect'] in ['客服', '价格', '物流']]
                    if missing:
                        parts.append(rng.choice(missing)['text'])
                        tags.append('仅提及不评价')
                suffix = rng.choice(scene['intents'][intent])
                if suffix:
                    parts.append(suffix.rstrip('。'))
                if index % 13 == 0:
                    parts.append(rng.choice(scene['injections']).rstrip('。'))
                    tags.append('评论内指令')
                core = tuple(sorted(parts))
                text_value = rng.choice(scene['joins']).join(parts) + '。'
                key, fingerprints = normalized(text_value), grams(text_value)
                if key in seen or core in cores:
                    rejected['duplicate_within_split'] += 1
                    continue
                cross = max((similarity(fingerprints, p[1]) for p in prior), default=0)
                if any(key == p[0] for p in prior) or cross >= 0.65:
                    rejected['cross_split_or_old_near_duplicate'] += 1
                    continue
                answer = {'评价': [{'方面': a['aspect'], '情感': a['sentiment'], '依据': a['evidence']}
                                 for a in atoms], '复购意愿': intent}
                assert all(a['依据'] and a['依据'] in text_value for a in answer['评价'])
                assert len({a['方面'] for a in answer['评价']}) == count
                rows.append({'id': f'{split}-{index:04d}', 'text': text_value, 'answer': answer,
                             'features': tags, 'sources': [a['source'] for a in atoms], 'synthetic': True})
                seen.add(key)
                cores.add(core)
                features.update(tags)
                cells.update((a['aspect'], a['sentiment']) for a in atoms)
                counts[intent] += 1
                max_cross = max(max_cross, cross)
                break
            else:
                raise RuntimeError(f'Cannot make unique example: {split}/{index}')
        assert len(rows) == size and all(cells[(a, s)] > 0 for a in ASPECTS for s in SENTIMENTS)
        write_lines(ROOT / 'annotated' / f'{split}.jsonl', rows)
        chats = [{'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': row['text']},
                              {'role': 'assistant', 'content': json.dumps(row['answer'], ensure_ascii=False)}]}
                 for row in rows]
        write_lines(ROOT / f'{split}.jsonl', chats)
        if split == 'test':
            write_lines(ROOT / 'test_inputs.jsonl', [{'id': row['id'], 'messages': chat['messages'][:2]}
                                                    for row, chat in zip(rows, chats)])
        else:
            preview = [f'# {split} 数据预览\n', '以下为合成实验数据，完整数据见同名 JSONL。\n']
            for row in rows[:10]:
                preview.extend([f"## {row['id']}\n", row['text'] + '\n',
                                '```json\n' + json.dumps(row['answer'], ensure_ascii=False, indent=2) + '\n```\n'])
            (ROOT / f'{split}_preview.md').write_text('\n'.join(preview), encoding='utf-8')
        manifest['splits'][split] = {'rows': size, 'aspect_count_distribution': dict(Counter(len(r['answer']['评价']) for r in rows)),
                                     'repurchase_distribution': dict(counts), 'features': dict(features),
                                     'aspect_sentiment_counts': {f'{a}/{s}': cells[a, s] for a in ASPECTS for s in SENTIMENTS},
                                     'rejected_candidates': dict(rejected), 'max_similarity_to_previous_splits_and_old': round(max_cross, 6),
                                     'exact_duplicate_rows': 0, 'duplicate_compositions_ignoring_order': 0}
        prior.extend((normalized(r['text']), grams(r['text'])) for r in rows)
    manifest['cross_split_evaluation_phrase_overlap'] = 0
    manifest['files'] = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ['.jsonl', '.json', '.txt']
                         and p.name not in ['manifest.json', 'token_audit.json']}
    write_json(ROOT / 'manifest.json', manifest)
    print(json.dumps({k: v['rows'] for k, v in manifest['splits'].items()}))


if __name__ == '__main__':
    main()
