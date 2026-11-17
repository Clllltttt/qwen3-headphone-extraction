"""Check exported data independently and measure local Qwen tokenizer lengths."""
import json
import itertools
from pathlib import Path
from transformers import AutoTokenizer
from build_dataset import ROOT, ASPECTS, SENTIMENTS, INTENTS, grams, similarity, normalized, write_json


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    model = Path(__file__).resolve().parents[2] / 'models/Qwen3-1.7B'
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    system = (ROOT / 'task_prompt.txt').read_text(encoding='utf-8')
    report, all_texts = {}, {}
    for split, expected in [('train', 500), ('validation', 100), ('test', 100)]:
        annotated = lines(ROOT / 'annotated' / f'{split}.jsonl')
        chats = lines(ROOT / f'{split}.jsonl')
        assert len(annotated) == len(chats) == expected
        lengths, answer_lengths = [], []
        for row, chat in zip(annotated, chats):
            messages = chat['messages']
            assert [m['role'] for m in messages] == ['system', 'user', 'assistant']
            assert messages[0]['content'] == system and messages[1]['content'] == row['text']
            answer = json.loads(messages[2]['content'])
            assert answer == row['answer'] and set(answer) == {'评价', '复购意愿'}
            assert answer['复购意愿'] in INTENTS
            assert len({x['方面'] for x in answer['评价']}) == len(answer['评价'])
            for item in answer['评价']:
                assert set(item) == {'方面', '情感', '依据'}
                assert item['方面'] in ASPECTS and item['情感'] in SENTIMENTS
                assert item['依据'] and item['依据'] in row['text']
            prefix = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            complete = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
            assert complete[:len(prefix)] == prefix
            assert len(complete) > len(prefix)
            lengths.append(len(complete))
            answer_lengths.append(len(complete) - len(prefix))
        all_texts[split] = [normalized(r['text']) for r in annotated]
        assert len(set(all_texts[split])) == expected
        ordered = sorted(lengths)
        report[split] = {'rows': expected, 'schema_and_evidence_substring_checks_passed': expected,
                         'token_lengths': {'min': min(lengths), 'median': ordered[len(ordered)//2],
                                           'p95': ordered[int(len(ordered)*0.95)], 'max': max(lengths)},
                         'rows_above_512_tokens': sum(n > 512 for n in lengths),
                         'rows_above_1024_tokens': sum(n > 1024 for n in lengths),
                         'minimum_answer_tokens_including_eos': min(answer_lengths),
                         'generation_prefix_matches_all_rows': True}
    for a, b in itertools.combinations(all_texts, 2):
        assert not set(all_texts[a]) & set(all_texts[b])
    blind = lines(ROOT / 'test_inputs.jsonl')
    gold = lines(ROOT / 'annotated/test.jsonl')
    assert len(blind) == 100
    for row, expected in zip(blind, gold):
        assert set(row) == {'id', 'messages'} and row['id'] == expected['id']
        assert [m['role'] for m in row['messages']] == ['system', 'user']
        assert row['messages'][1]['content'] == expected['text']
    report['test_inputs_contains_no_assistant_answers'] = True
    report['recommended_max_length'] = 1024
    report['limitations'] = 'Automated checks verify structure and substrings, not semantic truth. All sets are synthetic.'
    write_json(ROOT / 'token_audit.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
