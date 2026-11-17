"""Evaluate a frozen validation/test split with optional saved LoRA; no training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import prepare_model_for_kbit_training
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('scorer', root / 'scoring.py')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    data = args.data or root.parent / 'data/qwen3-dataset-v1'
    model_dir = args.model
    split_file = data / f'{args.split}.jsonl'
    annotated_file = data / 'annotated' / f'{args.split}.jsonl'
    split_name = '测试集' if args.split == 'test' else '验证集'
    rows = [json.loads(line) for line in split_file.read_text(encoding='utf-8').splitlines()]
    annotated = [json.loads(line) for line in annotated_file.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len(annotated) == 100
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    assert digest(split_file) == manifest['files'][f'{args.split}.jsonl']
    assert digest(annotated_file) == manifest['files'][f'annotated/{args.split}.jsonl']
    for chat, record in zip(rows, annotated):
        assert chat['messages'][1]['content'] == record['text']
        assert json.loads(chat['messages'][2]['content']) == record['answer']
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    settings = {'created_utc': datetime.now(timezone.utc).isoformat(), 'model': str(model_dir),
                'adapter': str(args.adapter) if args.adapter else None, 'split': args.split, 'rows': 100, 'batch_size': args.batch_size,
                'do_sample': False, 'num_beams': 1, 'seed': 42, 'enable_thinking': False,
                'max_new_tokens': 512, 'padding_side': 'left', 'compute_dtype': str(dtype),
                'quantization': 'NF4, double quantization', 'attention': 'sdpa',
                'base_preparation': 'prepare_model_for_kbit_training without gradient checkpointing',
                'autocast': str(dtype), 'torch': torch.__version__, 'transformers': transformers.__version__,
                f'{args.split}_sha256': digest(split_file),
                'annotated_sha256': digest(annotated_file),
                'system_sha256': digest(data / 'task_prompt.txt'), 'scorer_sha256': digest(root / 'scoring.py'),
                'test_split_accessed': args.split == 'test', 'training_performed': False,
                'model_revision': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'}
    save(args.output / 'protocol.json', settings)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, device_map={'': 0}, dtype=dtype,
        attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True, is_trainable=False)
        settings['adapter_sha256'] = digest(args.adapter / 'adapter_model.safetensors')
        save(args.output / 'protocol.json', settings)
    model.eval()
    model.config.use_cache = True
    assert not any(p.requires_grad for p in model.parameters())
    assert any('lora_' in name for name, _ in model.named_parameters()) == bool(args.adapter)
    print('MODEL_READY: all parameters frozen; adapter=' + str(args.adapter), flush=True)
    records = []
    start = time.perf_counter()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        prompts = [tokenizer.apply_chat_template(row['messages'][:2], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for row in batch]
        inputs = tokenizer(prompts, padding=True, return_tensors='pt', add_special_tokens=False).to('cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=dtype):
            generated = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1,
                                       temperature=None, top_p=None, top_k=None,
                                       pad_token_id=tokenizer.eos_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
        for j, tokens in enumerate(generated[:, inputs['input_ids'].shape[1]:]):
            source = annotated[offset + j]
            ids = tokens.tolist()
            stopped = tokenizer.eos_token_id in ids
            count = ids.index(tokenizer.eos_token_id) + 1 if stopped else len(ids)
            answer = tokenizer.decode(ids[:count], skip_special_tokens=True)
            case = {'text': source['text'], 'gold': source['answer']}
            record = {**case, 'id': source['id'], 'features': source['features'], 'answer': answer,
                      'generated_tokens': count, 'stopped_at_eos': stopped,
                      'score': scorer.score_case(case, answer)}
            records.append(record)
            with (args.output / 'predictions.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps({'completed': len(records), 'exact': sum(r['score']['semantic_exact'] for r in records),
                          'seconds': round(time.perf_counter() - start, 1)}), flush=True)
    sums = Counter()
    for r in records:
        sums.update({k: r['score'][k] for k in ['tp', 'predicted', 'gold', 'schema_ok', 'intent_ok',
                     'semantic_exact', 'all_ok', 'evidence_substrings_valid', 'evidence_total']})
    precision = sums['tp'] / max(1, sums['predicted'])
    recall = sums['tp'] / max(1, sums['gold'])
    metrics = {'completed': len(records), 'aspect_sentiment_micro_precision': precision,
               'aspect_sentiment_micro_recall': recall,
               'aspect_sentiment_micro_f1': 2 * precision * recall / max(1e-10, precision + recall),
               'repurchase_accuracy': sums['intent_ok'] / 100, 'schema_accuracy': sums['schema_ok'] / 100,
               'semantic_exact_accuracy': sums['semantic_exact'] / 100, 'all_checks_accuracy': sums['all_ok'] / 100,
               'evidence_substring_rate': sums['evidence_substrings_valid'] / max(1, sums['evidence_total']),
               'length_limit_hits': sum(not r['stopped_at_eos'] for r in records), 'counts': dict(sums),
               'inference_seconds': time.perf_counter() - start,
               'peak_gpu_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
               'split': args.split,
               'limitation': f'Synthetic {args.split} set; substring checks do not establish semantic evidence support.'}
    save(args.output / 'metrics.json', metrics)
    report = [(f'# LoRA 微调后{split_name}评测' if args.adapter else f'# 未微调 Qwen3-1.7B {split_name}基线'), '',
              ('加载已保存的 LoRA。' if args.adapter else '未加载 LoRA。') + f'NF4 双重量化，关闭思考，贪心生成；本次评测不训练，评测范围为{split_name}。', '',
              '| 指标 | 结果 |', '| --- | ---: |']
    for label, key in [('方面与情感 Micro-F1', 'aspect_sentiment_micro_f1'), ('复购意愿准确率', 'repurchase_accuracy'),
                       ('JSON 格式合规率', 'schema_accuracy'), ('整条标签与复购全部正确', 'semantic_exact_accuracy'),
                       ('标签、格式与依据原文检查全部通过', 'all_checks_accuracy'), ('依据原文匹配率', 'evidence_substring_rate')]:
        report.append(f'| {label} | {metrics[key]:.1%} |')
    report += ['', f"完成 100 条，生成长度上限触发 {metrics['length_limit_hits']} 条。",
               '整条正确指标不要求依据与参考答案字面相同。依据检查仅验证连续原文，不自动判断是否支持情感结论。',
               '数据是合成评论，成绩只用于本实验对照，不能代表真实评论总体水平。',
               '微调后应复用 protocol.json 中的量化、提示、精度、批量及生成设置。', '', '## 未全部通过的样本', '']
    for r in records:
        if r['score']['all_ok']:
            continue
        report += [f"### {r['id']}", '', r['text'], '', '参考答案：', '```json',
                   json.dumps(r['gold'], ensure_ascii=False, indent=2), '```', '', '模型输出：', '```text', r['answer'], '```', '']
    (args.output / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print('RESULT ' + json.dumps(metrics, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
