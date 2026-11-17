"""Train a fresh QLoRA adapter on a frozen synthetic training split."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    import torch
    import bitsandbytes as bnb
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed, EarlyStoppingCallback
    set_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    root = Path(__file__).resolve().parent
    data = args.data or root.parent / 'data/qwen3-dataset-v2'
    model_dir = args.model
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    datasets, audit = {}, {}
    for split, expected in [('train', args.expected_train_rows), ('validation', 100)]:
        path = data / f'{split}.jsonl'
        assert sha(path) == manifest['files'][path.name]
        raw = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert len(raw) == expected
        encoded = []
        for row in raw:
            messages = row['messages']
            assert [m['role'] for m in messages] == ['system', 'user', 'assistant']
            prefix = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
            assert full[:len(prefix)] == prefix
            assert tokenizer.decode(full[len(prefix):]).startswith(messages[2]['content'])
            end = full.index(tokenizer.eos_token_id, len(prefix))
            full = full[:end + 1]
            assert len(full) <= 1024
            labels = [-100] * len(prefix) + full[len(prefix):]
            assert labels[-1] == tokenizer.eos_token_id
            encoded.append({'input_ids': full, 'attention_mask': [1] * len(full), 'labels': labels})
        datasets[split] = Dataset.from_list(encoded)
        audit[split] = {'rows': len(raw), 'sha256': sha(path), 'max_sequence_length': max(len(x['input_ids']) for x in encoded),
                        'min_answer_tokens': min(sum(t != -100 for t in x['labels']) for x in encoded)}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config = {'epochs': args.epochs, 'early_stopping_patience': args.patience,
              'early_stopping_min_delta': 0.0, 'train_batch_size': 1, 'gradient_accumulation_steps': 4, 'learning_rate': 1e-4,
              'scheduler': 'cosine', 'warmup_ratio': 0.05, 'max_length': 1024, 'lora_rank': 8,
              'lora_alpha': 16, 'lora_dropout': 0.0, 'target_modules': ['q_proj', 'v_proj'],
              'compute_dtype': str(dtype), 'quantization': 'NF4 double quantization', 'seed': 42,
              'checkpoint_selection': 'lowest validation teacher-forced loss after each epoch',
              'loss_scope': 'assistant JSON plus EOS only; system, user and padding masked',
              'base_model': str(model_dir), 'fresh_adapter': True, 'test_split_accessed': False,
              'data': audit, 'torch': torch.__version__, 'transformers': transformers.__version__,
              'script_sha256': sha(Path(__file__)), 'created_at': datetime.now().isoformat()}
    save(args.output / 'training-config.json', config)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, dtype=dtype, device_map={'': 0},
        attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias='none',
                                            task_type='CAUSAL_LM', target_modules=['q_proj', 'v_proj']))
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all('lora_' in name for name in trainable)
    before = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    frozen_name, frozen = next((n, p) for n, p in model.named_parameters() if isinstance(p, bnb.nn.Params4bit))
    frozen_before = frozen.detach().cpu().clone()
    config['trainable_parameters'] = sum(p.numel() for p in trainable.values())
    save(args.output / 'training-config.json', config)

    def collate(rows):
        batch = tokenizer.pad([{'input_ids': r['input_ids'], 'attention_mask': r['attention_mask']} for r in rows],
                              padding=True, return_tensors='pt')
        width = batch['input_ids'].shape[1]
        batch['labels'] = torch.tensor([r['labels'] + [-100] * (width - len(r['labels'])) for r in rows], dtype=torch.long)
        return batch

    class Progress(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kwargs):
            row = {'step': state.global_step, **(logs or {})}
            with (args.output / 'progress.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            print('PROGRESS ' + json.dumps(row), flush=True)

    training_args = TrainingArguments(
        output_dir=str(args.output / 'checkpoints'), num_train_epochs=args.epochs, per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=4, learning_rate=1e-4,
        lr_scheduler_type='cosine', warmup_ratio=0.05, optim='adamw_torch', weight_decay=0.0, max_grad_norm=1.0,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, logging_steps=10,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False, prediction_loss_only=True,
        report_to='none', dataloader_num_workers=0, dataloader_pin_memory=False,
        seed=42, data_seed=42, disable_tqdm=True)
    early_stop = EarlyStoppingCallback(early_stopping_patience=args.patience, early_stopping_threshold=0.0) if args.patience else None
    trainer = Trainer(model=model, args=training_args, train_dataset=datasets['train'],
                      eval_dataset=datasets['validation'], data_collator=collate,
                      processing_class=tokenizer, callbacks=[Progress()] + ([early_stop] if early_stop else []))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    baseline_loss = trainer.evaluate()['eval_loss']
    print('START_TRAINING ' + json.dumps(config, ensure_ascii=True), flush=True)
    result = trainer.train()
    max_steps = args.epochs * math.ceil(args.expected_train_rows / 4)
    assert 0 < trainer.state.global_step <= max_steps
    if not early_stop:
        assert trainer.state.global_step == max_steps
    elif trainer.state.global_step < max_steps:
        assert early_stop.early_stopping_patience_counter >= args.patience
    assert math.isfinite(result.training_loss)
    assert torch.equal(frozen_before, frozen.detach().cpu())
    assert all(not p.requires_grad for n, p in model.named_parameters() if 'lora_' not in n)
    delta = sum((p.detach().cpu() - before[n]).float().square().sum().item() for n, p in trainable.items()) ** 0.5
    assert delta > 0
    adapter = args.output / 'adapter'
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    trainer.save_state()
    report = {'status': 'training_completed', 'optimizer_steps': trainer.state.global_step,
              'epochs': trainer.state.epoch, 'max_epochs': args.epochs,
              'early_stopping_patience': args.patience,
              'early_stopping_triggered': bool(early_stop and early_stop.early_stopping_patience_counter >= args.patience),
              'stopped_before_max_epochs': trainer.state.global_step < max_steps,
              'training_rows': args.expected_train_rows, 'validation_rows': 100,
              'baseline_validation_loss': baseline_loss, 'best_validation_loss': trainer.state.best_metric,
              'selected_checkpoint': trainer.state.best_model_checkpoint, 'adapter': str(adapter),
              'trainable_parameters': config['trainable_parameters'], 'adapter_delta_l2': delta,
              'all_base_parameters_frozen': True, 'sampled_frozen_weight_unchanged': True,
              'sampled_frozen_weight': frozen_name, 'metrics': result.metrics,
              'peak_gpu_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
              'seconds_including_validation': time.perf_counter() - started,
              'note': 'Selected adapter saved; generation evaluation must load this adapter in a fresh process.'}
    save(args.output / 'training-result.json', report)
    print('RESULT ' + json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Local Qwen3-1.7B directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--expected-train-rows', type=int, default=700)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=2, help='Consecutive non-improving epoch evaluations; zero disables early stopping')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 0:
        parser.error('epochs must be positive and patience must be nonnegative')
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            save(args.output / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise
