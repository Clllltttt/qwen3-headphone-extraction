"""Build 200 targeted synthetic examples, audit them, mix with frozen v1 train."""
import hashlib
import itertools
import json
import random
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / 'qwen3-dataset-v1'
ASPECTS = ['音质', '舒适度', '续航', '连接', '物流', '客服', '价格']

# Each row contrasts non-evaluation, explicit neutral, and current mixed evaluation.
BANK = {
    '音质': {
        '未评价': ['还没选曲试听，声音好坏我答不上来', '宣传页夸声场开阔，我没有听过，不能认可也不能否定'],
        '中性': ['声音听着就是寻常水准，我给的评价是一般', '试听后对音效持中立态度，既不称赞也不批评'],
        '正负混合': ['低频弹性好让我喜欢，高频又刺得难受，听感有赞有怨', '听独奏细节丰富很讨喜，听大编制却混乱难听，两种听感各有好坏'],
    },
    '舒适度': {
        '未评价': ['耳机还留在收纳袋里没戴过，佩戴舒服与否无法回答', '店员说上耳很轻巧，这只是介绍，我还没有佩戴体验'],
        '中性': ['上耳试过后觉得舒适性一般，没有特别好或差', '这副的佩戴感在我这里是中间档，我不褒也不贬'],
        '正负混合': ['耳罩内衬软和很舒服，头顶却被压得酸痛，佩戴有利有弊', '静坐时戴着放松值得夸，低头时耳缘被硌疼又让我不满'],
    },
    '续航': {
        '未评价': ['只是接上充电器，没有用到电量耗尽，电池耐力不下结论', '页面标着可播三十小时，但我没实际测，续航暂不打分'],
        '中性': ['完整用过一轮后觉得电池持久性普通，评价居中', '续航给我的感受是一般般，没觉得出色也没觉得糟'],
        '正负混合': ['低功耗模式撑得久让我高兴，普通模式很快没电让我失落', '单耳使用电量很经用值得夸，双耳同时开却掉电太快让人烦'],
    },
    '连接': {
        '未评价': ['无线功能尚未开启，不能判断配对表现', '看到广告说能无缝切换设备，我没试过，不算使用评价'],
        '中性': ['实际配过几台设备，连接这一项只能说平常，无褒无贬', '无线体验在我看来中规中矩，我的态度不偏好也不偏坏'],
        '正负混合': ['连接笔记本快捷稳定值得夸，连随身播放器却屡试屡败让我恼火', '同屋使用信号稳定很省心，走到阳台老断让我不满，连接好坏都有'],
    },
    '物流': {
        '未评价': ['只登记了运单号，快递好不好我没发表意见', '知道签收是在周五，但对配送质量我没有作评价'],
        '中性': ['这单收货过程给我的感觉普通，没觉得满意也没觉得不满', '配送我打中间分，没有值得表扬或指责的地方'],
        '正负混合': ['仓库出货利索让我欣慰，快递员漏送又让我火大，配送有好有坏', '包裹转运速度快值得夸，约定送货时间却一拖再拖让我不满'],
    },
    '客服': {
        '未评价': ['从买来到现在没有向商家咨询，服务水平还无从得知', '只看见有在线客服入口，没有交谈，服务先不评价'],
        '中性': ['问过人工之后觉得接待一般，没有特别满意或失望', '客服给我的感觉是普通水平，谈不上优劣'],
        '正负混合': ['咨询时解释清楚让我满意，办理退换却互相推诿让我不满', '接线员语气和善值得表扬，答应的回电一再落空又让人生气'],
    },
    '价格': {
        '未评价': ['小票写着实付一百七十六元，是否值得我没有表态', '这里只记录券后金额为二百一十九元，不涉及便宜贵或划算与否'],
        '中性': ['买下来对花费的评价是中立，谈不上值或不值', '我认为售价处于一般水平，没有赞同或反感'],
        '正负混合': ['会员价我认为合算，非会员价又觉得吃亏，价格有褒有贬', '单副售价值得买，两副套装反而很不值，两个价格让我有不同评价'],
    },
}

# Repeat characters and negations must survive evidence copying unchanged.
COPY_BANK = [
    ('连接', '负面', '信号断断续续，几乎句句都漏，连接让我烦透了'),
    ('连接', '负面', '不是没有掉线，是一遍一遍掉线，蓝牙体验很差'),
    ('连接', '正面', '连接没有断过，真的没有断过，这点我很满意'),
    ('音质', '负面', '歌声里夹着“滋滋、滋滋”的噪声，听起来特别难受'),
    ('音质', '正面', '不是不好听，确实是清清楚楚很好听，我很喜欢这声音'),
    ('舒适度', '负面', '耳朵疼了又疼，摘下再戴还是疼，佩戴让人受罪'),
    ('舒适度', '正面', '戴着不挤，也不硌，连续使用仍然很舒服'),
    ('续航', '负面', '充了又充还是不经用，电池表现真让人失望'),
    ('续航', '正面', '电池耐用是真耐用，一路用下来都没断电，很满意'),
    ('物流', '负面', '这快递真够“及时”：说好昨天到，今天仍没派送，让人窝火'),
    ('物流', '负面', '配送真是“快极了”，足足等了十八天，等得我火大'),
    ('物流', '正面', '快递按时到了，按时到这件事让我满意'),
    ('客服', '负面', '售后让我等等、再等等，最后还是没答复，服务很差'),
    ('客服', '正面', '人工把细节一项一项说明白，服务让我安心'),
    ('价格', '负面', '不是没优惠，是优惠了我仍觉得不值，不值这笔支出'),
    ('价格', '正面', '买得值就是买得值，这个花费我认可'),
    ('连接', '正负混合', '首次连接很顺手值得夸，再次连接却总失败令人不快'),
    ('音质', '正负混合', '细节清晰让我喜欢，可沙沙沙的底噪又让我厌烦'),
    ('物流', '正负混合', '发出得快我赞成，送达得慢我抱怨，物流两头不一样'),
    ('舒适度', '正负混合', '耳垫柔柔软软很舒服，头梁紧紧绷绷又压得疼'),
]
OUTSIDE = [
    '盒子上那朵花的图案太俗了',
    '印在外盒上的字很好看',
    '贴纸的配色真漂亮',
    '商品展示图的构图不好看',
    '赠送的书签图案精美',
    '挂绳印花不合审美',
    '收纳袋的商标图形很讨喜',
    '外包装的插画太丑',
    '礼盒烫金文字很漂亮',
    '说明书封面排版难看',
]
INTENTS = {
    '未明确': ['', '还没想好是否再次购买相同型号。', '等明年看情况再说，现在没有再购决定。', '假使价格合适也许再考虑，眼下没有决定。'],
    '愿意': ['已经定下再买一副本型号。', '今后需要时肯定再次购买这一款。'],
    '不愿意': ['确定以后不再购入这一款耳机。', '再次购买时会选别的型号，不会再选当前这款。'],
}


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def norm(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if c.isalnum())


def grams(text):
    s = norm(text)
    return {s[i:i+3] for i in range(max(1, len(s)-2))}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from transformers import AutoTokenizer
    rng = random.Random(20261106)
    system = (V1 / 'task_prompt.txt').read_text(encoding='utf-8')
    old = read_lines(V1 / 'train.jsonl')
    assert len(old) == 500
    # The old evaluation sets are exclusions only, never copied into training.
    exclusions = [r['messages'][1]['content'] for split in ['train', 'validation', 'test']
                  for r in read_lines(V1 / f'{split}.jsonl')]
    old_fingerprints = [grams(t) for t in exclusions]
    forbidden = {norm(t) for t in exclusions}
    val_test_atoms = json.loads((V1 / 'phrase_bank.json').read_text(encoding='utf-8'))
    eval_phrases = {p for split in ['validation', 'test'] for sentiments in val_test_atoms[split].values()
                    for phrases in sentiments.values() for p in phrases}
    new_atoms = {p for aspect in BANK.values() for phrases in aspect.values() for p in phrases}
    new_atoms.update(text for _, _, text in COPY_BANK)
    assert not new_atoms & eval_phrases
    supplemental, seen, rejected = [], set(), Counter()
    for category, size in [('未评价边界', 40), ('明确中性', 40), ('同方面混合', 40),
                           ('逐字引用', 40), ('范围外评价', 20), ('多方面完整提取', 20)]:
        for index in range(size):
            for attempt in range(1000):
                a = ASPECTS[(index + attempt) % 7]
                variant = (index // 7 + attempt // 7) % 2
                segments, items = [], []

                def add(aspect, sentiment, text, evidence=None):
                    segments.append(text)
                    if sentiment != '未评价':
                        assert aspect not in {x['方面'] for x in items}
                        items.append({'方面': aspect, '情感': sentiment, '依据': evidence or text})

                def bank(aspect, sentiment, v):
                    add(aspect, sentiment, BANK[aspect][sentiment][v % 2])

                if category == '未评价边界':
                    bank(a, '未评价', variant)
                    if index % 2:
                        bank(ASPECTS[(ASPECTS.index(a) + 2) % 7], '未评价', 1 - variant)
                elif category == '明确中性':
                    bank(a, '中性', variant)
                    bank(ASPECTS[(ASPECTS.index(a) + 1 + index // 14) % 7], '未评价', 1 - variant)
                elif category == '同方面混合':
                    bank(a, '正负混合', variant)
                    bank(ASPECTS[(ASPECTS.index(a) + 2 + index // 14) % 7], '中性', 1 - variant)
                elif category == '逐字引用':
                    aspect, sentiment, text = COPY_BANK[(index + attempt) % len(COPY_BANK)]
                    add(aspect, sentiment, text)
                    other = [x for x in ASPECTS if x != aspect][(index // 20 + attempt) % 6]
                    bank(other, '中性', variant)
                elif category == '范围外评价':
                    segments.append(OUTSIDE[(index + attempt) % 10])
                    if index >= 10:
                        bank(a, '中性' if index % 2 else '正负混合', variant)
                else:
                    bank(a, '中性', variant)
                    bank(ASPECTS[(ASPECTS.index(a) + 2) % 7], '正负混合', 1 - variant)
                    choices = [x for x in COPY_BANK if x[0] not in {item['方面'] for item in items}]
                    aspect, sentiment, text = choices[(index + attempt) % len(choices)]
                    add(aspect, sentiment, text)
                intent = ['未明确', '未明确', '未明确', '愿意', '不愿意'][index % 5]
                suffix = rng.choice(INTENTS[intent])
                # Keep answer order aligned with evaluation clauses; no commentary-only padding.
                text = '。'.join(segments) + '。' + suffix
                key = norm(text)
                if key in seen or key in forbidden:
                    rejected['duplicate'] += 1
                    continue
                fp = grams(text)
                near = max(len(fp & prev) / max(1, len(fp | prev)) for prev in old_fingerprints)
                if near >= 0.65:
                    rejected['near_old_data'] += 1
                    continue
                assert all(item['依据'] in text for item in items)
                record = {'id': f'supp-{len(supplemental)+1:04d}', 'category': category,
                          'text': text, 'answer': {'评价': items, '复购意愿': intent},
                          'synthetic': True, 'max_similarity_to_v1': near}
                supplemental.append(record)
                seen.add(key)
                break
            else:
                raise RuntimeError(f'No unique row: {category}/{index}')
    assert len(supplemental) == 200
    chats = [{'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': r['text']},
                           {'role': 'assistant', 'content': json.dumps(r['answer'], ensure_ascii=False)}]}
             for r in supplemental]
    mixed = old + chats
    rng.shuffle(mixed)
    assert len(mixed) == len({norm(r['messages'][1]['content']) for r in mixed}) == 700
    original_counts = Counter(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in old)
    mixed_counts = Counter(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in mixed)
    assert original_counts <= mixed_counts
    tokenizer = AutoTokenizer.from_pretrained(ROOT.parents[1] / 'models/Qwen3-1.7B', local_files_only=True)
    lengths = []
    for row in mixed:
        messages = row['messages']
        answer = json.loads(messages[2]['content'])
        assert set(answer) == {'评价', '复购意愿'}
        assert len({x['方面'] for x in answer['评价']}) == len(answer['评价'])
        for item in answer['评价']:
            assert set(item) == {'方面', '情感', '依据'}
            assert item['方面'] in ASPECTS and item['情感'] in ['正面', '负面', '中性', '正负混合']
            assert item['依据'] and item['依据'] in messages[1]['content']
        prefix = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        full = tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=False)
        assert full[:len(prefix)] == prefix
        assert len(full) <= 1024
        lengths.append(len(full))
    write_lines(ROOT / 'supplement_200.jsonl', chats)
    write_lines(ROOT / 'supplement_annotated.jsonl', supplemental)
    write_lines(ROOT / 'train.jsonl', mixed)
    shutil.copyfile(V1 / 'validation.jsonl', ROOT / 'validation.jsonl')
    shutil.copyfile(V1 / 'task_prompt.txt', ROOT / 'task_prompt.txt')
    bank_doc = {'contrast_bank': BANK, 'copy_bank': COPY_BANK, 'outside_bank': OUTSIDE, 'intent_bank': INTENTS}
    save(ROOT / 'authored_bank.json', bank_doc)
    manifest = {'seed': 20261106, 'training_rows': 700, 'original_rows': 500, 'new_rows': 200,
                'validation_rows': 100, 'original_500_preserved': True,
                'validation_unchanged': sha(ROOT / 'validation.jsonl') == sha(V1 / 'validation.jsonl'),
                'task_prompt_unchanged': sha(ROOT / 'task_prompt.txt') == sha(V1 / 'task_prompt.txt'),
                'supplement_categories': dict(Counter(r['category'] for r in supplemental)),
                'supplement_repurchase': dict(Counter(r['answer']['复购意愿'] for r in supplemental)),
                'exact_duplicates_with_v1': 0, 'supplement_exact_duplicate_rows': 0,
                'max_similarity_to_v1': max(r['max_similarity_to_v1'] for r in supplemental),
                'rejected_candidates': dict(rejected), 'training_max_tokens': max(lengths),
                'new_training_phrase_count': len(new_atoms),
                'data_provenance': 'synthetic combinations of authored atoms; not real comments or independent human annotation',
                'test_status': 'v1 test accessed only for exclusion and previously analyzed; use as regression, not fresh final evaluation',
                'files': {p.name: sha(p) for p in ROOT.iterdir() if p.suffix in ['.jsonl', '.txt']}}
    save(ROOT / 'manifest.json', manifest)
    preview = ['# 新增200条针对性训练样本', '',
               '这些是新编素材的合成组合，已做格式、标签集合、证据原文、去重和分词长度检查；不是独立人工标注的真实评论。', '',
               '原500条保持内容不变，与新增200条打乱混合为700条。原验证集和提示词不变。', '',
               '## 类别', '', '| 主要目标 | 条数 |', '| --- | ---: |']
    preview += [f'| {k} | {v} |' for k, v in manifest['supplement_categories'].items()]
    preview += ['', '## 全部新增样本', '']
    for row in supplemental:
        preview += [f"### {row['id']}：{row['category']}", '', row['text'], '', '```json',
                    json.dumps(row['answer'], ensure_ascii=False, indent=2), '```', '']
    (ROOT / '新增样本说明与预览.md').write_text('\n'.join(preview), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
