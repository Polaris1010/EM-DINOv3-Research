"""Equal-source and equal-benchmark means, sample SD, paired seed deltas; no seed selection."""
import csv
import math
import statistics
from pathlib import Path
from common import R, atomic, read, sha

KEYS = ['dice', 'precision', 'recall', 'hd95_nm', 'nsd_16nm', 'predicted_foreground_fraction']

def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None

def stats(v):
    assert len(v) == 3
    return {'mean': mean(v), 'sd': statistics.stdev(v) if all(x is not None and math.isfinite(x) for x in v) else None,
            'minimum': min(x for x in v if x is not None) if any(x is not None for x in v) else None,
            'maximum': max(x for x in v if x is not None) if any(x is not None for x in v) else None,
            'finite_seeds': sum(x is not None and math.isfinite(x) for x in v)}

def csvout(name, rows):
    with (R / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)

def main():
    protocol = read(R / 'protocol.json')
    regions, sources, datasets, macros, hashes, images = [], [], [], [], [], []
    for model, spec in protocol['models'].items():
        for seed in protocol['seeds']:
            old = seed == protocol['existing_seed']
            folder = Path(spec['existing_head_dir']) if old else R / 'downstream/runs' / model / f'seed{seed}'
            complete = read(folder / 'training_complete.json')
            assert complete['completed_steps'] == 40000 and complete['encoder_unchanged']
            rows = [read(p) for p in folder.glob('*_metrics.json')]
            assert len(rows) == 38 and len({v['region_id'] for v in rows}) == 38
            for row in rows:
                assert row['encoder_sha256'] == spec['encoder_sha256']
                row.update(model=model, seed=seed)
            regions += rows
            ss = [{'model': model, 'seed': seed, 'source_id': s,
                   **{k: mean([r[k] for r in rows if r['source_id'] == s]) for k in KEYS}}
                  for s in sorted({r['source_id'] for r in rows})]
            assert len(ss) == 12
            sources += ss
            bench = Path(spec['existing_benchmark_metrics']) if old else R / 'benchmark' / model / f'seed{seed}/metrics.json'
            b = read(bench)
            assert b['status'] == 'complete' and len(b['images']) == 2595 and len(b['datasets']) == 7
            ds = [{**v, 'model': model, 'seed': seed} for v in b['datasets']]
            datasets += ds
            images += [{**v, 'model': model, 'seed': seed} for v in b['images']]
            for cohort, vv in [('MV3', ss), ('MB1', ds)]:
                macros.append({'model': model, 'seed': seed, 'cohort': cohort, **{k: mean([v[k] for v in vv]) for k in KEYS}})
            path = folder / 'head_step040000.pth'
            digest = sha(path)
            assert digest == read(path.with_suffix('.json'))['sha256']
            hashes.append({'model': model, 'seed': seed, 'encoder_path': spec['encoder_path'],
                           'encoder_sha256': spec['encoder_sha256'], 'head_path': str(path), 'head_sha256': digest,
                           'decoder_initial_sha256': complete['decoder_initial_sha256'], 'sampling_hash': complete['sampling_hash']})
    assert len(regions) == 456
    for seed in protocol['seeds']:
        rows = [v for v in hashes if v['seed'] == seed]
        assert len({v['decoder_initial_sha256'] for v in rows}) == 1
        assert len({v['sampling_hash'] for v in rows}) == 1
    summary = []
    for model in protocol['models']:
        for cohort in ['MV3', 'MB1']:
            vv = [v for v in macros if v['model'] == model and v['cohort'] == cohort]
            item = {'model': model, 'cohort': cohort, 'seeds': 3}
            for k in KEYS:
                item.update({k + '_' + name: value for name, value in stats([v[k] for v in vv]).items()})
            summary.append(item)
    pairs = []
    lookup = {(v['model'], v['seed'], v['cohort']): v for v in macros}
    for label, a, b in [('D2-D1_30k', 'D1-30k', 'D2-30k'), ('D2-D1_100k', 'D1-100k', 'D2-100k'),
                         ('100k-30k_D1', 'D1-30k', 'D1-100k'), ('100k-30k_D2', 'D2-30k', 'D2-100k')]:
        for cohort in ['MV3', 'MB1']:
            for seed in protocol['seeds']:
                left, right = lookup[a, seed, cohort], lookup[b, seed, cohort]
                pairs.append({'comparison': label, 'cohort': cohort, 'seed': seed,
                              **{k: right[k] - left[k] if right[k] is not None and left[k] is not None else None for k in KEYS}})
    pairsummary = []
    for label in sorted({v['comparison'] for v in pairs}):
        for cohort in ['MV3', 'MB1']:
            vv = [v for v in pairs if v['comparison'] == label and v['cohort'] == cohort]
            item = {'comparison': label, 'cohort': cohort, 'positive_dice_seeds': sum(v['dice'] > 0 for v in vv)}
            for k in KEYS:
                item.update({k + '_' + name: value for name, value in stats([v[k] for v in vv]).items()})
            pairsummary.append(item)
    for name, rows in [('per_region', regions), ('per_source_seed', sources), ('MB1_per_dataset_seed', datasets),
                       ('MB1_per_image', images), ('seed_macros', macros), ('summary', summary),
                       ('paired_seed_deltas', pairs), ('paired_summary', pairsummary), ('checkpoint_hashes', hashes)]:
        csvout('D12_three_seed_' + name + '.csv', rows)
    # Per-source/dataset three-seed summaries preserve weak-source changes, including muscle.
    for name, table, group in [('sources', sources, 'source_id'), ('datasets', datasets, 'dataset')]:
        rows = []
        for model in protocol['models']:
            for value in sorted({v[group] for v in table}):
                vv = [v for v in table if v['model'] == model and v[group] == value]
                item = {'model': model, group: value}
                for k in KEYS:
                    item.update({k+'_'+n: x for n,x in stats([v[k] for v in vv]).items()})
                rows.append(item)
        csvout('D12_three_seed_' + name + '_summary.csv', rows)
    atomic(R / 'D12_three_seed_summary.json', {'status': 'complete', 'seeds': protocol['seeds'], 'summary': summary,
           'paired_summary': pairsummary, 'new_SSL_training': False, 'further_experiments_started': False})
    lines = ['# D1/D2 30k/100k：三个 decoder seeds', '',
             '固定 MV-Generalist-v2：156 train / 38 dev，原版冻结 Primus，各40k，TTA OFF。'
             '复用20261014，新增20261015/20261016；四个encoder没有重新预训练。', '',
             '| Encoder | 评测 | Dice 均值 ± 样本标准差 |', '|---|---|---:|']
    for v in summary:
        lines.append(f"| {v['model']} | {v['cohort']} | {v['dice_mean']:.6f} ± {v['dice_sd']:.6f} |")
    lines += ['', '| 配对比较 | 评测 | Dice变化均值 ± 标准差 | 正向seeds |', '|---|---|---:|---:|']
    for v in pairsummary:
        lines.append(f"| {v['comparison']} | {v['cohort']} | {v['dice_mean']:+.6f} ± {v['dice_sd']:.6f} | {v['positive_dice_seeds']}/3 |")
    lines += ['', 'MV3先在各来源内平均区域，再12来源等权；MB1先在组内平均逐图/逐层指标，再7组等权。'
              '表中标准差为三个decoder seeds的样本标准差，配对差值按相同seed计算。'
              '逐来源、逐benchmark、区域、逐图、失败空预测及非有限距离均保留CSV。', '',
              '这是固定划分的随机seed重复，不是三折交叉验证，也没有估计SSL训练seed或独立采集体的不确定性。'
              'MB1是已用于开发决策的诊断集。不能凭这三个seeds声称跨独立采集体统计显著。', '',
              '评分继续调用原validation.distances和MONAI；本轮CPU exact EDT避免抢占现有SSL显存，'
              '历史指标原样复用。所有checkpoint SHA见checkpoint_hashes.csv。完成后停止，不自动新实验。']
    (R / 'D12_three_seed_report.md').write_text('\n'.join(lines) + '\n')

if __name__ == '__main__':
    main()
