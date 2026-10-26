"""Aggregate completed tables only; no images, model loading, inference or distance scoring."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'experiments/Coverage-NR'
KEYS = ['dice', 'iou', 'precision', 'recall', 'hd95_nm', 'nsd_16nm', 'predicted_foreground_fraction']


def table(name):
    with (E / name).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def number(value):
    return None if value in (None, '') else float(value)


def mean(values):
    values = [number(v) for v in values if number(v) is not None]
    return sum(values) / len(values) if values else None


def same(a, b):
    a, b = number(a), number(b)
    if a is None or b is None:
        assert a == b, (a, b)
    elif math.isnan(a) or math.isnan(b):
        assert math.isnan(a) and math.isnan(b), (a, b)
    else:
        assert a == b or math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10), (a, b)


def fmt(value, digits=4):
    if value is None or math.isnan(value):
        return '未定义'
    if math.isinf(value):
        return '∞' if value > 0 else '−∞'
    return f'{value:.{digits}f}'


def csvout(name, rows):
    with (E / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    summary = json.loads((E / 'Coverage_NR_summary.json').read_text())
    regions = table('Coverage_NR_per_region_metrics.csv')
    sources = table('Coverage_NR_per_source_metrics.csv')
    benchmark = json.loads((E / 'MB1_metrics.json').read_text())
    images, datasets = benchmark['images'], benchmark['datasets']
    ref_regions = table('reference_D2_100k_seed20261014_per_region.csv')
    ref_images = table('reference_D2_100k_seed20261014_MB1_per_image.csv')
    ref_sources = table('reference_D12_three_seed_per_source_seed.csv')
    ref_datasets = table('reference_D12_three_seed_MB1_per_dataset_seed.csv')
    macros = table('reference_D12_three_seed_seed_macros.csv')
    head = json.loads((E / 'head_training_complete.json').read_text())
    reference_head = table('reference_D12_three_seed_checkpoint_hashes.csv')[0]
    assert len(regions) == len(ref_regions) == 38
    assert len(images) == len(ref_images) == 2595
    assert len(sources) == len(ref_sources) == 12
    assert len(datasets) == len(ref_datasets) == 7
    assert head['decoder_initial_sha256'] == reference_head['decoder_initial_sha256']
    assert head['sampling_hash'] == reference_head['sampling_hash']
    assert head['completed_steps'] == 40000 and head['encoder_unchanged']
    assert head['seed'] == 20261014 and head['tta'] is False
    assert all(row['distance_backend'] == 'MV2_exact_GPU_EDT_MONAI' for row in regions)
    assert any(int(row['host_backed_EDT_calls']) > 0 for row in regions)
    assert benchmark['distance_backend'] == 'MV2_exact_GPU_EDT_MONAI'
    assert benchmark['full_device_EDT_calls'] > 0

    def image_key(row):
        return row['dataset'], str(row.get('image_id', '')), int(row['z'])

    for rows, refs, key in [(regions, ref_regions, lambda r: r['region_id']),
                            (images, ref_images, image_key)]:
        reference = {key(row): row for row in refs}
        assert len(reference) == len(rows) and set(reference) == {key(row) for row in rows}
        for row in rows:
            tp, fp, fn, tn = [int(row[k]) for k in ['tp', 'fp', 'fn', 'tn']]
            same(row['dice'], 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)
            other = reference[key(row)]
            assert tp+fn == int(other['tp'])+int(other['fn'])
            assert fp+tn == int(other['fp'])+int(other['tn'])

    for row in regions + ref_regions:
        tp, fp, fn = [int(row[k]) for k in ['tp', 'fp', 'fn']]
        row['iou'] = tp/(tp+fp+fn) if tp+fp+fn else 1.
    for current, raw in [(sources, regions), (ref_sources, ref_regions)]:
        for row in current:
            row['iou'] = mean(v['iou'] for v in raw if v['source_id'] == row['source_id'])
    summary['dev_macro']['iou'] = mean(v['iou'] for v in sources)
    for macro in macros:
        macro['iou'] = mean(v['iou'] for v in (ref_sources if macro['cohort']=='MV3' else ref_datasets))

    for row in sources:
        for metric in KEYS:
            same(row[metric], mean(v[metric] for v in regions if v['source_id'] == row['source_id']))
    for metric in KEYS:
        same(summary['dev_macro'][metric], mean(v[metric] for v in sources))
    for row in datasets:
        members = [v for v in images if v['dataset'] == row['dataset']]
        assert len(members) == row['planes']
        for metric in KEYS:
            same(row[metric], mean(v[metric] for v in members))
    same(summary['MB1_macro_dice'], mean(v['dice'] for v in datasets))
    assert all(v['hd95_nm'] is None and v['nsd_16nm'] is None for v in images if v['dataset'] == 'TEM-100')
    assert summary['coverage']['unique_cem_ids'] == 1425433 and summary['coverage']['coverage'] == 1.

    def compare(current, reference, key):
        lookup = {v[key]: v for v in reference}
        rows = []
        for row in current:
            out = {key: row[key]}
            for metric in KEYS:
                a, b = number(row[metric]), number(lookup[row[key]][metric])
                out.update({f'NR_{metric}': a, f'D2_{metric}': b,
                            f'delta_{metric}': a-b if a is not None and b is not None else None})
            rows.append(out)
        return rows

    source_delta = compare(sources, ref_sources, 'source_id')
    dataset_delta = compare(datasets, ref_datasets, 'dataset')
    csvout('Coverage_NR_vs_D2_seed20261014_sources_full.csv', source_delta)
    csvout('Coverage_NR_vs_D2_seed20261014_datasets_full.csv', dataset_delta)
    nr_macros = {'MV3': summary['dev_macro'], 'MB1': {k: mean(v[k] for v in datasets) for k in KEYS}}
    ref_macros = {v['cohort']: v for v in macros}
    overall = []
    for cohort, values in nr_macros.items():
        for metric in KEYS:
            a, b = values[metric], number(ref_macros[cohort][metric])
            overall.append({'cohort': cohort, 'metric': metric, 'Coverage_NR': a,
                            'D2_100k_seed20261014': b, 'delta': a-b})
    csvout('Coverage_NR_vs_D2_seed20261014_overall.csv', overall)
    failure_counts = []
    for dataset in datasets:
        rows = [v for v in images if v['dataset'] == dataset['dataset']]
        failure_counts.append({'dataset': dataset['dataset'], 'images': len(rows),
            'empty_GT': sum(v['tp']+v['fn'] == 0 for v in rows),
            'empty_prediction': sum(v['tp']+v['fp'] == 0 for v in rows),
            'HD95_infinite': sum(v['hd95_nm'] is not None and math.isinf(v['hd95_nm']) for v in rows),
            'physical_spacing_missing': sum(v['hd95_nm'] is None for v in rows)})
    csvout('Coverage_NR_MB1_failure_counts.csv', failure_counts)
    proof = json.loads((ROOT / 'evidence/Coverage-NR/final_status_20261022T151108Z.json').read_text())
    assert proof['status']['status'] == 'complete' and 'ActiveState=inactive' in proof['service']
    assert 'MainPID=0' in proof['service'] and 'ExecMainStatus=0' in proof['service']
    assert proof['status']['historical_inputs_unchanged'] and proof['website_http'] == 200
    lines = ['# Coverage-NR：100k完成报告与D2-100k对照', '',
        '截至香港2026-10-22 23:11现场核查，SSL100k、冻结Primus40k、38开发区域和2595张MB1图像均完成。最终服务inactive、MainPID0、退出0；任务停止，无新增实验。', '',
        '## 协议与覆盖率', '',
        'Coverage-NR从官方P0初始化，固定MitoVerse/CEM池、约50:50域配比、global48、DINO+iBOT+0.1KoLeo、连续100k LR。已消费CEM抽样2,398,767次、MitoVerse2,401,233次，保留CEM的1,425,433个图像ID全部覆盖（100%）；历史D2为955,756/1,425,433（67.05%）。', '',
        '固定MV-Generalist-v2的156train/38dev，原版Primus_Multiscale、冻结ViT-L/16、层[5,11,17,23]、decoder seed20261014、40k选模、TTA OFF及原输入/阈值。与D2相同head初始化SHA和监督sampling hash已核对。Han沿用用户指定整幅GT规则。', '',
        '**主对照使用D2-100k同一decoder seed20261014。** 历史D2有不同collection分布和分段30k+70k LR，Coverage-NR使用连续100k LR；这是完整系统参考对照，差值不能单独归于覆盖率。NR只有一个decoder seed；D2三seed均值仅作背景。MB1已参与开发决策。', '',
        '## 总体指标', '', '| 评测 | 指标 | Coverage-NR | D2-100k同seed | 差值NR−D2 |', '|---|---|---:|---:|---:|']
    for row in overall:
        lines.append('| '+row['cohort']+' | '+row['metric']+' | '+ ' | '.join(fmt(row[k], 2 if row['metric']=='hd95_nm' else 6) for k in ['Coverage_NR','D2_100k_seed20261014','delta'])+' |')
    lines += ['', '开发指标先按区域、再12来源等权。MB1先逐图、再7组等权；HD95/NSD仅对6个可靠spacing体数据组汇总，TEM100保持NA。MB1存在单边空前景失败，HD95宏均值为∞，∞−∞差值未定义，未删除这些图。预测前景占比以0–1比例表示。', '',
        'D2-100k三decoder seed背景：开发Dice0.839914±0.001345，MB1 Dice0.463563±0.010832（样本SD）。NR单seed亦低于这些均值，但未测量NR的seed波动。', '',
        '## 开发逐来源Dice', '', '| 来源 | NR | D2同seed | 差值 |', '|---|---:|---:|---:|']
    for row in source_delta:
        lines.append('| '+row['source_id']+' | '+' | '.join(fmt(row[k],6) for k in ['NR_dice','D2_dice','delta_dice'])+' |')
    lines += ['', '## MB1逐组Dice', '', '| Benchmark | NR | D2同seed | 差值 |', '|---|---:|---:|---:|']
    for row in dataset_delta:
        lines.append('| '+row['dataset']+' | '+' | '.join(fmt(row[k],6) for k in ['NR_dice','D2_dice','delta_dice'])+' |')
    lines += ['', '## 结论与停止决定', '',
        '1. 全图像ID覆盖目标已实现，但本轮没有带来总体Dice收益：开发0.831821对0.841379，MB1为0.440027对0.475754。',
        '2. 开发集差距主要来自OpenOrganelle（−0.111681）；其余来源有涨有跌。开发HD95下降约28.94nm，但NSD略降，不能将结果概括为所有指标均退化。',
        '3. Glycolytic muscle从D2同seed的0.215677改善到0.393368；其余六组MB1 Dice均下降，因此肌肉局部恢复没有转化为总体优势。',
        '4. 当前证据不支持把图像ID覆盖率当作主要性能瓶颈或继续自动投入覆盖率实验。保留NR为完成的探索性系统对照；本轮停止，下一科研动作等待用户决定。', '',
        '## GPU执行、恢复与耗时', '',
        '训练与推理使用物理GPU1/2/3；开发38区域均记录exact GPU EDT，最大Wei使用主存配合完整轴GPU分块。MB1记录4818次GPU EDT调用；有/无ignore的HD95/NSD与原实现一致。GPU0未用，网站HTTP200。CPU用于I/O、表面准备、调度及表格聚合。', '',
        'GPU合成校验曾误触旧controller未知PID保护，在94450步停止；从94000完整状态恢复student/EMA、optimizer、loss centers、三rank RNG和采样位置，重放450步。此错误、失败原件和恢复证据已保留。总模型预算仍为100k，未改LR或训练器。', '',
        'SSL中断前进度记录84238.7秒，接续6000步4437.1秒，记录段合计约24.63小时（含450步重放，不含恢复加载等额外墙钟开销）；head40k为49.61分钟，完整开发推理32.91分钟，评分与MB1另计。接续result中的4437秒不能冒充整轮SSL耗时。', '',
        '## 资产、核验与交付', '',
        '完整checkpoint路径/SHA在experiments/Coverage-NR/Coverage_NR_checkpoint_hashes.csv；100k EMA为a886d8b5447230b097bb2de06716d8f18ff51968b8837c122daf03e5a403293c，40k head为a9dc0b0ca064f5726696ced127e72b6707092f7a14e2426e6412ee2474ceba53。', '',
        '服务器原报告保留在experiments/Coverage-NR/Coverage_NR_final_report.md，原始38区域、12来源、2595逐图、7组表以及完整MB1 JSON均同目录；新增同seed完整逐来源/逐组/总体差值表与失败计数。最终状态证据见evidence/Coverage-NR/final_status_20261022T151108Z.json。', '',
        'scripts/finalize_coverage_nr_report.py只重算小型表格：核对Dice、相同GT体素、head初始化/采样hash、两级聚合和GPU记录；不读取图像/权重或重新评分。代码、指标、报告、交接和本轮Markdown提交到私人仓库draft PR11，不自动合并。', '']
    report = ROOT / 'docs/Coverage_NR_final_report.md'
    report.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    validation = {'status':'PASS','regions':38,'sources':12,'MB1_images':2595,'datasets':7,
        'same_GT_counts_as_D2':True,'same_head_initialization_and_sampling':True,'GPU_metrics_complete':True,
        'new_training':False,'new_inference_or_distance_scoring':False,
        'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest()}
    (ROOT/'evidence/Coverage-NR/final_report_validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
