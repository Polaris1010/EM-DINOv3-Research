"""Independently verify completed counts/aggregation and render the full three-seed report.

CPU only. Reads committed result tables, never opens images or checkpoints and starts no jobs.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'experiments/D12-three-seeds'
MODELS = ['D1-30k', 'D2-30k', 'D1-100k', 'D2-100k']
SEEDS = [20261014, 20261015, 20261016]
KEYS = ['dice', 'precision', 'recall', 'hd95_nm', 'nsd_16nm', 'predicted_foreground_fraction']

def readcsv(name):
    with (FOLDER / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def number(v):
    return None if v in ['', None, 'None', 'null'] else float(v)

def average(values):
    values = [v for v in values if v is not None]
    return st.mean(values) if values else None

def sd(values):
    return st.stdev(values) if len(values) > 1 and all(v is not None and math.isfinite(v) for v in values) else None

def equal(a, b):
    if a is None or b is None:
        return a is b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b or math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)

def fmt(v, digits=4):
    if v is None or math.isnan(v):
        return '未定义'
    if math.isinf(v):
        return '∞' if v > 0 else '−∞'
    return f'{v:.{digits}f}'

def pm(a, b, digits=4):
    return fmt(a, digits) + ' ± ' + fmt(b, digits)

def csvout(name, rows):
    with (FOLDER / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    regions = readcsv('D12_three_seed_per_region.csv')
    datasets = readcsv('D12_three_seed_MB1_per_dataset_seed.csv')
    summary = readcsv('D12_three_seed_summary.csv')
    hashes = readcsv('D12_three_seed_checkpoint_hashes.csv')
    pairsummary = readcsv('D12_three_seed_paired_summary.csv')
    sources = readcsv('D12_three_seed_sources_summary.csv')
    status = json.loads((FOLDER / 'STATUS.json').read_text(encoding='utf-8'))
    assert status['status'] == 'complete' and not status['further_experiments_started']
    assert len(regions) == 456 and len(datasets) == 84 and len(hashes) == 12
    gt = {}
    macros, ious, imagefailures, datasetfailures = {}, [], [], []
    for model in MODELS:
        for seed in SEEDS:
            vv = [r for r in regions if r['model'] == model and int(r['seed']) == seed]
            assert len(vv) == 38 and len({v['region_id'] for v in vv}) == 38
            assert len({v['source_id'] for v in vv}) == 12
            for row in vv:
                tp, fp, fn, tn = (int(row[k]) for k in ['tp', 'fp', 'fn', 'tn'])
                assert equal(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1., float(row['dice']))
                key = row['region_id']
                identity = (tp+fn, tp+fp+fn+tn, row['manifest_sha256'])
                assert gt.setdefault(('MV3', key), identity) == identity
                row['iou'] = tp/(tp+fp+fn) if tp+fp+fn else 1.
                assert row['tta'] == 'False'
            for metric in [*KEYS, 'iou']:
                by_source = [average([number(v[metric]) for v in vv if v['source_id'] == s])
                             for s in sorted({v['source_id'] for v in vv})]
                macros[model, seed, 'MV3', metric] = average(by_source)
            images = readcsv(f'D12_three_seed_MB1_per_image_{model}_seed{seed}.csv')
            assert len(images) == 2595
            assert len({(v['dataset_id'], v['image_id'], v['z']) for v in images}) == 2595
            for row in images:
                tp, fp, fn, tn = (int(row[k]) for k in ['tp', 'fp', 'fn', 'tn'])
                identity = (tp+fn, tp+fp+fn+tn)
                key = (row['dataset_id'], row['image_id'], row['z'])
                assert gt.setdefault(('MB1', key), identity) == identity
                assert equal(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1., float(row['dice']))
                assert equal(tp/(tp+fp+fn) if tp+fp+fn else 1., float(row['iou']))
                if int(row['dataset_id']) == 7:
                    assert number(row['hd95_nm']) is number(row['nsd_16nm']) is None
            ds = [v for v in datasets if v['model'] == model and int(v['seed']) == seed]
            assert len(ds) == 7
            for metric in [*KEYS, 'iou']:
                reconstructed = []
                for d in ds:
                    entries = [v for v in images if v['dataset_id'] == d['dataset_id']]
                    value = average([number(v[metric]) for v in entries])
                    assert equal(value, number(d[metric])), (model, seed, d['dataset'], metric)
                    reconstructed.append(value)
                macros[model, seed, 'MB1', metric] = average(reconstructed)
            for d in ds:
                entries = [v for v in images if v['dataset_id'] == d['dataset_id']]
                datasetfailures.append({'model': model, 'seed': seed, 'dataset': d['dataset'],
                    'images': len(entries), 'empty_prediction_images': sum(int(v['tp'])+int(v['fp']) == 0 for v in entries),
                    'empty_GT_images': sum(int(v['tp'])+int(v['fn']) == 0 for v in entries),
                    'HD95_infinite_images': sum(number(v['hd95_nm']) is not None and math.isinf(number(v['hd95_nm'])) for v in entries),
                    'physical_spacing_unavailable_images': sum(number(v['hd95_nm']) is None for v in entries)})
            for cohort in ['MV3', 'MB1']:
                ious.append({'model': model, 'seed': seed, 'cohort': cohort, 'iou': macros[model, seed, cohort, 'iou']})
    assert len(gt) == 38+2595
    for seed in SEEDS:
        vv = [v for v in hashes if int(v['seed']) == seed]
        assert len(vv) == 4
        assert len({v['decoder_initial_sha256'] for v in vv}) == len({v['sampling_hash'] for v in vv}) == 1
    assert len({v['encoder_sha256'] for v in hashes}) == 4
    assert len({v['head_sha256'] for v in hashes}) == 12
    for row in summary:
        for key in KEYS:
            values = [macros[row['model'], seed, row['cohort'], key] for seed in SEEDS]
            assert equal(average(values), number(row[key+'_mean'])), (row['model'], row['cohort'], key)
            assert equal(sd(values), number(row[key+'_sd']))
    iou_summary = [{'model': model, 'cohort': cohort,
                    'iou_mean': average([macros[model,s,cohort,'iou'] for s in SEEDS]),
                    'iou_sd': sd([macros[model,s,cohort,'iou'] for s in SEEDS])}
                   for model in MODELS for cohort in ['MV3', 'MB1']]
    csvout('D12_three_seed_IoU_per_seed.csv', ious)
    csvout('D12_three_seed_IoU_summary.csv', iou_summary)
    csvout('D12_three_seed_MB1_failure_counts.csv', datasetfailures)
    for model in MODELS:
        vv = [v for v in datasetfailures if v['model'] == model]
        imagefailures.append({'model':model, **{key:sum(v[key] for v in vv) for key in
            ['images','empty_prediction_images','empty_GT_images','HD95_infinite_images','physical_spacing_unavailable_images']}})
    csvout('D12_three_seed_MB1_failure_totals.csv', imagefailures)
    runtime = []
    for model in MODELS:
        for seed in SEEDS[1:]:
            d = FOLDER / 'downstream/runs' / model / f'seed{seed}'
            train = json.loads((d/'training_complete.json').read_text(encoding='utf-8'))
            infer = json.loads((d/'inference_complete.json').read_text(encoding='utf-8'))
            assert train['completed_steps'] == 40000 and train['encoder_unchanged'] and infer['regions'] == 38
            runtime.append({'model':model,'seed':seed,'head_seconds':train['seconds'],'development_inference_seconds':infer['seconds']})
    csvout('D12_three_seed_new_run_times.csv', runtime)
    lookup = {(v['model'], v['source_id']): v for v in sources}
    ds_summary = readcsv('D12_three_seed_datasets_summary.csv')
    dl = {(v['model'], v['dataset']): v for v in ds_summary}
    source_deltas = []
    per_source = readcsv('D12_three_seed_per_source_seed.csv')
    sl = {(v['model'], int(v['seed']), v['source_id']): v for v in per_source}
    for source in sorted({v['source_id'] for v in sources}):
        a, b = lookup['D1-100k',source], lookup['D2-100k',source]
        delta = [float(sl['D2-100k',seed,source]['dice'])-float(sl['D1-100k',seed,source]['dice']) for seed in SEEDS]
        source_deltas.append({'source':source, 'D1_100k_dice_mean':float(a['dice_mean']),
            'D1_100k_dice_sd':float(a['dice_sd']), 'D2_100k_dice_mean':float(b['dice_mean']),
            'D2_100k_dice_sd':float(b['dice_sd']), 'paired_Dice_delta_mean':average(delta),
            'paired_Dice_delta_sd':sd(delta), 'positive_seeds':sum(v>0 for v in delta),
            'NSD_delta_mean':float(b['nsd_16nm_mean'])-float(a['nsd_16nm_mean'])})
    assert all(next(v for v in source_deltas if v['source']==s)['positive_seeds']==3 for s in ['guay21','muller21'])
    muscle = {(v['model'],int(v['seed'])):float(v['dice']) for v in datasets if v['dataset']=='glycolytic_muscle'}
    assert all(muscle['D2-100k',s] < muscle['D2-30k',s] for s in SEEDS)
    csvout('D12_three_seed_100k_source_paired.csv', source_deltas)
    auto = FOLDER / 'D12_three_seed_generated_report.md'
    report = ROOT / 'docs/D12_three_seed_report.md'
    if not auto.exists():
        auto.write_bytes(report.read_bytes())
    final = json.loads((ROOT/'evidence/D12-three-seeds/final_verification_20261022.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(auto.read_bytes()).hexdigest() == final['server_report_sha256']
    lines = ['# D1/D2 30k与100k：三个decoder seed完整报告', '',
        '**本轮已完成并停止。** 新增8个head各40k，8个Wei评分全部完成；合并历史seed后共有12个run、456条开发区域指标、31,140条MB1逐图指标。2026-10-22香港09:20自动收取推送；本报告补齐完整指标、独立数值复核及研究结论。', '',
        '## 数据、模型与统计口径', '',
        'D1为MitoVerse来源均衡SSL，D2为MitoVerse/CEM约50:50混合SSL（CEM集合sqrt加权、有放回抽样）。既有SSL池为153个MitoVerse体积、34,143个合法Z；D2另有1,425,433张CEM图像/627个集合。四个已完成encoder固定取D1/D2的30k、100k EMA；100k沿用历史30k后续训70k的分段LR日程。监督均为MV-Generalist-v2：156 train / 38 dev、12来源，原版Primus_Multiscale、ViT-L/16全程冻结、层[5,11,17,23]、40k共同预算及固定40k选模、TTA OFF。Han整幅GT规则沿用。', '',
        '复用decoder seed20261014，新增20261015/16；每个seed的四个encoder使用一致head初始化SHA与采样hash。这里测量decoder初始化和监督采样的波动；SSL本身未重复训练，数据划分固定。三个decoder seeds不增加独立acquisition数。', '',
        '开发集Dice/IoU先汇总每区域有效体素，再区域内指标按来源平均，最后12来源等权；HD95/NSD为既有完整3D区域物理指标。MB1逐层二维推理，逐图指标先在各benchmark内平均，再7组等权。标准差为三个seed样本SD（ddof=1）；差值按同seed配对计算。', '',
        '**MB1物理指标仅覆盖6个已标定体数据组。** TEM-100无可靠spacing，HD95/NSD保持NA，未伪造nm指标；这些物理指标的宏均值为6组等权。Precision/Recall的空分母按原evaluator保留NA。MB1已有开发暴露，继续作为固定诊断。', '',
        '## 完整总体指标', '']
    for cohort, title in [('MV3','MV3开发集：12来源等权'), ('MB1','MB1：7组等权；物理指标6组')]:
        lines += [f'### {title}', '', '| Encoder | Dice | IoU | Precision | Recall | HD95 nm | NSD@16nm | 前景占比% |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for model in MODELS:
            row = next(v for v in summary if v['model']==model and v['cohort']==cohort)
            io = next(v for v in iou_summary if v['model']==model and v['cohort']==cohort)
            cells = [pm(number(row[k+'_mean']), number(row[k+'_sd']),1 if k=='hd95_nm' else 4)
                     for k in ['dice']]
            cells += [pm(io['iou_mean'],io['iou_sd'])]
            cells += [pm(number(row[k+'_mean']),number(row[k+'_sd']),1 if k=='hd95_nm' else 4)
                      for k in ['precision','recall','hd95_nm','nsd_16nm']]
            cells += [pm(100*float(row['predicted_foreground_fraction_mean']),100*float(row['predicted_foreground_fraction_sd']),2)]
            lines.append('| '+model+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['MB1有空GT或空预测的单边前景失败图，因此各模型HD95宏均值均为∞，样本SD及∞−∞配对差值未定义；保留这些失败，没有删除后另报有限均值。逐benchmark/逐图结果与空预测、空GT、∞及spacing缺失数量见失败计数CSV。', '',
              '| Encoder | 三seed总图数 | 空预测图数 | HD95=∞图数 | 缺spacing图数 |', '|---|---:|---:|---:|---:|']
    for row in imagefailures:
        lines.append('| '+row['model']+' | '+' | '.join(str(row[k]) for k in ['images','empty_prediction_images','HD95_infinite_images','physical_spacing_unavailable_images'])+' |')
    lines += ['',
              '## 同seed配对结论', '', '| 比较 | 评测 | Dice变化均值±SD | Dice正向seeds | NSD变化均值±SD | HD95变化nm均值±SD |', '|---|---|---:|---:|---:|---:|']
    for row in pairsummary:
        lines.append('| '+row['comparison']+' | '+row['cohort']+' | '+
            ' | '.join([pm(number(row['dice_mean']),number(row['dice_sd']),6),row['positive_dice_seeds']+'/3',
                       pm(number(row['nsd_16nm_mean']),number(row['nsd_16nm_sd']),6),
                       pm(number(row['hd95_nm_mean']),number(row['hd95_nm_sd']),2)])+' |')
    lines += ['', '## 100k逐来源比较', '', '| 来源 | D1 Dice±SD | D2 Dice±SD | 配对Dice变化±SD | 正向seeds | NSD均值变化 |', '|---|---:|---:|---:|---:|---:|']
    for row in source_deltas:
        lines.append('| '+row['source']+' | '+pm(row['D1_100k_dice_mean'],row['D1_100k_dice_sd'])+' | '+
                     pm(row['D2_100k_dice_mean'],row['D2_100k_dice_sd'])+' | '+pm(row['paired_Dice_delta_mean'],row['paired_Dice_delta_sd'],6)+' | '+str(row['positive_seeds'])+'/3 | '+fmt(row['NSD_delta_mean'],6)+' |')
    lines += ['', '逐来源全部Precision/Recall/HD95/NSD/前景占比均值与SD保存在sources_summary.csv；30k与100k均保留，没有只展示有利来源。', '',
              '## MB1逐benchmark Dice', '', '| Benchmark | D1-30k | D2-30k | D1-100k | D2-100k | D2−D1@100k |', '|---|---:|---:|---:|---:|---:|']
    for dataset in sorted({v['dataset'] for v in ds_summary}):
        vals = [pm(float(dl[m,dataset]['dice_mean']),float(dl[m,dataset]['dice_sd'])) for m in MODELS]
        delta = float(dl['D2-100k',dataset]['dice_mean'])-float(dl['D1-100k',dataset]['dice_mean'])
        lines.append('| '+dataset+' | '+' | '.join(vals)+' | '+fmt(delta,6)+' |')
    lines += ['', '六个volume的全体素semantic IoU另列于MB1_per_dataset_seed.csv的volumetric_semantic_iou，仍来自逐层二维推理，不等同于作者完整3D实例流程。各组IoU、Precision、Recall、HD95、NSD及前景占比保存在datasets_summary及原始逐组表。', '',
        '## 研究结论与本轮决定', '',
        '1. **D2的内部开发收益稳定。** 100k时D2−D1 Dice为+0.021238±0.001069，三个seed均正；NSD为+0.023787，三个seed均正。Guay和Müller的100k Dice也均为三seed正向。',
        '2. **这项内部收益没有转化为MB1总体Dice优势。** 100k时D2−D1为−0.013398±0.008921，三个seed均负。D2的MB1 NSD较高（六标定组+0.018951），Recall更高但Precision更低、预测前景更多，说明不同指标反映出不同取舍，不能称全部指标一致退化或改善。',
        '3. **D2续训到100k确有部分稳定收益。** 相对30k，开发Dice+0.010705、MB1 Dice+0.016740，三个seed均正；开发NSD也提高。但开发HD95从497.72升到546.45nm，配对+48.73nm且三个seed均升，远距离误差仍是短板。',
        '4. **肌肉退化并非某次head初始化偶然。** Glycolytic muscle的D2 Dice三seed均值从0.443746降到0.196840，D1则从0.577401升到0.668534。完整系统在该域的假阳性问题仍然成立；沿用既有Glycolytic muscle复核，不改阈值或尺度。',
        '5. **修正此前D1单seed解读。** D1续训的MB1均值变化为+0.004030±0.012049，只有2/3 seed正向。此前seed14下降−0.009803不能作为稳定续训退化结论。',
        '6. **本轮收尾。** 保留D2-100k作为当前MitoVerse内部较优的固定冻结readout，保留D1作为MB1总体Dice更高的匹配对照。三seed确认主要差异方向，同时揭示单seed对D1续训结论的敏感性。它们没有检验新结构loss；本轮结束，不自动启动D3、延长SSL或新增搜索。', '',
        '## 可复现性、加速与停止状态', '',
        'checkpoint_hashes.csv记录四个encoder和12个head的完整路径/SHA256、head初始化SHA、训练采样hash。自动生成报告完成前已逐文件重新验证head SHA；本轮收尾不重复读取大checkpoint。源码保留初始训练86492fc、GPU接续682574d、连续块全轴EDT4281f22及各作业源码SHA。', '',
        '评分前段采用CPU exact EDT，后段采用既有float64 CUDA EDT；最大Wei距离场放主存，按连续块传输并在CUDA上转置，每次变换覆盖完整空间轴。CPU/GPU完整距离场和MONAI校验通过；没有局部空间截断、距离近似或参数调优。277条和296条已完成指标在两次调度切换后SHA保持不变。', '',
        '最终服务d12-three-seeds-gpu-score-20261022.service：inactive、MainPID=0、退出码0，新增8个Wei均有完整HD95/NSD及两次host-backed EDT记录；本轮训练/推理/评分无残留进程。原d12-three-seeds服务在运维切换时以非零退出留档，已由成功完成的接续服务取代。NR监督器已恢复；独立已授权Coverage-NR服务沿原任务继续，未由本轮新启动或改动训练。网站HTTP200，GPU0未用于本轮。', '',
        '### 本轮实际耗时', '',
        '以下为新增作业自身记录时长，包含既有数据读取、计算和固定monitor；三个GPU与Coverage-NR共享，因此不是独占GPU速度基准。seed14未重新训练。', '',
        '| Encoder | seed | head训练分钟 | 完整开发推理分钟 |', '|---|---:|---:|---:|']
    for row in runtime:
        lines.append(f"| {row['model']} | {row['seed']} | {row['head_seconds']/60:.1f} | {row['development_inference_seconds']/60:.1f} |")
    lines += ['',
        '按服务器现场时间记录，香港03:39所有head完成、06:14开发预测完成、07:32全部MB1完成；09:20自动收取推送。先前约1小时仅为剩余head估计，没有给出完整推理/评测总耗时。Wei的完整评分每次约164–1487秒，包含CPU表面提取、主存传输和GPU EDT等；Han约22秒不能外推为Wei总耗时。原始stage/作业时间见现场JSON与new_run_times.csv。', '',
        '## 文件与核验', '',
        '- 完整总体/来源/benchmark汇总：experiments/D12-three-seeds/D12_three_seed_{summary,sources_summary,datasets_summary}.csv。',
        '- 原始456条开发区域、144条来源×seed、84条benchmark×seed、12份无损MB1逐图分片（31,140行）均在同目录。',
        '- 新增IoU汇总与逐seed、MB1失败计数、100k逐来源配对表；历史原指标表保持原样。',
        '- 最终停止/Wei/保留文件SHA现场证据：evidence/D12-three-seeds/final_verification_20261022.json。',
        '- 独立CPU复核脚本：scripts/finalize_d12_seed_report.py。按原始TP/FP/FN重算Dice/IoU，核实所有seed使用相同GT体素与图像集合，重建两级等权均值和样本SD并与自动汇总逐项一致。',
        '- 自动生成报告原件保存在experiments/D12-three-seeds/D12_three_seed_generated_report.md；本文件为基于同一CSV的完整叙述版。代码、报告、指标与每轮Markdown进入私人仓库draft PR12，不自动合并。', '']
    report.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    verification = {'status':'PASS','region_rows':len(regions),'MB1_image_rows':12*2595,'dataset_seed_rows':len(datasets),
        'matched_seeds':SEEDS,'checkpoint_rows':len(hashes),'common_GT_image_ids':2595,'common_GT_region_ids':38,
        'independent_aggregation_matches':True,'new_GPU_work':False,'new_training':False,
        'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),
        'auto_report_sha256':hashlib.sha256(auto.read_bytes()).hexdigest()}
    (ROOT/'evidence/D12-three-seeds/final_report_validation.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(verification))

if __name__ == '__main__':
    main()
