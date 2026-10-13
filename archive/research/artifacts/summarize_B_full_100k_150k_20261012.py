"""Summarize matched frozen probes from downloaded, inspectable result files."""
from pathlib import Path
import csv,json,math,statistics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'mitoverse_B_full_extend150k_20261012'

def main():
    old=json.loads((ROOT/'mitoverse_boundary_metrics_20261012/results.json').read_text(encoding='utf-8'))
    new=json.loads((OUT/'comparison_100k_150k.json').read_text(encoding='utf-8'))
    assert new['status']=='complete'
    audit=json.loads((OUT/'100k_decoder_initialization_reconstruction.json').read_text(encoding='utf-8'))
    assert audit['status']=='pass'
    rows=[];paired=[]
    for ds in ['Dataset111_CMCC','Dataset131_CMCT']:
        baseline=next(r for r in old['summary'] if r['dataset']==ds and r['encoder_kind']=='B_full')
        extended=next(r for r in new['summary'] if r['dataset']==ds)
        assert baseline['n']==extended['n']
        name=ds.split('_')[-1]
        old_probe=json.loads((OUT/f'100k_{name}_metrics.json').read_text(encoding='utf-8'))
        new_probe=json.loads((OUT/f'150k_{name}_metrics.json').read_text(encoding='utf-8'))
        assert old_probe['status']==new_probe['status']=='complete'
        assert old_probe['split_sha256']==new_probe['split_sha256']
        assert old_probe['seed']==new_probe['seed']==20260907
        assert old_probe['steps']==new_probe['steps']==3000
        assert old_probe['encoder_step']==100000 and new_probe['encoder_step']==150000
        assert audit['original_probe_sha256']==old_probe['source_sha256']
        assert audit['decoder_initial_sha256']==new_probe['decoder_initial_sha256']
        cases={r['case']:r for r in old['per_volume'] if r['dataset']==ds and r['encoder_kind']=='B_full'}
        fresh={r['filename'].removesuffix('.nii.gz'):r for r in new['cases'] if r['dataset']==ds}
        assert cases.keys()==fresh.keys() and len(cases)==baseline['n']
        for step,r in [(100000,baseline),(150000,extended)]:
            rows.append({'dataset':ds,'step':step,'n':r['n'],**{f'{m}_{v}':r[f'{m}_{v}'] for m in ['dice','hd95','nsd'] for v in ['mean','std']}})
        for case in sorted(cases):
            row={'dataset':ds,'case':case}
            for m in ['dice','hd95','nsd']:
                a=cases[case][m];b=fresh[case][m+'_class1']
                assert math.isfinite(a) and math.isfinite(b)
                row.update({m+'_100k':a,m+'_150k':b,m+'_delta':b-a})
            paired.append(row)
        for m in ['dice','hd95','nsd']:
            vals=[fresh[c][m+'_class1'] for c in cases]
            assert math.isclose(statistics.mean(vals),extended[m+'_mean'],rel_tol=2e-7,abs_tol=1e-7)
            assert math.isclose(statistics.stdev(vals),extended[m+'_std'],rel_tol=2e-6,abs_tol=1e-7)
            assert math.isclose(extended[m+'_mean']-baseline[m+'_mean'],extended[m+'_150k_minus_100k'],abs_tol=1e-12)
    for name,data in [('summary_100k_150k.csv',rows),('paired_volume_changes.csv',paired)]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# B_full 100k 与 150k 冻结评测比较','','## 评测结果','','| 数据集 | SSL step | 验证体积 n | Dice ↑ | HD95 ↓ | NSD ↑ |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['dataset'].split('_')[-1]} | {r['step']} | {r['n']} | {r['dice_mean']:.6f} ± {r['dice_std']:.6f} | {r['hd95_mean']:.2f} ± {r['hd95_std']:.2f} | {r['nsd_mean']:.6f} ± {r['nsd_std']:.6f} |")
    lines+=['','## 均值变化（150k − 100k）','','| 数据集 | Dice | HD95 | NSD |','|---|---:|---:|---:|']
    for r in new['summary']:
        lines.append(f"| {r['dataset'].split('_')[-1]} | {r['dice_150k_minus_100k']:+.6f} | {r['hd95_150k_minus_100k']:+.2f} | {r['nsd_150k_minus_100k']:+.6f} |")
    lines+=['','## 逐体积变化','','| 数据集 | 体积 | Dice 变化 | HD95 变化 | NSD 变化 |','|---|---|---:|---:|---:|']
    for r in paired:
        lines.append(f"| {r['dataset'].split('_')[-1]} | {r['case']} | {r['dice_delta']:+.6f} | {r['hd95_delta']:+.2f} | {r['nsd_delta']:+.6f} |")
    lines+=['','## 协议与解释','','- Encoder 全部冻结；相同 decoder 初始状态、随机种子 20260907、fold-0 划分和 3000 步 decoder 训练预算。最终原始网格完整 3D 体积推理，沿用相同四翻转 TTA 和 decoder 选择规则。','- Dice/HD95/NSD 使用此前核验过的 MedDINOv3 官方测试入口。NSD 容差为 1.0，当前 spacing=(1,1,1)，距离数值按 voxel 等效单位解释；真实 EM 物理尺度未校准。','- 均值为体积等权平均；SD 为体积间样本标准差（ddof=1），不是随机种子标准差。CMCC n=3，CMCT n=2，单种子，不作统计显著性声明。','- 续训完整恢复 100k 状态，新增 50k 使用基础学习率 1e-6。此结果不能等同于从头采用 150k 余弦调度的实验。','- 当前为验证集冻结表征测量，SSL 数据覆盖与最终独立测试集需要另行审计；尚未建立与 MitoNet 的直接性能结论。','- 均值、SD、逐体积对应关系和差值已独立复算检查。完整精度数据见同目录 JSON 与 CSV。']
    lines+=['','## 本轮判断与下一步建议','','- 两个数据集的平均 Dice、NSD 均下降，平均 HD95 均改善；150k 没有显示出一致的分割质量提升。暂时优先保留 100k 为主基线，150k 保留作对照。','- CMCC 三个体积的 Dice 均下降，主要下降集中在 crop110（约 -0.01970）。该体积召回率从 0.70063 降到 0.66915；其 HD95 改善不能代表漏检减少。CMCT 肝脏体积略有提升，肾脏体积下降。','- 不建议立即延长到 200k。可以先补 125k 的相同冻结评测，检查是否存在中间较优 checkpoint；必要时用额外匹配随机种子复核当前差异。','- 随后优先建立与官方 MitoNet 的同体积分割对照、训练数据覆盖审计及漏检/边界错误分析，再选一个机制做短程验证。P 继续暂缓，当前不启动新模块长训。','- 这次实验仅说明原最小学习率下增加 50k 的收益有限，不能据此证明最优收敛、过拟合或已超过 MitoNet。']
    (OUT/'REPORT_100k_150k_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'summary':rows,'paired_changes':paired},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
