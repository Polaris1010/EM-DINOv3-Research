from pathlib import Path
import json,csv,statistics,math
ROOT=Path(__file__).resolve().parent/'mitoverse_CMCC_CMCT_label1_20261012'

def main():
    results=json.loads((ROOT/'results.json').read_text(encoding='utf-8'))
    audit=json.loads((ROOT/'label_audit.json').read_text(encoding='utf-8'))
    assert results['status']=='complete' and audit['status']=='pass'
    assert len(results['summary'])==6 and len(results['per_volume'])==15
    for row in results['summary']:
        cases=[r for r in results['per_volume'] if r['dataset']==row['dataset'] and r['encoder_kind']==row['encoder_kind']]
        assert len(cases)==row['n']
        for m in ['dice','hd95','nsd']:
            vals=[r[m] for r in cases]
            assert math.isclose(statistics.mean(vals),row[m+'_mean'],rel_tol=2e-7,abs_tol=1e-7)
            assert math.isclose(statistics.stdev(vals),row[m+'_std'],rel_tol=2e-6,abs_tol=1e-7)
    names={'official_natural':'官方 DINOv3 自然图像权重','B_full100k':'B_full 100k','B_full150k':'B_full 150k'}
    lines=['# CMCC / CMCT：仅 label == 1 的线粒体冻结评测','','已对三个编码器的现有预测重新计算，GT 和预测均明确使用 `== 1`。其他非负语义标签视为背景。','','## 三模型结果','','| 数据集 | 编码器 | 验证体积 n | Dice ↑ | HD95 ↓ | NSD ↑ |','|---|---|---:|---:|---:|---:|']
    for r in results['summary']:
        lines.append(f"| {r['dataset'].split('_')[-1]} | {names[r['encoder_kind']]} | {r['n']} | {r['dice_mean']:.4f} ± {r['dice_std']:.4f} | {r['hd95_mean']:.2f} ± {r['hd95_std']:.2f} | {r['nsd_mean']:.4f} ± {r['nsd_std']:.4f} |")
    lines+=['','SD：完整验证体积之间的样本标准差，ddof=1；不是多种子标准差。CMCC n=3，CMCT n=2。HD95 数值按 voxel 等效单位报告；NSD 固定容差 1，读取原标签 spacing=(1,1,1)。','','## 本次标签核查','','实际使用的两个目录是：','','- `/opt/data/3_117/tanghr/Med_DINOv3/nnUNet_raw/Dataset111_CMCC/labelsTr`','- `/opt/data/3_117/tanghr/Med_DINOv3/nnUNet_raw/Dataset131_CMCT/labelsTr`','','两个 dataset.json 均声明 `background: 0, mitochondria: 1`。5 个完整验证标签都只有 0/1，当前输入中不存在其他正类别。全部 54 个预处理训练/验证标签文件均仅含 -1、0、1，-1 为忽略值。','','| 数据集 | 体积 | 标签 1 体素数 | 其他正类别体素数 |','|---|---|---:|---:|']
    for r in audit['validation_labels']:
        lines.append(f"| {r['dataset'].split('_')[-1]} | {r['case']} | {r['mito_voxels']} | {r['other_positive_class_voxels']} |")
    lines+=['','此前正式 Dice、HD95、NSD 的 MedDINOv3 指标入口已经使用 `label_array == c`，本轮只传入一个类别，c=1。冻结脚本中曾使用的 `GT > 0` 在当前二值输入上与 `GT == 1` 完全相同。此次重新导出明确的 label1 掩码并执行同一指标入口后，六组均值和 SD 与原结果一致。','','没有根据标签纠正重训 encoder 或 decoder：已核实当前训练标签没有额外正类别。另已保存后续使用的 `frozen_probe_label1.py`，训练目标显式将 1 映射为前景、其他非负标签映射为背景、负值映射为忽略值；验证 GT 使用 `== 1`。原始历史脚本保留以保证已有实验可追溯。','','## 结论','','- 100k 相比官方自然权重，在两组的平均 Dice、HD95、NSD 均更好。','- 150k 相比 100k，两组平均 HD95 改善，但平均 Dice 和 NSD 下降；原来的比较结论不因此次明确 label1 重算而改变。','- 当前仍是小样本、单种子的验证集冻结测量，不能据此声称已超过 MitoNet 或已达到预训练上限。','','完整精度和逐体积结果见 results.json、summary.csv、per_volume_metrics.csv；标签值计数、文件路径和哈希见 label_audit.json。']
    (ROOT/'REPORT_label1_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'verified','summary':results['summary'],'matches_previous':results['matches_previous_class1_results'],'audited_preprocessed_files':audit['preprocessed_count']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
