from pathlib import Path
import csv,json,statistics as st
p=Path(__file__).resolve().parent
r=list(csv.DictReader((p/"square_holes.csv").open()))
receipt=json.loads((p/"receipt.json").read_text())
assert len(receipt["images"])==12 and all(int(x["mask_replay_differing_pixels"])==0 for x in receipt["images"])
out=["# 单窗与滑窗融合归因","","沿用 HQ-5nm Top-6 的 FOV ID，并统一使用各模型原网格 256 / stride 128 掩膜。P0 与 C1 Barrier100k 各运行全图的 225 个窗口，完整复算原历史均匀概率融合。所有 12 张历史二值掩膜均逐像素复现。",
     f"物理 GPU {receipt['gpu']} · {receipt['seconds']:.2f} 秒。单窗图取完整覆盖孔洞而最接近孔中心的真实滑窗。另在 CSV 中逐孔检查所有完整覆盖窗口。红区是融合结果中的参考孔。前景 margin = 前景 logit 减背景 logit。",
     "",
     "| 模型 | 方孔 | 任一单窗同位置封闭孔覆盖 ≥90% | 居中单窗覆盖 ≥90% | 所有贡献窗口均为负的孔内像素，按孔等权 | 至少一窗为正的像素，按孔等权 | 边界距 16 格线 ≤1 px | 边界距 128 切缝 ≤1 px |",
     "|---|---:|---:|---:|---:|---:|---:|---:|"]
for name in dict.fromkeys(x["model"] for x in r):
    a=[x for x in r if x["model"]==name]
    n=len(a)
    f=lambda k:sum(bool(x[k]) and float(x[k])>=.9 for x in a)
    mean=lambda k:100*st.mean(float(x[k]) for x in a)
    border=lambda k:sum(int(x[k]) for x in a)
    out.append(f"| {name} | {n} | {f('any_tile_max_enclosed_fraction')} | {f('single_enclosed_fraction')} | {mean('all_votes_negative_fraction'):.1f}% | {mean('any_positive_vote_fraction'):.1f}% | {border('borders_within1_patch')}/{4*n} | {border('borders_within1_stride')}/{4*n} |")
    ranges=sorted(a,key=lambda x:int(x["area"]),reverse=True)
    print(name,"N",n,"any_single",f('any_tile_max_enclosed_fraction'),"center_single",f('single_enclosed_fraction'),"all_neg_pct",round(mean('all_votes_negative_fraction'),2),"border16",border('borders_within1_patch'),"border128",border('borders_within1_stride'),"max",[(x['fov'],x['height'],x['width'],x['any_tile_max_enclosed_fraction'],x['panel_cos_hole'],x['panel_cos_normal']) for x in ranges[:2]])
out+=["","[逐孔坐标与几何统计](square_holes.csv) 包含左上角和右下角（排他坐标）、宽高、模 16/128 及各窗口符号统计。任一窗口的 ≥90% 定义是该窗独立阈值预测的封闭孔覆盖了原孔至少 90% 的像素。均匀随机边缘相位下，距格线 ≤1 像素的描述性比例约为 3/16 和 3/128。孔边缘存在样本内依赖，此处未做显著性检验。",
      "如果某部分孔在融合前已有负 logit 或完整闭孔，滑窗融合不是该部分孔形成的必要条件。来自不同窗口的符号分歧可使融合改变孔的边界。HQ-5nm 无 GT，当前结果能定位输出环节，无法单凭余弦或概率证明物理视野不足或某一层是唯一成因。",
      f"共 {len(list((p/'panels').glob('*.png')))} 张并排诊断图，位于 panels。"]
(p/"FUSION_TRACE_SUMMARY.md").write_text("\n".join(out)+"\n",encoding="utf-8")
