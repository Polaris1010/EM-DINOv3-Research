from pathlib import Path
import csv,json,statistics as st
p=Path(__file__).resolve().parent
gpu=json.loads((p/"probe1_receipt.json").read_text())
top=json.loads((p/"probe2_top_fov.json").read_text())
samp=json.loads((p/"sample_manifest.json").read_text())
assert len(samp)==10 and len(list((p/"panels").glob("*_comparison.png")))==10
H=list(csv.DictReader((p/"probe2_hq5nm_holes_summary.csv").open()))
A=list(csv.DictReader((p/"probe1_patch_stats.csv").open()))
assert len(H)==22 and len(A)==150 and gpu["gpu"] in (1,2,3)
lines=["# HQ-5nm 正方形空洞诊断","","## 结论",
       "P0 预测本身有大量类正方形空洞，因此 EM 持续预训练与 Barrier 不是这一现象出现的必要条件。输入降采样和滑窗变化都能显著改变空洞形状及计数。总封闭空洞面积与类正方形计数可向不同方向变化。",
       "在固定的空洞位置，高范数离群 token 存在，但比相应正常线粒体位置更少。末层 PCA 的空洞 patch 与线粒体原型距离在各模型中均远低于背景。P0 中空洞比正常线粒体略远，四个 EM 编码器中排序相反。没有观测到持续预训练后新增的、在五模型对照中一致的 PCA 隔离。",
       "优先检验编码器上下文、Primus 读出和滑窗概率融合之间的交互。这里的形态统计和末层特征归因无法单独定位形成空洞的确切环节。HQ-5nm 无 GT，其封闭空洞可能包括真实结构。",
       f"CPU 统计耗时 {top['seconds']:.2f} 秒。GPU {gpu['gpu']} 推理与可视化耗时 {gpu['seconds']:.2f} 秒。GPU 预检与 checkpoint SHA256 见对应回执。",
       "", "## 探测 2 · 四包形态统计","",
       "| 压缩包 | 模型 | 窗口 | 范围 | 空洞面积占比 %，均值 ± SD（中位） | 类正方形空洞数，均值 ± SD（中位） |",
       "|---|---|---:|---|---:|---:|"]
def cell(r,key,mul=1):
    mean,sd,median=(float(r[key+"_"+k])*mul for k in ("mean","sd","median"))
    return f"{mean:.2f} ± {sd:.2f} ({median:.2f})"
for r in H:
    lines.append("| "+" | ".join([r["pack"],r["model"],r["window"] if r["window"]!="0" else "整图",r["scope"],cell(r,"hole_area_ratio",100),cell(r,"num_square_like_holes")])+" |")
lines+=["","每张掩膜使用 4 连通背景填孔与组件标记。共同 20 张采用相同 FOV；130k 额外报告完整 25 张。无孔样本的孔面积分位数记作零。逐图与完整统计见 [per-image CSV](probe2_hq5nm_holes_per_image.csv) 和 [summary CSV](probe2_hq5nm_holes_summary.csv)。",
         "挑样时将每个 FOV 在四包中的空洞比例最大值和类正方形孔数最大值分别排名，按两种排名之和选 6 个。参考掩膜取该 FOV 类正方形孔最多的一组。两个单独排名和最终入选见 [Top FOV](probe2_top_fov.json)。",
         "", "## 探测 1 · 末层特征", "",
         "| 模型 | 孔洞 L2 均值 | 孔洞 L2 P95 | 孔洞高范数 | 正常线粒体高范数 | 余弦 孔洞 / 正常 / 背景 | PCA 原型距离 孔洞 / 正常 / 背景 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for name in gpu["checkpoint_hashes"]:
    groups=[[r for r in A if r["model"]==name and r["region"]==k and int(r["n"])] for k in ("hole","normal_mito","bg")]
    h,n,b=groups
    avg=lambda rows,key:st.mean(float(r[key]) for r in rows if r[key])
    high=lambda rows:f"{sum(int(r['norm_high_count']) for r in rows)}/{sum(int(r['n']) for r in rows)}"
    trip=lambda key:" / ".join(f"{avg(rows,key):.3f}" for rows in groups)
    lines.append(f"| {name} | {avg(h,'norm_mean'):.1f} | {avg(h,'norm_p95'):.1f} | {high(h)} | {high(n)} | {trip('cos_mean')} | {trip('pca_proto_distance')} |")
lines+=["","孔洞 L2 使用 LayerNorm 前的最后一层 token，余弦和 PCA 使用 Primus 所用的归一化 token。每个 256 裁块严格包含 256 个空间 token，排除 CLS 和 register tokens。空洞覆盖至少 12.5% 的 patch 计为空洞组。预测前景覆盖至少 50% 且空洞至多 5% 计为正常线粒体。预测前景与空洞覆盖均不超过 5% 计为背景。补边和其他混合 patch 排除。高范数阈值是同一模型同一视野有效 patch 的均值加 2 个总体标准差。",
         "L2、P95、余弦与 PCA 距离先按视野计算，再对有对应 patch 的视野等权平均。离群分子和分母跨视野合计。空洞 token 有 399 个，其中 HQ 有 388 个，MB-1 有 11 个。MB-1 的 4 个视野从既有 52 视野列表中以锁定的 C1 130k / 512 预测复算，选择至少 24 个像素且过半面积位于 GT 前景的孔。该子样本用于检查现象，无法代表全量 MB-1。",
         "PCA 距离为三维 PCA 中到正常线粒体原型的距离，以本视野 PCA 坐标总体 RMS 归一化。不同模型与视野各自拟合，伪彩颜色不能跨图直接比对。此距离支持判断空洞是否总体接近正常线粒体，无法证明单个空洞在空间上拓扑连通。",
         "", "### 检查样本",""]
for s in samp:
    lines.append(f"- {s['id']}（{s['domain']}，裁剪起点 y/x {s['origin']}） [拼图](panels/{s['id']}_comparison.png)")
lines+=["","完整模型 × 样本 × 区域数据见 [Patch CSV](probe1_patch_stats.csv)。原始样本坐标、预测版本与归因统计记录于 [样本清单](sample_manifest.json)。"]
(p/"DIAGNOSIS_SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
