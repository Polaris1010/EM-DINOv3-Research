# Pilot C-1：唯一Barrier预检与预算

用户仅批准实现、状态核验、单元测试和C0/C1各最多20步冒烟。后补Argon原文锁定layer23/postnorm、global256、Chebyshev>2、bottom50%、margin0.05；原文不含固定λ，本轮量级测量使用1，完成后仍建议锁定1，无搜索。

完整D2/30k checkpoint真实存在并按registry SHA核验；student/teacher/optimizer/RNG状态恢复。scheduler原本为函数式，BF16训练原本无GradScaler；没有将不存在字段误报为丢失状态。C0三rank首20步loss及采样链与历史逐值复现。

6项GPU单元测试通过；60组rank-step的global/local crop、mask、RNG和schedule完全相同。C1全部step存在有限非零梯度，anchor无grad且未改变；两接口第23层token对应差0。实际20步激活40.5–69.0%，中位60.4%；局部barrier/主loss backbone梯度比中位6.94%。

两次工程错误均在相关分支首个optimizer更新前：导入缺历史PYTHONPATH（0/0更新）；诊断VJP不支持同一选择性checkpoint图二次反向（C0已20、C1为0）。已保留失败原件，补回原依赖路径并用独立RNG相同诊断图测量；C0不重跑，文件SHA保持原样，最终C1完成20。无改margin/λ/分位数、无装包。

稳态最慢rank中位C0=766.97ms、C1含诊断=1262.67ms、剔除诊断估计861.12ms；anchor57.60ms；peak allocated9.53/11.86GiB。正式15k配对+两head约8.22h，含完整下游约9.23h，建议预留10–11h；仍是20步外推，未实际跑正式训练。

冻结reference使用unmasked eval，student保留原iBOT/DropPath；激活不等于长期结构漂移，更非拓扑保护。15k未来只比较45k同预算终点，不能预设防止100k遗忘。Pilot B中点不证明线性模式连通。

报告docs/Pilot_C_Preflight_and_Budget.md；逐rank和step表、配对检查、哈希及停止回执已保存。所有历史资产只读，GPU0未用、网站200，服务inactive/退出0。无15k SSL、Primus、额外融合或loss。完成CPU检查后提交独立draft PR，不自动合并或继续。

独立[draft PR#16](https://github.com/Polaris1010/EM-DINOv3-Research/pull/16)已创建并推送，未合并；初始结果commit `0c99770a091be0a85b9ab2ec185feb6c43056468`，PR远程head已核验。

CPU registry/仓库检查及51项unittest通过（3项既有平台skip）；6项Barrier测试在服务器GPU通过。补记交付不触发任何实验。
