# OrgMIM / IsoOrg-1K 分组采样核查

2026-10-20。用户询问OrgMIM是否提出针对均匀采样的方法。本轮仅文献/公开源码核查，不改变P0设计、不训练、不新增实验。

## 已确认

官方 `orgmim/trainers/pretrain_provider.py` 的 `Train.__getitem__` 先用random.randint均匀选择group，再random.choice选择组内volume，最后在该体积内随机取三维crop。组g的概率1/G，组内体积v概率1/(G*n_g)。这是分组均衡、组内均匀的层级有放回抽样；没有未访问ID队列，不保证全部体积或空间crop被遍历。DataLoader的shuffle=False不改变其内部随机抽样。

官方 `orgmim/utils/group_file.json` 有5组，共928个条目，大小183/210/106/229/200。启用此文件时每组概率20%；106体积组中每个体积的概率约为229体积组的2.16倍。因此“均匀”针对组，不针对全池体积。分组文件中的组含多个来源名称，不可直接把组解释成独立acquisition或来源。

当前 `scripts/config/orgmim.yaml` 的TRAIN.group_path为空，loader会退化成一个组，按所有列出文件均匀抽样；`legacy/orgmim_mae/config/orgmim.yaml` 明确指定./group_file.json，旧版loader也有两级抽样。不能宣称官方任意默认启动均启用5组均衡。

## 尚未确认

论文DOI为10.1109/TMI.2026.3667612；已读PubMed摘要和官方README，但IEEE全文访问未成功。未核实论文对分组算法的命名、生成特征、聚类方法或新颖性主张。不能把代码事实扩写成“已读原文并证实一种全覆盖新算法”。也不能把5组无证据称为按语义/器官/灰度聚类。

OrgMIM另有结构先验和重建反馈驱动的masking机制；它选择图像内被遮挡的patch，与上述选择训练体积的机制分开。

## 对本项目的含义

我们的来源/collection均衡与其层级抽样思想相关，但它不解决100k预算中的全图ID覆盖保证。本轮不自动引入分组、SAM或新masking，也不增加数据集。若未来考虑内容分组，需要先取得论文对应段落以确认分组定义，不能将其当本轮启动的前置门槛。

## 来源与访问边界

- https://pubmed.ncbi.nlm.nih.gov/41734129/
- https://github.com/yanchaoz/OrgMIM
- https://github.com/yanchaoz/OrgMIM/blob/main/orgmim/trainers/pretrain_provider.py
- https://github.com/yanchaoz/OrgMIM/blob/main/orgmim/utils/group_file.json
- https://github.com/yanchaoz/OrgMIM/blob/main/scripts/config/orgmim.yaml
- https://github.com/yanchaoz/OrgMIM/blob/main/legacy/orgmim_mae/config/orgmim.yaml

源码从上述官方main原始文本于本轮读取，链接为可变分支，非锁定论文运行版本。GitHub tree API限流，使用先前本地目录索引定位文件，再读取实时官方raw文件；没有运行上游代码。
