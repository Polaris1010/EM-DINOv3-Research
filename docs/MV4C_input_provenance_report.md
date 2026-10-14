# MV-4C：Guay / Müller high_c1输入链核查

**Guay的作者原图→当前Zarr已完成像素核验；两来源的当前Zarr→实际输入缓存/GT缓存/既有概率坐标链均一致。Müller转换前原始TIFF尚未取得，原图→Zarr的灰度方向和空间映射仍为UNVERIFIED。** 本轮没有调阈值、搜索尺度、运行模型或训练，也没有修复/重写任何图像或GT。

## 直接测量结果

| 检查 | Guay vol1 | Müller high_c1 |
|---|---:|---:|
| 全原生体积形状ZYX | 24×800×800 | 1097×699×760 |
| 当前图像像素数 | 15,360,000 | 582,770,280 |
| 转换前原图像素验证 | 作者原图已取得 | 未取得，不能认证 |
| 官方raw→uint8复现与Zarr不一致像素 | **0** | UNVERIFIED |
| Zarr img与实际模型输入cache不一致像素 | **0** | **0** |
| Zarr mito>0与实际GT cache不一致像素 | **0** | **0** |
| 当前cache SHA与既有概率输入SHA | 相同 | 相同 |
| 原图灰度方向/几何 | 正向，identity ZYX | 仅metadata声明passthrough，原图未验证 |

检查读取的是当前Zarr内容，不只看shape和元数据。逐64层slab核对全部像素、GT二值前景及缓存值SHA；概率直接读取MV4B已有float32文件，原推理回执native_region_zyx与当前数组范围相同。本轮未进行平移、翻转、转置或配准搜索。GT轮廓叠加图保留在figures/，用户要求的四类照片在comparison_figures/。

## Guay：转换方向、归一化与作者原图

原catalog H5路径在本机不存在，不能假装读取了该H5。由[作者项目页](https://leapmanlab.github.io/dense-cell/)的公开原数据ZIP仅提取`platelet_data/eval-images.tif`，ZIP member CRC通过；TIFF为uint16、ZYX、MINISBLACK（低值为黑），Orientation未设、按TIFF默认top-left。它是作者发布的24层验证裁剪块，并不是由我们的Zarr导出或逆量化重建。原图文件SHA256：`bb4996d78d663d4e270b25a85a6c1e5c8f50cb584377a6e94207ee9ad49a4169`。CRC认证完整下载，不等于独立发表者提供了SHA；该SHA是本轮实算。

仅从本机官方ingest源码AST执行原`to_uint8`定义，不运行writer/main。原uint16→uint8使用全24层的0.5/99.5百分位，记录阈值35752/49283；浮点32计算、clip、×255、round。复现结果在同ZYX坐标上逐像素等于当前Zarr，最大绝对误差0。原图与Zarr同坐标Pearson r=0.999480503，轻微偏离1来自规定裁剪/量化，而非反相。空间形状、位置逐像素复现相符，没有全局flip/transpose/crop/offset的证据。

因此，Guay在此区域的MitoNet低覆盖不能归因于“当前Zarr由这份作者图像转换时反相或错位”。该结果不证明分位归一化对于MitoNet就是最优输入，也不单独区分灰度分布、目标外观、物理尺度或模型语义识别因素。没有执行任何反相、不同归一化或尺度推理。

## Müller：可以验证的下游链与仍缺的原图

catalog及store attrs都声明`uint8(passthrough)`，原路径为`/projects/weilab/liupeng/data/raw/mito/betaSeg/high_c1_im.tiff`，本机不存在。官方ingest在uint8输入时原样返回，没有反相/转置/翻转操作；这只是源码和元数据依据，不能替代实际原TIFF像素及photometric标签验证。

本机`MitoVerse_tiff/data/muller21/high_c1_img.tif`由`MitoVerse_data/zarr_to_tiff.py`从Zarr导出，不是转换前原图，没有把它放到原图栏。仅在当前研究目录进行了有界文件定位，未发现原文件；不是对全服务器不存在的证明。也检查了[BetaSeg作者项目的公开下载入口](https://betaseg.github.io/)；公开目录DAV返回401，legacy目录接口500/404，没有获取high_c1原图，不尝试密码、不下载60GB整库。

当前Zarr图像与实际cache全582,770,280像素一致，二值GT全像素一致；cache值SHA又与MV4B概率生成时记录的输入SHA一致。可以排除这条**Zarr之后**的数据链引入反相/错位、概率对应了另一份图像或另一个crop。没有由本轮认证**原TIFF到Zarr之前**的强度方向或坐标；原图框明确UNVERIFIED。补齐该项仍需要真实转换前high_c1 TIFF或能确认对应文件的可访问路径。

## 描述性强度分布（不改输入）

| 当前uint8体积统计 | Guay | Müller high_c1 |
|---|---:|---:|
| 均值 | 157.930611 | 53.511878 |
| 标准差 | 60.047910 | 68.343245 |
| 零强度比例 | 0.511250% | 57.756534% |
| GT前景区域平均强度 | 95.081797 | 182.597721 |
| 固定官方norm后全体积均值 | +0.341761 | -2.866114 |
| 固定官方norm后全体积标准差 | 1.844747 | 2.099590 |

官方norm为`(uint8/255 -0.57571)/0.12765`，方向为正，与MV4/MV4B原流程一致。不同图像经固定训练集均值/标准差后不必各自均值0、标准差1，不能凭这张表认定归一化实现错误。Müller大量黑色区域强烈影响全体积均值；没有据此改变有效区、切除图像或重算正式成绩。两来源GT区域的明暗差异只是当前图像测量，不能根据Dice或这张表决定反相极性。

本轮检查的是存储chunk中的ZYX像素对应；沿用既有voxel metadata，没有重新校准原采集的物理间距。当前Zarr GT与cache一致及所示轮廓合理，不等同于重新独立认证原始标注语义/完整性；未重标注或质疑用户确认的线粒体GT。

## 对照图与证据

Guay展示Z=1、12，Müller展示Z=525、548；使用前轮既有展示平面及原生中间平面，不通过新Dice排名挑选。四面板为作者原图、当前Zarr、既有GT、既有原语义概率。原图仅为线性显示，其MINISBLACK标签被尊重；Zarr固定0..255，概率固定0..1，显示拉伸不回写模型输入。

本机浏览入口：`E:/mira_website_demo/artifacts/mv4c_control_20261015/comparison_figures/MV4C_gallery.html`。四张图及SHA在`experiments/MV-4C/MV4C_figure_manifest.json`；两来源JSON含全体积histogram、逐slab不一致数量、来源路径和实算SHA。概率与原图留服务器，本轮不提交大数组或图片到Git。

独立服务器根`/opt/data/3_117/tanghr/codex/mv4c_input_provenance_20261015`。完整测量代码commit`d3497dc`，实际源SHA与协议SHA见`evidence/MV-4C/execution_receipt.json`；清晰四面板使用另存CPU renderer，不再次运行测量。服务已inactive/MainPID0/Resultsuccess，网站200；没有GPU或新模型调用。保护回执前后相同，旧40条模型注册记录及MV3/MV4/MV4B历史报告/指标保留原样。

## 结论与停止范围

Guay的当前转换链像素证据支持方向/几何正确；Müller当前Zarr后链也正确，但原TIFF缺口使完整转换前核验尚未完成。没有发现需要修改已存图像/GT/cache的证据，也没有凭低Dice决定灰度极性。此核查不证明原模型已达到最优尺度/输入适配，且不能据此确定新预训练方法。

本轮可执行核查和报告交付后停止。**Müller原图核查明确未完成，而不是PASS；收到该原TIFF实际路径后才能补齐这一项。** 不启动阈值优化、尺度搜索、反相推理、训练或新研究任务。
