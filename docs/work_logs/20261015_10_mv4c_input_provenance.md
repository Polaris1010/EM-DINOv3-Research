# MV4C两来源输入链核查

用户仅授权Guay vol1和Müller high_c1原图到当前Zarr的灰度方向、强度归一化和空间对齐，复用已经保存的原语义概率。禁止阈值调整、尺度搜索、训练/新推理；本轮仅CPU文件读取、数值对应和展示，完成后停止。

00:29:02香港时间实际catalog/attrs：Guay uint16→uint8(p0.5=35752,p99.5=49283)，Müller uint8 passthrough。catalog记录的/projects原图与原标签均不在当前主机。已找到本机官方ingest源码及导出脚本；后者明确从MitoVerse_data Zarr写出MitoVerse_tiff，不能把这些导出TIFF当作转换前原图。当前官方ingest定义没有反相、transpose或flip；该源码事实尚不能认证缺失原图像素。

仅在当前研究用户目录做一次25秒、maxdepth5的有界定位，排除环境和Zarr；未找到两个原文件，搜索并非全服务器穷举。向用户询问原文件其他路径，同时继续已有证据核查。Guay作者原数据ZIP已定位，只提取eval-images.tif(24层对应验证块)；初始HTTP大范围发生截断，未把截断数据当原图，改为有界分段续传+ZIP CRC。BetaSeg作者公开分享是60GB目录，仅尝试列对应high_c1；公开只读DAV返回401、legacy目录接口500，不尝试密码，不下载整库。

当前脚本仅读Zarr/img、mito、既有cache/GT与probability，对全原生体积逐slab比较并将实际输入SHA与原推理回执关联；概率不再推理，图片不使用Dice选择阈值/空间变换。原图存在时，仅使用官方to_uint8定义复现原转换，不执行writer/main；缺失时，原图栏明确UNVERIFIED，不由Zarr逆造原图。

## 实测结果与范围限制

00:56:24香港时间实际CPU服务正常退出。Guay官方发布uint16 eval-images TIFF通过CRC及实算SHA，24×800×800全像素经原to_uint8复现当前Zarr，差异0、最大误差0，Pearson+0.999480503，MINISBLACK/ZYX。两来源Zarr图像与cache、Zarr mito>0与GT全体积差异0；当前cache SHA均与原概率输入回执一致。Müller当前57.756534%零值，固定官方norm后均值-2.866114；描述性分布，不调整有效区、强度、极性或推理。

原high_c1 TIFF仍未取得，原图→Zarr未核验；作者公开目录接口失败，未把Zarr导出TIFF或逆量化重建当原图。audit_index整体UNVERIFIED且requested_scope_complete=false；可执行检查已完成、报告/四面板交付，明确待原TIFF路径而非把缺口写成PASS。原标注文件亦未重新取得，当前GT/cache一致不等于独立重新认证原GT。

四张原图/Zarr/GT/概率图已生成；Guay与Müller Z525图已实际查看，后者缺失原图框明确说明。六面板含histogram的初版保留；最终四面板单独CPU renderer提高可读性，不重做任何科学推理。原40条模型registry保持，MV4C是独立audit_index登记，不伪造新模型成绩行。

最终CPU核验PASS_AVAILABLE_EVIDENCE：Guay映射/当前两源alignment/SHA关联验证通过；requested_scope_complete=false、Müller原图仍UNVERIFIED。registry --check、repository及21项测试PASS；原40条registry、MV3/MV4/MV4B报告及成绩逐字节/逐项不变。初始代码d3497dc已推送，测量程序的实际SHA与该版一致。最终提交保存报告、四图manifest、结论、源码及本轮Markdown/交接；不自动启动新研究。
