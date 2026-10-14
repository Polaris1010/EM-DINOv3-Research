# MV4C两来源输入链核查

用户仅授权Guay vol1和Müller high_c1原图到当前Zarr的灰度方向、强度归一化和空间对齐，复用已经保存的原语义概率。禁止阈值调整、尺度搜索、训练/新推理；本轮仅CPU文件读取、数值对应和展示，完成后停止。

00:29:02香港时间实际catalog/attrs：Guay uint16→uint8(p0.5=35752,p99.5=49283)，Müller uint8 passthrough。catalog记录的/projects原图与原标签均不在当前主机。已找到本机官方ingest源码及导出脚本；后者明确从MitoVerse_data Zarr写出MitoVerse_tiff，不能把这些导出TIFF当作转换前原图。当前官方ingest定义没有反相、transpose或flip；该源码事实尚不能认证缺失原图像素。

仅在当前研究用户目录做一次25秒、maxdepth5的有界定位，排除环境和Zarr；未找到两个原文件，搜索并非全服务器穷举。向用户询问原文件其他路径，同时继续已有证据核查。Guay作者原数据ZIP已定位，只提取eval-images.tif(24层对应验证块)；初始HTTP大范围发生截断，未把截断数据当原图，改为有界分段续传+ZIP CRC。BetaSeg作者公开分享是60GB目录，仅尝试列对应high_c1；公开只读DAV返回401、legacy目录接口500，不尝试密码，不下载整库。

当前脚本仅读Zarr/img、mito、既有cache/GT与probability，对全原生体积逐slab比较并将实际输入SHA与原推理回执关联；概率不再推理，图片不使用Dice选择阈值/空间变换。原图存在时，仅使用官方to_uint8定义复现原转换，不执行writer/main；缺失时，原图栏明确UNVERIFIED，不由Zarr逆造原图。
