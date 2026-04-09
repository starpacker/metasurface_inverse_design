3. Self-design of arbitrary polarization-control waveplates via deep neural networks   \\\
构造BO-NET，利用DNN建立纳米结构和光学响应的关系，通过BO部分来进行搜索，目标是实现对入射光精准态的精确控制   \\\
E:\LZC91.5G\LZC-偏振转化率优化设计 \\\
BO_main.py 是主程序，用来实现meta-surface pattern的优化  \\\
当贝叶斯优化到达饱和的情况下，通过frac_optimization.py对结构进一步微调，随机翻转一些图案来测试是否达到更好性能 \\\
这种随机的优化一般情况下非常低效和费时，所以必须要在BO之后进行   \\\


