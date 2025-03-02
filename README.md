# metasurface_inverse_design
para_inference_pro.py
加大了模型的参数和分辨率，在4000组数据的情况下能达到0.32832的最小ce loss

para_inference.py
标准的小模型，4000组数据的ce loss 能达到 0.31224
小参数量甚至能够达到更好的ce loss，说明可能需要更多的数据

但是如果采用2000个准确的数据，ce loss 分别可以达到：
0.18137
0.18256

如果采用CNN训练的数据，那么最终的ce loss会达到：0.36并且非常不稳定，体现出了CRNNAG的优越性能


对于这2000个数据训练forward model的情况如下：
CNN:非常之差，训练过程非常不稳定，当然也有可能和数据量相关，但是实在是太差劲了
CRNNAG：训练较CNN更加稳定且最终效果更好了
CRNNAG_attention:似乎注意力机制在这个场景下没有起到很好的作用,初步猜测可能还是和数据量相关

我考察了一下激活函数relu和elu的区别，发现代码中改变的这个激活函数真的会对最终结果产生非常大的影响！！！ 必须考察一下激活函数的问题了。。。。


上述是对于三矩形的情况进行讨论，下面对于二矩形：

采用952个准确数据进行训练，
para_inference 可以达到 0.085的celoss

接下来打算采用CRNNAG和CNN再重新跑一下合成数据:
result:利用3000个CRNNAG的合成数据，
para_inference 可以达到大约0.26的validation loss
利用CNN的合成数据，效果反而好上了不少，可以达到0.166的ce loss

之后的任务可能是修改CRNNAG的模型性能
采用上述952个数据模型训练比较不稳定，test loss震荡比较大，可能需要更多的数据
利用注意力机制优化CRNNAG之后，虽然在模型训练的loss有所改进，但是在进行inverse design的时候依然没有什么改进

现在怀疑是数据的问题，现在统一数据集，再看看效果：
cuda:0 CNN  0.17
cuda:2 CRNNAG  0.26
cuda:3 CRNNAG_attention 0.233
并且我们发现，当teacher ratio调成0的时候，训练效果变差了，说明teacher ratio的重要性

现在尝试进行改变teacher ratio和网络结构，进一步强化
现在希望减少一层CNN_layer,使得它和resnet更加接近，并观察效果


关于teacher force ratio的讨论：
当ratio比较大的时候，模型训练的初期会发生很剧烈的震荡，然后在中后期ratio几乎接近0之后才慢慢稳定下来
ratio=0.3的效果最好，ratio=0和ratio=0.8的效果很接近，都不如0.3，但是比任何的不加attention的模型效果好



目前最好的test loss： CRNNAG with attention teaching-force-ratio of 0.3 （复现成功）
最好的ce loss： CNN  （复现成功）

发现问题了： CNN对于比较简单的pattern（双矩形）能够很好get到其中的规律，但是对于L形加上一个矩形就非常难弄
而CRNNAG就能够通过attention和auto-regressive很好地去做这个事情

1.可能需要做一些ablation study：  attention auto-regressive
结论：对于L pattern: CRNNAG_attention>=CRNN>CNN_attention>=CNN

2.可能需要做一些数据量扩充后的验证： 2000、3000、4000  来对ablation study给出更加有力的论证

3.激活函数？？？ what the fuck
实际上在实验中会有这样一个现象：看似两个模型架构最终收敛的loss相差很大，但是他们的best loss model 其实差别不大，那么这样看最终loss还是有效的吗
查询资料后，可能需要用early stopping来保存住那个我们最需要的model 

在加上early stopping之后，效果：
用了es之后，CNN和CRNNAG都会在非常开始就停下来，这可能是由于数据量的问题
这也导致了最终CNN训练inverse model的效果不是很好，但是CRNNAG却能够保持和不用es一样的水平

我们可以对CRNNAG不如CNN这个问题作出总结了：就是因为数据太简单但是数据量却不够大，导致了CRNNAG学习到了一些无用的知识，不如CNN；但是在更加复杂的数据面前，CRNNAG全面碾压CNN


现在要做的事情：筛选出30954个数据里面，有用的数据，然后进行进一步训练！

我选取了24000个数据，现在可以开跑了
遇到了一些小问题，之前2000个数据比现在8000个数据跑出来还要好。。。
不知道为什么。。。我打算用这24000个数据先试试看跑跑CNN和CRNNAG
效果：
CNN的效果非常好，甚至跑出来的inverse model超出真实数据的好！！
但是CRNNAG似乎遇到了问题？？？ 我觉得我需要考虑这个问题了
并没有问题！！！ CRNNAG的效果也特别特别好！！！只不过在后期才会好起来，前期的振幅非常大，但是并不影响后期的收敛效率

现在进行对比实验： CNN和CRNNAG和CRNNAG_attentiono采用相同的数据进行训练，发现CNN对于更多数据似乎并不能产生更加好的效果；但是CRNNAG 和with attention都可以进一步降低test loss！！这简直太帅了！！


我们将产生的3000数据喂给para_inference.py,然后发现：
确实发现CRNNAG>CNN
说明这一切都是有效的，但是两者之间的差距正在被缩小，我希望CNN能够更加差一点。。。

接下来，我造了6000个数据，打算用这6000个数据继续喂给para_inference.py，然后发现：
随着数据量进一步增大，
ce loss 已经可以降低到0.15————CNN； 0.135————CRNNAG_attention

进一步增大数据量呢？ 我选择增大到19000的数据量！
CNN 到达了0.10
似乎还没有到达极限
接下来采用28000个数据，我希望能够探索到模型的极限
CNN: 0.0796

今天晚上可以开始做这么一件事情，就是使用现有的model去尝试设计exceptional points
设计出来的成果：
差强人意，整个结构的形状可以很好模拟出来
但是对于一些细节上还是会有所疏漏，需要进一步改进！！
怎么改进呢

老师给出了一个改进方案：那就是利用10个数据作为output，而不是1206个数据？！
这样真的可以吗
可能需要归一化，之后继续
