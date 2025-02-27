这里将会保存各个版本的核心模型架构
1. para_inference.py  \\
  融合了ssim_loss,pixel_loss,boundary_loss,bce_loss  \\
  效果一般，边缘模糊，矩形之间位置关系不明确  \\
NOTE: 需要调整各个损失函数的权重  \\

2. para_inference_1.py   \\
   重新构造了损失函数，利用sobel算符 \\
   会出现很多无用的边缘图案，正在尝试调整权重优化  \\

3. para_inference_resnet.py \\
   利用了resnet-like的网络架构，但似乎爆内存了，需要更大的GPU才能跑 \\

4.auto_regressive.py \\
  实现了forward model和inverse design的结合，正向网络利用CRNNAG来实现精确预测，反向网路则利用效果最好的hybriddecoder作为骨架实现



----------------------------------------------------------------------------------------------------------------------------
                                            实验结果汇总
para_inference_pro.py
加大了模型的参数和分辨率，在4000组数据的情况下能达到0.32832的最小ce loss

para_inference.py
标准的小模型，4000组数据的ce loss 能达到 0.31224
小参数量甚至能够达到更好的ce loss，说明可能需要更多的数据

但是如果采用2000个准确的数据，ce loss 分别可以达到：
0.18137
0.18256

上述是对于三矩形的情况进行讨论，下面对于二矩形：

采用952个准确数据进行训练，
para_inference 可以达到 0.085的celoss

接下来打算采用CRNNAG和CRNN再重新跑一下合成数据

之后的任务可能是修改CRNNAG的模型性能
采用上述952个数据模型训练比较不稳定，test loss震荡比较大，可能需要更多的数据


关于teacher force ratio的讨论：
尝试了0.8,0.5,0.3的三种版本
   
   
   
