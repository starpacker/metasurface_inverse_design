这里将会保存各个版本的核心模型架构
1. para_inference.py  \\
  融合了ssim_loss,pixel_loss,boundary_loss,bce_loss  \\
  效果一般，边缘模糊，矩形之间位置关系不明确  \\
NOTE: 需要调整各个损失函数的权重  \\

2. para_inference_1.py   \\
   重新构造了损失函数，利用sobel算符
   会出现很多无用的边缘图案，正在尝试调整权重优化

3. para_inference_resnet.py \\
   利用了resnet-like的网络架构，但似乎爆内存了，需要更大的GPU才能跑

   
   
   
