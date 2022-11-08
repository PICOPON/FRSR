# FRSR

An improved network based on small target detection network [SOD-MTGAN:Small Object Detection via Multi-Task Generative Adversarial Network](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yongqiang_Zhang_SOD-MTGAN_Small_Object_ECCV_2018_paper.pdf)

FRSR is mainly used for small target detection enhancement, which mainly consists of three main components as shown in the figure below, RPN network for proposal box generation, hyper-segmentation network for small target ROI region feature enhancement, and Resnet network for classification and location regression.

![FRSR](https://user-images.githubusercontent.com/49949166/197488938-8cc3d17c-5ad1-4917-81ad-811b40d80e62.jpg)

# Sequence of training steps
1.Fix the weights of the other model parts and train the RPN network parts
2.Fix the weights of the other model parts and train the SRCNN network parts
3. Fix the weights of the other model parts and train the MTHead network parts



