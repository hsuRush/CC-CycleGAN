# CC-CycleGAN
Class-Consistency CycleGAN restrict generated data by given labels. Derived from original CycleGAN, we insert two classification Networks with shared weights as the two constrain branches.

In this repo, I generate license plate and restricted the generated license plate characters by inserting a crnn network as the classifcation Network and training the branch with ctc-loss.

# Architecture 
<p align="center">
<img src="https://github.com/hsuRush/CC-CycleGAN/blob/master/demo/cccgan_demo.gif?raw=true" width="750" height="450"/>
</p>

# Comparison from original CycleGAN in license plate style transfer
<p align="center">
<img src="https://github.com/hsuRush/CC-CycleGAN/blob/master/demo/demo_camparion.png?raw=true" width="750" height="450"/>
</p>
The the license plate number in proposed CC-CycleGAN are clear and recognizable. 

# Disclaimer
This is highly based on the subfolder "cyclegan" in https://github.com/eriklindernoren/Keras-GAN repo.
