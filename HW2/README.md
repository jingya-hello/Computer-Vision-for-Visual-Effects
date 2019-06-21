ISA525700 Computer Vision for Visual Effects<br/>Homework 2 (Style Transfer)
===
## [MUNIT](https://arxiv.org/pdf/1804.04732.pdf)

### Introduction
assume the image representation can be decomposed into a content code that is domain-invariant and a style sode that captures demai 

### Architecture
![](https://i.imgur.com/7I5BbwW.png)

### Training result
![](https://i.imgur.com/DLjEUKb.png)
(training途中電腦當機因此附上path file)

### Model
![](https://i.imgur.com/84ovboJ.png)

#### Loss

1. Bidirectional reconstruction loss
ensures the encoders and decoders are inverses

    * Image reconstruction loss
    ![](https://i.imgur.com/yhVhF5z.png)

    * Latent reconstruction loss
    ![](https://i.imgur.com/Z552Y9Y.png)
2. Adversarial loss
matches the distribution of the translated images to the image distribution in the target domain
![](https://i.imgur.com/td6uQhu.png)

#### Inference 
| style / content|![](https://i.imgur.com/9ltT2vM.jpg)|![](https://i.imgur.com/lzxHFcV.jpg)|![](https://i.imgur.com/SQejoux.jpg)|![](https://i.imgur.com/W8yY4xK.jpg)| 
| ----------------- | --------------- |--------------- |--------------- |--------------- |
|![](https://i.imgur.com/R3tYQtZ.jpg)|![](https://i.imgur.com/PsVtvgs.jpg)|![](https://i.imgur.com/PJzwECp.jpg)|![](https://i.imgur.com/qChSK76.jpg)|![](https://i.imgur.com/wphGNtr.jpg)|
|![](https://i.imgur.com/oy1qcOq.jpg)|![](https://i.imgur.com/xsiV8am.jpg)|![](https://i.imgur.com/XHSB7uK.jpg)|![](https://i.imgur.com/xwUZQW3.jpg)|![](https://i.imgur.com/aeSLwAe.jpg)|
|![](https://i.imgur.com/O1oQeaY.jpg)|![](https://i.imgur.com/URrnvwK.jpg)|![](https://i.imgur.com/hsOj7Vl.jpg)|![](https://i.imgur.com/jtClG5V.jpg)|![](https://i.imgur.com/6uWeorv.jpg)|
|![](https://i.imgur.com/dToD0ZG.jpg)|![](https://i.imgur.com/qgZ2sIL.jpg)|![](https://i.imgur.com/Wm9wqPW.jpg)|![](https://i.imgur.com/AhjcV5u.jpg)|![](https://i.imgur.com/hZgX1eX.jpg)|
|![](https://i.imgur.com/umi8vfw.jpg)|![](https://i.imgur.com/LhDUXCp.jpg)|![](https://i.imgur.com/b5bq0Sk.jpg)|![](https://i.imgur.com/W5ePKaQ.jpg)|![](https://i.imgur.com/5y2bcl4.jpg)|
|![](https://i.imgur.com/v5PWWzP.jpg)|![](https://i.imgur.com/a9H54NN.jpg)|![](https://i.imgur.com/tNEA8DQ.jpg)|![](https://i.imgur.com/trOxmEp.jpg)|![](https://i.imgur.com/dVSlLH4.jpg)|
|![](https://i.imgur.com/wPNhza1.jpg)|![](https://i.imgur.com/5QQ7NAo.jpg)|![](https://i.imgur.com/38pS5mD.jpg)|![](https://i.imgur.com/YwRSJTW.jpg)|![](https://i.imgur.com/XBUTxAj.jpg)|
|![](https://i.imgur.com/GP7AmNz.jpg)|![](https://i.imgur.com/KsbkMyB.jpg)|![](https://i.imgur.com/4R2daP6.jpg)|![](https://i.imgur.com/Sgllz4p.jpg)|![](https://i.imgur.com/NQQnrmH.jpg)|


## Other mothods

### [Unsupervised Image-to-Image Translation Networks](https://arxiv.org/pdf/1703.00848.pdf)
#### Introduction
UNIT assumes for any given pair of images x1 and x2, there exists a shared latent code z in a shared-latent space, such that we can recover both images from this code.
#### Architecture
![](https://i.imgur.com/n3kPAfD.png)
E1 , E2 : 2 encoder / G1 , G2 : 2 generation function

UNIT implements shared-latent space assumption using a weight sharing constraint where the connection weights of the last few layers( high-level layers_ un E1 and E2 are tied (illustrated using dashed lines) and the connection weights of the first few layers (high-level layers) in G1 and G2 are tied.

##### Interpretation of the roles of the subnetworks in the proposed framework
{E1 , G1} : VAE for X1
{E1 , G2} : Image Translator X1 -> X2
{G1 , D1} : GAN for X1
{E1 , G1 , D1} : VAE-GAN
{G1 , G2 , D1 , D2} : CoGAN

#### Details
##### VAE
VAE1 first maps x1 to a code in a latent space Z via the encoder E1 and then decodes a random-perturbed version of the code to reconstruct the input image via the generator G1.

##### weight sharing

enforce a weight-sharing constraint in two VAE
a pair of corresponding images in the two domains can be mapped to a common latent code by E1 and E2, and a latent code will be mapped to a pair of corresponding images in the two domains G1 and G2.

#### GANs
apply adversarial training to images from the translation stream

#### loss function
VAE : 
![](https://i.imgur.com/JnonWfO.png)
GAN :
![](https://i.imgur.com/4z2Amw6.png)

### Comparisons 
1. In UNIT, it assumes that different domain share same latent code, however, in MUNIT, it only assumes that they share same content code
2. comparison table from MUNIT paper
![](https://i.imgur.com/GTsGKL1.png)
diversity score is the average LPIPS distanse and the quality score is the human preference score, from both metrics, the higher the better.
We can see from both metrics, MUNIT performs better. Since the difference in the paper's assumption, UNIT fails to generate diverse outputs compare to UNIT model.


## [Cycle GAN](https://arxiv.org/pdf/1703.10593.pdf)
### Introduction
Cycle gan跟傳統的gan做圖像轉換的方式不同，它不需要配對的數據集(paired image data set)；利用兩個generator、discrimnator和轉換的一致性(consistency)，cycle gan只需要不同風格的unpaired image data set即可運作。

### Architecture
Cycle gan總共有兩個Genertoar(G, F)，分別把圖像從Domain X到Domain Y，跟反向；同時也有兩個相對應的Discriminator(DY, DX)。

不過，因為input為未配對數據集，因此在風格轉換後(X -> Y')，還需要將結果再逆轉換回來(Y' -> X')，並做比較以確保轉換過後結構相似。

![](https://i.imgur.com/vqnmh1F.png)

### Loss Function
```
L(G, F, DX, DY) = Lgan(G, DY, X, Y) + Lgan(F, DX, Y, X) +　λLcyc(G, F) 
```
_Lcyc(G, F) = Ex∼     pdata(x)[‖F(G(x))−x‖1] + Ey∼pdata(y)[‖G(F(y))−y‖1], whcih represents loss of structure cmparison_

### Comparisons 
1. comparison from MUNIT paper
![](https://i.imgur.com/GTsGKL1.png)
diversity score is the average LPIPS distanse and the quality score is the human preference score, from both metrics, the higher the better.
We can see from both metrics, MUNIT performs better. The diversity of MUNIT is better than CycleGan.

2. comparison of results (summer2winter)

| summer| winter - MUNIT(output1)  | winter - MUNIT(output2)  | winter - Cycle GAN | 
| ----------------- | --------------- |--------------- |--------------- |
|![](https://i.imgur.com/d0sYOIk.jpg)|![](https://i.imgur.com/sPw4lRd.jpg)|![](https://i.imgur.com/pF11lGx.jpg)|![](https://i.imgur.com/3JLPDwV.png)|
|![](https://i.imgur.com/IhaMsUo.jpg)|![](https://i.imgur.com/9dUgDzQ.jpg)|![](https://i.imgur.com/FDZnZB2.jpg)|![](https://i.imgur.com/KR1at34.png)|
|![](https://i.imgur.com/OV7WZ5k.jpg)|![](https://i.imgur.com/1FSHJYW.jpg)|![](https://i.imgur.com/15ft3rz.jpg)|![](https://i.imgur.com/PgoUd6H.png)|
|![](https://i.imgur.com/QsVC5fQ.jpg)|![](https://i.imgur.com/6nqt2hc.jpg)|![](https://i.imgur.com/0osNOIZ.jpg)|![](https://i.imgur.com/u3gD7fn.png)|
|![](https://i.imgur.com/v9KqfZq.jpg)|![](https://i.imgur.com/xbIwQ9F.jpg)|![](https://i.imgur.com/EsRCHkq.jpg)|![](https://i.imgur.com/oRCyFQI.png)|
|![](https://i.imgur.com/aPEMFuN.jpg)|![](https://i.imgur.com/uzopE4Q.jpg)|![](https://i.imgur.com/uRHiIOH.jpg)|![](https://i.imgur.com/4v4t2ER.png)|
|![](https://i.imgur.com/GFlRYK9.jpg)|![](https://i.imgur.com/NtiwYtq.jpg)|![](https://i.imgur.com/6feAxuw.jpg)|![](https://i.imgur.com/0UpyStG.png)|
|![](https://i.imgur.com/ifOuh48.jpg)|![](https://i.imgur.com/XuyYMO3.jpg)|![](https://i.imgur.com/JFQwJ8l.jpg)|![](https://i.imgur.com/xJNCyY8.png)|



### [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)

#### Introduction

該論文透過卷積神經網路，將圖片的內容及風格分開並重建，提供一個style transfer的做法。
#### Architecture
![](https://i.imgur.com/kuZTYgs.png)
##### Concept
利用pre-trained VGG19 model提取圖片不同layer的特徵，作為圖像和風格的特徵。
- Content:在network中，較低層級保留較多的原始圖像，而較高層級中雖然沒有細節的pixel但保留了high level的內容feature，因此可以利用高層的特徵來對圖像做recontruct。
- Style:在原本的CNN架構上建立新的feature space來提取風格的特徵。作法為計算不同layer所產生的不同feature之間的correlations。在reconstruction中，該結果確實獲得的原始影像的texture、color等特徵。

#### Loss Function
- Loss Function = content loss + style loss
- Content loss : 原圖與預測圖的content feature差距
- Style loss : 原圖與預測圖的style feature差距

#### Flow
![](https://i.imgur.com/zVpm139.png)
![](https://i.imgur.com/FseG3XF.png)

#### Example
![](https://i.imgur.com/felJKFp.jpg)

![](https://i.imgur.com/FkezUlt.png)

#### Comparison

1. comparison of results

| summer| winter - MUNIT(output1)  | winter - MUNIT(output2)  | winter - neural | 
| ----------------- | --------------- |--------------- |--------------- |
|![](https://i.imgur.com/d0sYOIk.jpg)|![](https://i.imgur.com/sPw4lRd.jpg)|![](https://i.imgur.com/pF11lGx.jpg)|![](https://i.imgur.com/4wJEQUk.jpg)|
|![](https://i.imgur.com/IhaMsUo.jpg)|![](https://i.imgur.com/9dUgDzQ.jpg)|![](https://i.imgur.com/FDZnZB2.jpg)|![](https://i.imgur.com/H93A4Fc.jpg)|
|![](https://i.imgur.com/OV7WZ5k.jpg)|![](https://i.imgur.com/1FSHJYW.jpg)|![](https://i.imgur.com/15ft3rz.jpg)|![](https://i.imgur.com/AheBfl4.jpg)|
|![](https://i.imgur.com/QsVC5fQ.jpg)|![](https://i.imgur.com/6nqt2hc.jpg)|![](https://i.imgur.com/0osNOIZ.jpg)|![](https://i.imgur.com/oi0aWlY.jpg)|
|![](https://i.imgur.com/v9KqfZq.jpg)|![](https://i.imgur.com/xbIwQ9F.jpg)|![](https://i.imgur.com/EsRCHkq.jpg)|![](https://i.imgur.com/Zz46hrs.jpg)|
|![](https://i.imgur.com/aPEMFuN.jpg)|![](https://i.imgur.com/uzopE4Q.jpg)|![](https://i.imgur.com/uRHiIOH.jpg)|![](https://i.imgur.com/E4mMIZ4.jpg)|
|![](https://i.imgur.com/ifOuh48.jpg)|![](https://i.imgur.com/XuyYMO3.jpg)|![](https://i.imgur.com/JFQwJ8l.jpg)|![](https://i.imgur.com/YLX7LuY.jpg)|

2. 此論文中，style feature的提取是設計來得到style image的紋路、筆觸等特徵，在winter2summer此類題目中，較沒有辦法達到style(weather)的轉換。


