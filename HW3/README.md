ISA525700 Computer Vision for Visual Effects<br/>Homework 3 (homework3-GAN-Dissection)
===

## [GAN Dissection: Visualizing and Understanding Generative Adversarial Networks](https://arxiv.org/pdf/1811.10597.pdf)

### Introduction
此論文主要是藉由視覺化的方式來探討、解析GAN，以便人們更容易了解每個unit所代表的意思，或者是說圖片中的每一個部分是由哪一個unit所生成，因為卷積神經網路（CNN）具有空間相關性，任一層的每一個feature map都可以對應到前一層的某一個feature map，所以此論文以生成特定圖片的任務為例，利用segmentation和根據feature map產生的mask進行比對，判定哪個unit跟哪個class相關聯，如此，就能藉由修正特定unit來達到所需的結果。


## Generate images with GANPaint
|input image| mask |result|
|---|---|---|
|![](https://i.imgur.com/BFoqSB9.png)|![](https://i.imgur.com/njtSTgg.png)|![](https://i.imgur.com/B7tYLL9.png)|

## Dissect and analyze GAN model

### 生成之HTML

#### Living Room
![](https://i.imgur.com/aDAsNXr.jpg)
![](https://i.imgur.com/HIPezoF.jpg)

#### Bed Room
![](https://i.imgur.com/56uIPtY.jpg)


### 分析
這次我們dissect的是living room GAN model，從dissect的結果我們可以觀察並分析GAN的特徵：
1. 當我們加入或是刪除一個units時，該units所代表的類別的物件即會同時被加入或是刪除(或是放大、縮小)，因此可以推斷即使物件的類型豐富、外觀迥異，但同一個neuron仍學習到該物件的不同特徵，因此可以被用來控制同一個類別的物件。
![](https://i.imgur.com/nxIYpLb.jpg)
例如上圖，我們於右方選擇sofa-iou對左圖的原圖進行ablation，並框出沙發的區域（如最右圖），即可產生中間的結果圖，在ablate 60個沙發unit後，圖中沙發被大幅縮小，代表這些unit確實可以用以控制圖中沙發的產生。
2. 從GAN Paint也可以看到一項特徵，例如下圖例子，若試圖於圖中天空的位置加入門的項目，最後的結果則會失敗，也就是說GAN可以學習到該物件適當的位置。

|input image| mask |result|
|---|---|---|
|![](https://i.imgur.com/01vpxAW.png)|![](https://i.imgur.com/MkUGdMZ.png)|![](https://i.imgur.com/lIt9mRo.png)|

## Other mothods

### [Globally and Locally Consistent Image Completion](http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf)

#### Introduction
本篇論文利用全域和局部判斷器（global/local discriminator）來提高修復圖像會局部模糊的問題。首先，先利用strided concolution的方式降低分辨率；同時利用dilated convolutional layers來擴大圖像訊息的獲得。接著，把修復好的完整圖片輸入global discriminator，並把局部修復的區域輸入local discriminator；最後，把兩個的卷積向量經過fully connected輸出。

#### Comparison
由於此作法主要是將圖片中的某個物件移除，因此我們將其結果與利用GAN Dissection的DEMO進行移除物件的結果做比較，可以發現，由於多考慮了全域（global）的圖像資訊，用以降低模糊程度，但在局部的細節裡，結果卻不甚理想，不僅有不規則且不自然的稜角，且顏色也無法根據周圍的結構做調整；反觀GAN在這部分就做得好很多，雖然能是有小缺點那就是有些地方沒有清除乾淨，但整體來說已經比較好很多了。

|Image|![](https://i.imgur.com/tkytevS.png)|![](https://i.imgur.com/jDmULYR.png)|![](https://i.imgur.com/Z1j4kyM.png )|
|---|---|---|---|
|GLCIC|![](https://i.imgur.com/42cnMaV.jpg)|![](https://i.imgur.com/N6XBxuY.jpg)|![](https://i.imgur.com/FQEC9E0.jpg)|
|GAN|![](https://i.imgur.com/A8E4WmR.jpg)|![](https://i.imgur.com/Mdj2nov.jpg)|![](https://i.imgur.com/WGxsi7v.png)|
|Mask|![](https://i.imgur.com/qsLHmEm.jpg)|![](https://i.imgur.com/5xCv6RO.jpg)|![](https://i.imgur.com/EqmKUyM.jpg)|

### [A Closed Form Solution to Natural Image Matting](http://webee.technion.ac.il/people/anat.levin/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf)

#### Introduction
該論文基於圖像的局部光滑假設，利用代數的方法推倒closed-form的方法解出以下的式子，其中F為前景、B為背景，以將前景與背景分離。
![](https://i.imgur.com/rVKFVst.png)

#### 使用closed-form matting之結果合成影像

利用alpha matting的方法，我們選取一張樹的圖片，將此作為前景F並將input圖片作為背景圖B，兩者做合併。將最後的結果與GAN加入unit物件作比較。其中alpha matting作法參考自[此Github](https://github.com/MarcoForte/closed-form-matting)，並用以下方法合成：
``` python
import cv2
import numpy as np

fore = cv2.imread('fore1.jpg')
back = cv2.imread('1_in.jpg')
w, h, _ = back.shape
alpha = cv2.imread('fore1_mask.jpg', 0)

print(back.shape, fore.shape, alpha.shape)

output = np.zeros((w,h,3))
for x in range(w):
    for y in range(h):
        output[x, y, :] = fore[x, y, :] * (alpha[x, y]/255.) + back[x, y, :] * ((255 - alpha[x, y])/255.)
cv2.imwrite('output.jpg', output)
``` 

#### Comparison
這篇論文主要應用為將某個物件合成到圖片上，因此我們將其結果與利用GAN Dissection的DEMO進行加上物件的結果做比較，可以發現因為alpha matting主要是在處理前景跟後景分離及合成，因此在合成邊界的處理上較為細緻，但並不會影響圖片中的其他部分，而GAN Dissection在邊界上就容易出現瑕疵，但是會調整圖片的其他部分，讓整張圖看起來更無違和。

|Image|![](https://i.imgur.com/s0On7IR.jpg)|
|---|---|
|matting|![](https://i.imgur.com/k6qxRxU.jpg)|
|matting Mask|![](https://i.imgur.com/w2fWiAQ.jpg)|
|GAN|![](https://i.imgur.com/sxZ9JJK.jpg)




### [Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/pdf/1903.07291.pdf)

#### Introduction
這篇論文提出了適用於給定特定語義佈局（semantic layout），以合成照片的layer：spatially-adaptive normalization，通過學習到的spatially-adaptive變換，用語義佈局調整啟動函數，讓語義資訊在整個網絡中能夠有效傳播，避免流失。

#### Comparison
這篇論文主要應用為根據語義佈局合成照片，因此我們將其結果與利用GAN Dissection的DEMO進行圖片合成的結果做比較，不過，比較可惜的是，因為目前這篇論文尚未釋出DEMO或其他可供他人使用的實作，我們只能擷取其論文中所附的結果，觀察其優缺點，再與GAN Dissection進行比較，這兩者雖然都能夠達到良好的合成照片效果，但其實他們在作法上就有相當大的差異，其中，GAN Dissection主要目的是解析GAN，所以他是以一張照片為基礎，利用定義出負責某一類物件的unit來修改照片，這篇論文則是以語義佈局為基礎來修改照片，並且有特別優化合成照片的效果，在結果上，可以看到在比較單純的圖上，兩者都可以達到良好的效果，但是在較為複雜的圖中，Gan Dissection產生的圖容易發生一些扭曲或模糊，這篇論文則可以得到更清晰自然的結果，相較於過去其他的一些方法也更完整。

|Ground Truth|GAN Dissection|Ground Truth|GAN Dissection|
|---|---|---|---|
|![](https://i.imgur.com/hU5LiZv.png)|![](https://i.imgur.com/dHDYk3M.png)|![](https://i.imgur.com/jRPNWzf.png)|![](https://i.imgur.com/IxC6gkb.png)|


![](https://i.imgur.com/KG5XoQP.jpg)

