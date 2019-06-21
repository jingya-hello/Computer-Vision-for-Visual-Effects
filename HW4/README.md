ISA525700 Computer Vision for Visual Effects<br/>Homework 4
===

## A sequence of moving-forward images in NTHU campus
我們拍攝了台達七樓711到715外的走廊，每三塊磁磚拍攝一次，並以天花板上的線進行對齊，共拍攝九張照片。

|1|2|3|
|---|---|---|
|![](https://i.imgur.com/KDrpTq5.jpg)|![](https://i.imgur.com/2pPvRcn.jpg)|![](https://i.imgur.com/6g2tlEk.jpg)|

|4|5|6|
|---|---|---|
|![](https://i.imgur.com/8RGm1pw.jpg)|![](https://i.imgur.com/bCrvq77.jpg)|![](https://i.imgur.com/LPvUSx5.jpg)|

|7|8|9|
|---|---|---|
|![](https://i.imgur.com/n05dRMI.jpg)|![](https://i.imgur.com/kc2h7rP.jpg)|![](https://i.imgur.com/hSe8PtY.jpg)|

## Feature extraction and matching results
我們將拍攝的九張照片兩兩配對(第一張和第二張、第二張和第三張......)後進行feature extraction和matching，步驟如下:
1. 初始化feature detection(這邊是使用ORB)
2. Detect SIFT points and extract SIFT descriptor
3. 利用BFMatcher比較descriptor並取得Hamming distance
4. 排序distance以取得最匹配的幾組(這邊取前十組)
最後取得結果如下:
![](https://i.imgur.com/mcN10bz.png)
![](https://i.imgur.com/ey2Kd9b.png)
![](https://i.imgur.com/oTYdt0N.png)
![](https://i.imgur.com/lN4jzj2.png)
![](https://i.imgur.com/56Fn7NL.png)
![](https://i.imgur.com/keCB1al.png)
![](https://i.imgur.com/FehRqyB.png)
![](https://i.imgur.com/8d93i3w.png)

## Image alignment
我們將兩兩配對的圖片第二張對第一張進行align。

|1-2|2-3|3-4|4-5|
|---|---|---|---|
|![](https://i.imgur.com/4Akbh5E.jpg)|![](https://i.imgur.com/Ainh1XE.jpg)|![](https://i.imgur.com/qqXs6BI.png)|![](https://i.imgur.com/rnsBnYH.jpg)|

|5-6|6-7|7-8|8-9|
|---|---|---|---|
|![](https://i.imgur.com/c6y15zP.jpg)|![](https://i.imgur.com/FD8Y18A.jpg)|![](https://i.imgur.com/NknqSel.jpg)|![](https://i.imgur.com/gvGnQSo.jpg)|

## Infinite zooming effect
這個部分我們嘗試了多種方式希望能讓infinite zooming effect的效果更好。

### Coding steps

- detect ORB features and compute descriptors and get key points
    - cv.ORB_create(MAX_FEATURES)
    - keypoints, descriptors = orb.detectAndCompute(image, None)
- match features
    - matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    -    matches = matcher.match(descriptors1, descriptors2, None)
- sort matches by score and store scores with higher value
- find homography and use it to warp perspective
    - h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    - Regresult = cv2.warpPerspective(im1, h, (width, height))

### Coding Method
- Code : use feature to align images
- Code : resize images to make walking effect
- Imovie : post-processing to balance color
- Imovie : post-processing to crop image to mask out the black area
- Code : use code to compose to video
- [![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/uwnE0OWCgCA/0.jpg)](https://youtu.be/uwnE0OWCgCA)

### imovie method
- Code : use feature to align images
- Imovie : resize images and compose by using imovie
- Imovie : post-processing to crop image to mask out the black area
- [![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/IpE5RzB1NP8/0.jpg)](https://youtu.be/IpE5RzB1NP8)

### Sliding window
我們有嘗試利用sliding window的方式去尋找兩兩配對的照片的align和前一張的哪個部分最相近，也就是說在第一張照片用不同大小的sliding window從左上到右下框出來的範圍和第二張照片進行SSIM的比較，以決定第一張照片要以哪裡為中心放大多少才能接第二張照片，但因為這個方式要跑非常久，目前只跑了第一組的一部分，下圖是目前的結果，藍色框框是目前分數最高的。
![](https://i.imgur.com/gbrKGlR.jpg)

## Different feature extrators
|ORB|SIFT|SURF|
|---|---|---|
|![](https://i.imgur.com/43Ukupv.jpg)|![](https://i.imgur.com/RiTeCUY.jpg)|![](https://i.imgur.com/0hiwfxx.jpg)|

三者因為演算法的不同而會抓取不同的特徵點，演算法簡介如下，另外，ORB是最快的，SIFT是最慢的。

- SIFT : 使用四個步驟取得特徵點，包含：
    - 利用DOG做可能的特征點檢測
    - 分析高斯差分算子的特性，找出受雜訊影響的可能特徵點，消除有高度邊緣影響但其位置不符合需求的特徵點
    - 特徵點以相鄰像素的梯度方向分佈作為指定方向參數，使特徵點描述子能以根據此方向來表示並具備旋轉不變性
    - 找到特徵點的位置、尺寸並賦予關鍵點方向後，將可確保其移動、縮放、旋轉的不變性

- SURF :
    - 基於SIFT的方法，利用Hessian matrix將計算簡化，使得速度加快

- ORB : FAST+BRIEF的改良版，藉以使速度有顯著提昇
    - 利用FAST(基於機器學習的快速角點特征檢測算法)來提取features
    - 利用BRIEF來得到描述子，並使用steered BRIEF的改進來達到旋转不變性

## Image processing
我們使用了下面幾種方式來讓我們的infinite zooming effect看起來效果更好:
- 濾鏡:將影片套用濾鏡，減少每張照片之間因光線而造成的色差影響。
- 裁減:將影片中因為align扭曲所造成黑色部分裁減掉避免突兀。
- 速度：微調圖片之間的速度來讓影片更順暢。


