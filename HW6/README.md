ISA525700 Computer Vision for Visual Effects<br/>Homework 6
===

(請點擊縮圖以前往YouTube觀看影片)

## Our video

我們一共拍了兩個影片，第一個影片用來放置一些物件，觀察在不同的做法下，對於相機動態的分析情形，第二個影片則主要用來做特徵點的追蹤，製作以動作為依據的特效。

<a href="http://www.youtube.com/watch?feature=player_embedded&v=NykvRc1kfnk
" target="_blank"><img src="http://img.youtube.com/vi/NykvRc1kfnk/0.jpg" 
alt="Our video (fail to load the image, please click here to see the video)" width="240" height="180" border="10" /></a>

<a href="http://www.youtube.com/watch?feature=player_embedded&v=OPxcWSGy1ME
" target="_blank"><img src="http://img.youtube.com/vi/OPxcWSGy1ME/0.jpg" 
alt="Our video (fail to load the image, please click here to see the video)" width="240" height="180" border="10" /></a>

## Visual effects (with 3D model)

我們分別使用ORB-SLAM2和AE來做特效，首先，在ORB-SLAM2的部分，我們在卡納赫拉皮卡丘和小火龍的中間放入愛心並在加入一個正在飛的蝴蝶的3D模型；在AE的部分，我們追蹤玩偶位置加入對話框。

### ORB-SLAM2

ORB-SLAM2利用提取稀疏特徵點的方式來達到定位的效果，在這次作業的部分我們將我們所拍攝的影片作為mono_tum的輸入，希望透過特徵點的提取，將其輸出的相機外部參數作為影片加入特效...等的依據。

#### Build up and Run ORB-SLAM2
![](https://i.imgur.com/jeYo9ij.png)

#### 實作

1. 實作中許多影片都有失敗的問題，對於轉動的精確度較低，複雜運鏡的影片結果不準確、背景單一或是移動單一的影片容易有特徵點提取失敗、keyframe過少等問題，在經過一些測試後選擇使用現在這個影片。

#### Add visual effect add 3D model

在影片中加入蝴蝶的3D model及愛心image，實作步驟如下：
1. 利用video中獲得frame，30fps的影片則每秒取得30張
2. 如example中的輸入檔，生成Timestamp+檔名的input txt
3. 將兩者作為input，獲得camera參數包含timestamp, position, rotation
4. 因為keyframe過少，撰寫code針對pisition及rotation進行內插
5. 利用unity讀取上步驟結果，作為camera移動及轉動依據
6. 於場景中加入3D物件及image![](https://i.imgur.com/3KLxEmO.png)


#### 結果Video
<a href="http://www.youtube.com/watch?feature=player_embedded&v=wAc5LuGO5PQ
" target="_blank"><img src="http://img.youtube.com/vi/wAc5LuGO5PQ/0.jpg" 
alt="Our video (fail to load the image, please click here to see the video)" width="240" height="180" border="10" /></a>

### AE

在AE的部分，我們利用Tracker中的3D camera tracker，首先分析出相機的位移以及旋轉，接著利用取得的特徵點將愛心貼到小火龍和皮卡丘之間形成的平面上，如此即可達到我們想要的效果。

<a href="http://www.youtube.com/watch?feature=player_embedded&v=-KAuE1tAJY4
" target="_blank"><img src="http://img.youtube.com/vi/-KAuE1tAJY4/0.jpg" 
alt="Our video (fail to load the image, please click here to see the video)" width="240" height="180" border="10" /></a>

### Comparison

#### ORB_SLAM2
* 優點
    1. 可以將整個過程自動化
    2. 在應用上如AR等，若搭配穩定的旋轉資訊，如九軸感測，能夠達到很好的效果(如github上的unity SLAM的旋轉資訊及來自IMU

* 缺點
    1. 第一次build up複雜
    2. 特徵點抓取在不少情況下會失敗、不準確(複雜運鏡的影片結果不準確、背景單一或是移動單一)

#### AE
* 優點
    1. 介面人性化好操作
    2. 追蹤效果準確
    3. 特效及功能多
* 缺點
    1. 影片格式不佳(過長時)容易crash
## Special effects based on the pose information

這個部分我們利用AE來進行實作，利用Tracker中的Track Motion，分別以卡納赫拉伊布和卡納赫拉皮卡丘的鼻子為特徵點進行追蹤，並依據追蹤結果讓他們手上拿著花花，另外，花花還會噴出星星，其中，卡納赫拉伊布的移動比較平行，我們只追蹤Position，卡納赫拉皮卡丘則有微微旋轉，所以我們還追蹤了Rotation，從影片中可以看到追蹤非常準確，效果非常好。

<a href="http://www.youtube.com/watch?feature=player_embedded&v=66tIZ3vtJ6s
" target="_blank"><img src="http://img.youtube.com/vi/66tIZ3vtJ6s/0.jpg" 
alt="Our video (fail to load the image, please click here to see the video)" width="240" height="180" border="10" /></a>


## Assign
- [x]  5%  (Take videos by yourselves)
- [x]  10% (Make these visual effects with ORB-SLAM2)
- [x]  10% (Make these visual effects with any post-production software)
- [x]  10% (Compare above methods)
- [x]  10% (Make some special effects based on the pose information, such as rotating, zooming in or out)
- [x]  5%  (Insert a 3D model to your video)
- [x]  30% (Assistant) 
- [x]  20% (Mutual evaluation)
