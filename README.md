# fileclipper
识别纸张边缘算法，运行clipper.py 可以达到以下的效果

##### 识别前
![识别前](https://github.com/Wangzg123/fileclipper/blob/master/test.jpg?raw=true)

##### 识别后
![识别后](https://github.com/Wangzg123/fileclipper/blob/master/result.jpg?raw=true)


# 票据识别前处理（深背景，找浅前景矩形并透视转换）

把XXXXXX_clipper.py中的两个全局变量进行修改：
**目标图片大小**
TARGET_H = 672
TARGET_W = 1369

## skimage方式接口调用
```python
from skimage_clipper import main_img_preprocess
from skimage import io

origin_img = io.imread('your_img_path')
crop_img = main_img_preprocess(origin_img)
# you can save img by below code
# io.imsave('croped.jpg', crop_img)
```

## opencv方式接口调用（opencv python 4.X）
```python
from opencv_clipper import main_img_preprocess
import cv2

origin_img = cv2.imread('your_img_path')
crop_img = main_img_preprocess(origin_img)
# you can save img by below code
# cv2.imwrite('croped.jpg', crop_img)
```

# 其他说明
同样的图片，opencv方式比skimage方式要快很多。
典型的（1400*700左右的图片）：
opencv方式:100 ms
skimage方式：1000 ms
