# fileclipper
识别纸张边缘算法，运行opencv_clipper.py 可以达到以下的效果

##### 识别前
![识别前](https://github.com/Wangzg123/fileclipper/blob/master/test.jpg?raw=true)

##### 识别后
![识别后](https://github.com/Wangzg123/fileclipper/blob/master/result.jpg?raw=true)


票据识别（深背景，找浅前景矩形）
接口调用方法：
```python
from skimage_clipper import main_img_preprocess
from skimage import io

origin_img = io.imread('your_img_path')
crop_img = main_img_preprocess(origin_img)
# you can save img by below code
# io.imsave('croped.jpg', crop_img)
```
