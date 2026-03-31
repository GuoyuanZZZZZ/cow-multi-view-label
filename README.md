# cow-multi-view-label

一个轻量的输入预处理脚本：让标定流程同时支持 **图片** 和 **视频** 输入。

## 功能

- 输入支持：
  - 单张图片
  - 图片文件夹
  - 视频文件（自动抽帧）
- 输出：
  - 抽帧图片（默认写到 `prepared_frames/`）
  - 清单文件 `manifest.txt`（每行一个图片绝对路径）

## 依赖

```bash
pip install opencv-python
```

## 用法

```bash
python calibration_input.py <输入1> <输入2> ... [--output-dir prepared_frames] [--manifest prepared_frames/manifest.txt] [--video-step 10]
```

示例：

```bash
# 同时输入图片目录 + 视频
python calibration_input.py ./images ./camera1.mp4 --video-step 15
```

输出的 `manifest.txt` 可直接对接你现有的标定流程（按行读取图片路径）。
