# cow-multi-view-label

跨平台（Windows / macOS）多相机二维/三维关键点标注演示程序，现已按模块拆分为清晰的工程结构，而不是单一 `app.py` 大文件。

## 目录结构

```text
multiview_labeler/
  core/
    constants.py      # 关键点/骨架/可见性常量
    models.py         # 标定、2D点、帧标注数据结构
    calibration.py    # 标定保存/加载/示例标定
    calibration_tool.py # OpenCV 棋盘格标定流程
    dataset.py        # 多相机图片/视频导入与同步
    annotation.py     # 标注数据管理、撤销重做、插值
    geometry.py       # 三角化、重投影、极线
    qc.py             # 重投影/骨长/对称性检查
  gui/
    views.py          # 2D视图与3D视图控件
    calibration_dialog.py # GUI 标定面板
    pages.py          # Project / Annotation / Calibration / Export 页面
    main_window.py    # 主窗口与交互逻辑
  demo/
    demo_builder.py   # 自动生成示例图片与标定
    bootstrap.py      # 组装演示数据集
  tools/
    exporters.py      # 2D/3D/QC 导出工具
    frame_sampler.py  # representative frames 推荐
app.py                # 顶层启动入口
```

## 已实现功能

- **多相机导入**
  - 支持按相机目录导入同步图片序列。
  - 支持多个视频抽帧导入为同步图片序列。
  - 当前演示按 `frame index` 同步，并保留 `sync_map`。
  - 新增 `Import` 页面作为多视频导入向导。
- **多相机标定**
  - `K`、`distCoeffs`、`R`、`t`、投影矩阵 `P = K[R|t]`。
  - JSON 保存 / 加载。
  - 新增基于 OpenCV 棋盘格的 GUI 标定入口，可运行内参/外参估计。
- **二维标注**
  - 点击落点。
  - 拖动修正。
  - 鼠标滚轮缩放、工具栏整体放大缩小、适配窗口。
  - 关键点检查表（每个点当前有多少视角完成、是否已生成 3D）。
  - 实例级标注：支持切换 `Instance` 对多只动物分别标注。
  - `visible / occluded / absent`。
  - 撤销/重做、复制上一帧、插值。
  - 多视角同步骨架显示。
- **页面 / 工作区**
  - `Project` 页面：项目概览、相机数量、帧数、页面结构。
  - `Import` 页面：多视频导入向导。
  - `Frames` 页面：代表帧推荐与快速跳转。
  - `Annotation` 页面：多视角标注 + 关键点检查器 + 实时详情。
  - `Constraints` 页面：骨长/真实测量值约束编辑与状态查看。
  - `Calibration` 页面：标定工作区和状态面板。
  - `Export/QC` 页面：QC 预览与导出入口。
- **代表帧辅助**
  - 参考 JARVIS AnnotationTool README 中提到的 representative frames 工作流。
  - 自动根据图像变化 + 均匀采样混合推荐更有代表性的帧，帮助优先标注关键姿态。
- **约束编辑**
  - 支持编辑骨段目标长度和容差。
  - 支持根据真实测量值对当前 3D 骨长进行对比检查。
- **实时 3D**
  - 至少两个视角标注后实时三角化。
  - 重投影到全部视角。
  - 3D 骨架显示。
- **几何辅助**
  - 极线显示。
  - 未标视角给出推荐 2D 坐标。
- **质量检查**
  - 重投影误差。
  - 骨段长度。
  - 左右对称性。
  - 异常点高亮。
- **导出**
  - 2D JSON/CSV。
  - 3D JSON/CSV。
  - QC 报告 JSON。

## 运行方式

```bash
pip install -r requirements.txt
python app.py
```

程序会在 `demo_data/` 下自动生成 3 个相机、20 帧的同步示例图片与标定文件，开箱即用。
