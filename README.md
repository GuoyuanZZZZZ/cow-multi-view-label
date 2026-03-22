# cow-multi-view-label

跨平台（Windows / macOS）多相机二维/三维关键点标注演示程序，现已按模块拆分为清晰的工程结构，而不是单一 `app.py` 大文件。

## 目录结构

```text
multiview_labeler/
  core/
    constants.py      # 关键点/骨架/可见性常量
    models.py         # 标定、2D点、帧标注数据结构
    calibration.py    # 标定保存/加载/示例标定
    dataset.py        # 多相机图片/视频导入与同步
    annotation.py     # 标注数据管理、撤销重做、插值
    geometry.py       # 三角化、重投影、极线
    qc.py             # 重投影/骨长/对称性检查
  gui/
    views.py          # 2D视图与3D视图控件
    main_window.py    # 主窗口与交互逻辑
  demo/
    demo_builder.py   # 自动生成示例图片与标定
    bootstrap.py      # 组装演示数据集
  tools/
    exporters.py      # 2D/3D/QC 导出工具
app.py                # 顶层启动入口
```

## 已实现功能

- **多相机导入**
  - 支持按相机目录导入同步图片序列。
  - 支持多个视频抽帧导入为同步图片序列。
  - 当前演示按 `frame index` 同步，并保留 `sync_map`。
- **多相机标定**
  - `K`、`distCoeffs`、`R`、`t`、投影矩阵 `P = K[R|t]`。
  - JSON 保存 / 加载。
- **二维标注**
  - 点击落点。
  - 拖动修正。
  - `visible / occluded / absent`。
  - 撤销/重做、复制上一帧、插值。
  - 多视角同步骨架显示。
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
