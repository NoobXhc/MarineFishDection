## README.md

基于YOLOv8深度学习模型的海洋鱼类智能识别系统，提供直观的PyQt5图形界面，支持图片、视频和实时摄像头检测。

界面 PyQt5   模型 YOLOv8    Python=3.9

## 核心功能

### 图片检测
- 支持单张/多张图片批量检测
- 自动导航：上一张/下一张切换
- 实时结果显示与保存
- 支持格式：JPG, JPEG, PNG, BMP, WEBP, TIFF, TIF

### 视频检测
- 支持多种视频格式：MP4, AVI, MOV, MKV, WMV, FLV
- 实时进度显示和控制
- 暂停/继续播放功能
- 保存任意帧为图片

### 实时摄像头
- 实时视频流处理
- FPS性能监控
- 帧捕获和保存
- 低延迟检测

### 识别设置
- 可调节置信度阈值（0.1-0.9）
- 实时模型状态监控
- 检测日志记录
- 统计信息显示

## 可识别物种

系统可精准识别以下13种海洋鱼类：

| 英文名称      | 中文名称 | 英文名称               | 中文名称   |
| :------------ | :------- | :--------------------- | :--------- |
| AngelFish     | 神仙鱼   | RibbonedSweetlips      | 带纹胡椒鲷 |
| BlueTang      | 蓝唐王鱼 | ThreeStripedDamselfish | 三线雀鲷   |
| ButterflyFish | 蝴蝶鱼   | YellowCichlid          | 黄色慈鲷   |
| ClownFish     | 小丑鱼   | YellowTang             | 黄三角倒吊 |
| GoldFish      | 金鱼     | ZebraFish              | 斑马鱼     |
| Gourami       | 丝足鱼   | MorishIdol             | 镰鱼       |
| PlatyFish     | 剑尾鱼   |                        |            |

## 系统要求

### 硬件要求
- **处理器**: Intel Core i5 或同等性能以上
- **内存**: 8GB RAM 或以上
- **显卡**: 集成显卡
- **存储空间**: 至少2GB可用空间

### 软件环境
- **操作系统**: Windows 10/11, Linux, macOS
- **Python版本**: 3.8, 3.9, 3.10, 3.11
- **CUDA版本**: 11.8 或 12.6 (推荐)
- **包管理**: Miniconda 或 pip

## 快速开始

 
方法一  环境安装
```bash
# 创建conda环境
conda create -n ultralytics python=3.9
conda activate ultralytics

# 安装PyTorch (CUDA 12.6版本)
conda install pytorch torchvision torchaudio pytorch-cuda=12.6 -c pytorch -c nvidia

# 安装项目依赖
pip install -r requirements.txt
```

```
## 项目结构

```text
PythonProject/
│
├── output/         # 检测结果输出目录
├── best.pt      # 训练好的模型权重
├── app.py                   # 主应用程序
├── train_model.py           # 模型训练脚本
└── requirements.txt         # 项目依赖
```

方法二  使用可执行文件

```text
1.下载完整的程序包 FishDection
2.解压到任意文件夹
3.确保以下文件在同一目录下：
	FishDetection.exe (主程序)
	best.pt (AI模型文件)
	sign.ico (图标文件，可选)
4.FishDetection.exe 启动程序
### 运行程序

```

## 🎮 使用指南

### 首次运行准备

1. 确保模型文件路径正确：`best.pt`
2. 系统会自动创建输出目录 `output/`

### 操作流程

1. **启动程序**: 运行 `python app.py`
2. **选择模式**: 在主菜单选择检测类型
3. **加载文件**: 选择要检测的图片/视频或开启摄像头
4. **设置参数**: 在设置页面调整置信度阈值（默认0.4）
5. **开始检测**: 点击"开始检测"按钮
6. **查看结果**: 在右侧面板查看检测结果
7. **保存结果**: 点击保存按钮保存检测结果

### 模型训练

如需重新训练或微调模型：

```
python train_model.py
```

## 技术特性

### 界面设计

- **现代化UI**: 基于PyQt5的现代化界面设计
- **响应式布局**: 支持窗口大小调整
- **多页面导航**: 流畅的页面切换体验
- **实时状态**: 实时显示检测状态和统计信息

### 性能优化

- **多线程处理**: 视频和摄像头使用独立线程，避免界面卡顿
- **GPU加速**: 支持CUDA加速推理
- **内存管理**: 自动释放资源，防止内存泄漏
- **格式兼容**: 支持多种图像和视频格式

### 检测算法

- **YOLOv8架构**: 最新的YOLO检测算法
- **实时性能**: 高帧率实时检测
- **高精度**: 针对海洋鱼类优化的检测精度
- **可调节阈值**: 动态调整检测灵敏度

## 训练参数

模型训练采用以下优化参数：

```python
# 关键训练参数
epochs=150, batch=16, imgsz=640
optimizer='AdamW', lr0=0.01
# 针对水下环境的数据增强
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
mosaic=1.0, mixup=0.2, copy_paste=0.2
```
