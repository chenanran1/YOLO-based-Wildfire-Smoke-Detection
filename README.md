# 火灾烟雾检测项目

基于YOLOv5的火灾和烟雾检测系统，支持实时检测和批量处理。

## 项目简介

本项目使用YOLOv5模型进行火灾和烟雾的检测，支持：
- 🖼️ 单张图片检测
- 🎥 视频文件检测  
- 📹 实时摄像头检测
- 🖥️ CPU/GPU双模式推理
- 🎯 实时显示检测结果

## 环境要求

### 基础环境
- Python 3.7+
- PyTorch 1.13.0+
- OpenCV 4.10.0+

### GPU环境（推荐）
- NVIDIA GPU
- CUDA 11.6+
- cuDNN

## 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装支持GUI的OpenCV（用于实时显示）
pip install opencv-contrib-python
```

## 数据集准备

### 数据格式
- 图片：支持 `.jpg`, `.jpeg`, `.png`, `.bmp` 等格式
- 标注：YOLO格式的 `.txt` 文件
- 类别：fire（火灾）, smoke（烟雾）

### 数据集结构
```
fire_smoke/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 模型训练

### 1. 准备配置文件
确保 `fire_smoke/data.yaml` 文件配置正确：
```yaml
train: fire_smoke/train/images
val: fire_smoke/valid/images
nc: 2  # 类别数量
names: ['fire', 'smoke']  # 类别名称
```

### 2. 开始训练
```bash
# 基础训练
python train.py --data fire_smoke/data.yaml --weights yolo-facev2s-preweight.pt  --cfg models/yolov5s_v2_RFEM_MultiSEAM.yaml --epochs 150 --batch-size 16

# 使用GPU训练（推荐）
python train.py --data fire_smoke/data.yaml --weights yolo-facev2s-preweight.pt  --cfg models/yolov5s_v2_RFEM_MultiSEAM.yaml --epochs 150 --batch-size 16 --device 0

# 自定义参数训练
python train.py \
    --data fire_smoke/data.yaml \
    --weights yolov5s.pt \
    --epochs 150 \
    --batch-size 16 \
    --img-size 640 \
    --device 0 \
    --project runs/train \
    --name fire_smoke_model
```

### 3. 训练参数说明
- `--data`: 数据集配置文件路径
- `--weights`: 预训练权重文件
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--img-size`: 输入图像尺寸
- `--device`: 训练设备（0为GPU，cpu为CPU）
- `--project`: 项目保存目录
- `--name`: 实验名称

## 模型检测

### 1. 单张图片检测
```bash
# CPU检测
python detect.py --weights runs/train/fire_smoke_150_epochs3/weights/best.pt --source path/to/image.jpg

# GPU检测（推荐）
python detect.py --weights runs/train/fire_smoke_150_epochs3/weights/best.pt --source path/to/image.jpg --device 0

# 实时显示检测结果
python detect.py --weights runs/train/fire_smoke_150_epochs3/weights/best.pt --source path/to/image.jpg --device 0 --view-img
```

### 2. 批量图片检测
```bash
# 检测文件夹中的所有图片
python detect.py --weights runs/train/fire_smoke_150_epochs3/weights/best.pt --source path/to/images/ --device 0
```

### 3. 视频检测
```bash
# 视频文件检测
python detect.py --weights runs/train/fire_smoke_150_epochs3/weights/best.pt --source path/to/video.mp4 --device 0

# 实时显示视频检测结果
python detect.py --weights runs/train/fire_smoke_150_epochs3/weights/best.pt --source path/to/video.mp4 --device 0 --view-img
```

### 4. 摄像头实时检测
```bash
# 使用默认摄像头（通常是0）
python detect.py --weights runs/train/fire_smoke_150_epochs3/weights/best.pt --source 0 --device 0 --view-img
```

### 5. 检测参数说明
- `--weights`: 训练好的模型权重文件
- `--source`: 输入源（图片路径/视频路径/摄像头索引）
- `--device`: 推理设备（0为GPU，cpu为CPU）
- `--view-img`: 实时显示检测结果
- `--img-size`: 推理图像尺寸（默认640）
- `--conf-thres`: 置信度阈值（默认0.25）
- `--iou-thres`: NMS阈值（默认0.45）
- `--nosave`: 不保存检测结果
- `--save-txt`: 保存检测结果为txt文件

## 结果说明

### 训练结果
训练完成后，结果保存在 `runs/train/` 目录下：
- `weights/best.pt`: 最佳权重文件
- `weights/last.pt`: 最后一轮权重文件
- 训练日志和图表

### 检测结果
检测完成后，结果保存在 `runs/detect/exp*/` 目录下：
- 带检测框的图片/视频
- 检测标签文件（如果使用 `--save-txt`）

## 性能优化建议

### 1. GPU加速
- 使用 `--device 0` 启用GPU推理
- 确保安装了CUDA版本的PyTorch

### 2. 推理速度优化
- 降低输入分辨率：`--img-size 320`
- 调整置信度阈值：`--conf-thres 0.5`
- 使用更小的模型（如yolov5n）

### 3. 检测精度优化
- 提高输入分辨率：`--img-size 1280`
- 降低置信度阈值：`--conf-thres 0.1`
- 使用更大的模型（如yolov5l, yolov5x）

## 常见问题

### 1. CUDA不可用
```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回False，请安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. OpenCV显示问题
```bash
# 安装支持GUI的OpenCV
pip install opencv-contrib-python
```

### 3. 内存不足
- 减小batch-size
- 降低图像分辨率
- 使用更小的模型

## 项目结构
```
YOLO-FaceV2-master/
├── train.py              # 训练脚本
├── detect.py             # 检测脚本
├── models/               # 模型定义
├── utils/                # 工具函数
├── data/                 # 数据集配置
├── fire_smoke/           # 火灾烟雾数据集
├── fire_smoke_new/       # 新数据集
├── runs/                 # 训练和检测结果
└── requirements.txt      # 依赖包列表
```

## 许可证

本项目基于MIT许可证开源。

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。
