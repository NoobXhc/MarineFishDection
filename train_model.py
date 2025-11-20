from ultralytics import YOLO


def train_yolov8():
    model = YOLO('yolov8n.pt')

    # 训练模型 - 修正后的参数
    results = model.train(
        data='datasets/data.yaml',
        epochs=150,
        patience=20,
        batch=16,
        imgsz=640,
        device='0',
        workers=4,
        project='runs/detect',
        name='marine_fish_train',
        exist_ok=True,

        # 数据增强优化 - 使用YOLOv8实际支持的参数
        augment=True,
        hsv_h=0.015,  # 色调增强，模拟水下颜色变化
        hsv_s=0.7,  # 大幅增加饱和度增强，应对水下颜色失真
        hsv_v=0.4,  # 亮度增强，应对水下光照不均
        degrees=10.0,  # 旋转角度
        translate=0.1,  # 平移
        scale=0.5,  # 缩放
        shear=5.0,  # 剪切变换
        perspective=0.0005,  # 透视变换
        flipud=0.3,  # 增加垂直翻转概率
        fliplr=0.5,  # 水平翻转

        # 水下特定增强
        mosaic=1.0,  # 马赛克增强
        mixup=0.2,  # mixup增强
        copy_paste=0.2,  # 复制粘贴，模拟鱼群重叠
        erasing=0.3,  # 随机擦除，应对部分遮挡

        # 针对水下模糊的替代方案
        # 使用更强的几何变换来模拟模糊效果
        # 或者可以在数据预处理阶段添加模糊增强

        # 学习率优化
        lr0=0.01,
        lrf=0.1,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 优化器调整
        optimizer='AdamW',
        weight_decay=0.0005,
        momentum=0.9,

        # 防止过拟合
        dropout=0.2,
        amp=True,

        # 损失函数调整
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # 验证设置
        val=True,
        save_json=True,
        plots=True
    )

    print("训练完成！")


if __name__ == '__main__':
    train_yolov8()

""""
def train_yolov8():

    model = YOLO('yolov8n.pt')

    # 训练模型
    results = model.train(
        data='datasets/data.yaml',
        epochs=100,  # 增加训练轮数
        patience=15,  # 增加早停耐心值
        batch=24,  # 由于imgsz增大，适当减小batch
        imgsz=640,  # 平衡精度和速度
        lr0=0.01,
        device='0',
        workers=4,  # 根据CPU核心数调整
        project='runs/detect',
        name='BVN_train',
        exist_ok=True,

        # 关键参数优化
        rect=True,  # 保持开启，处理不同尺寸
        augment=True,  # 数据增强
        mosaic=0.8,  # 增加马赛克增强，帮助学习部分遮挡
        mixup=0.2,  # 增加mixup，增强颜色不变性
        copy_paste=0.1,  # 新增：复制粘贴增强，模拟海葵环境
        erasing=0.2,  # 增加随机擦除，应对遮挡

        # 小目标检测优化
        multi_scale=False,  # 关闭以节省时间
        scale=0.5,  # 数据增强的缩放范围
        fliplr=0.5,  # 水平翻转
        flipud=0.1,  # 垂直翻转，模拟不同角度

        # 学习率调整
        warmup_epochs=3,  # 学习率预热
        warmup_momentum=0.8,
        lrf=0.1,  # 最终学习率倍数

        # 其他优化
        amp=True,  # 开启混合精度训练
        dropout=0.1,  # 防止过拟合
        weight_decay=0.0005,  # 权重衰减
    )

    print("训练完成！")

"""
