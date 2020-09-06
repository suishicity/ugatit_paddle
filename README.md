# UGATIT-PADDLE

## 介绍
使用 PaddlePaddle 编写的 UGATIT 代码，原始 Pytorch 实现参见 [UGATIT-Pytorch](https://github.com/znxlwm/UGATIT-pytorch)。
AIStudio 公开项目地址 https://aistudio.baidu.com/aistudio/projectdetail/721668， 项目有人像到动漫的模型权重。

（AIStudio 加载生成版本再退出保存居然直接将草稿覆盖了，晕，训练了几天的权重一下子全都没了，只剩下一个 genA2B 的权重。）

## 依赖库
```
paddlepaddle-gpu==1.8.3.post97
easydict
PyYAML
```

## 训练
从零开始训练
```
python main.py --config config/ai_ugatit.yaml
```
从 last_checkpoint 中恢复训练
```
python main.py --config config/ai_ugatit.yaml --resume
```

## 测试
```
python work/ugatit/predict.py \
--weight_path /home/aistudio/work/train_hist/genAB_last.pdparams \
--input /home/aistudio/data/selfi2anime/testA \
--output fakeA2B
```


