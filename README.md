## Climax
- Climax是天气和气候科学的第一个基础模型。
- 简单，灵活，易于使用。
- 工作流应用于各种下游任务的示例，从天气预报到气候缩减。
- 支持高效可扩展的分布式训练，由PyTorch Lightning提供支持。

## 环境创建

```python
conda env create --file docker/environment.yml
conda activate climaX
```

## 导入包
```python
pip install -e .
```

## 数据集下载
https://dataserv.ub.tum.de/index.php/s/m1524895

## 代码运行
```python
python src/data_preprocessing/nc2np_equally_era5.py \
    --root_dir /mnt/data/5.625deg \
    --save_dir /mnt/data/5.625deg_npz \
    --start_train_year 1979 --start_val_year 2016 \
    --start_test_year 2017 --end_year 2019 --num_shards 8

python src/climax/global_forecast/train.py --config configs/global_forecast_climax.yaml \
    --trainer.strategy=ddp --trainer.devices=8 \
    --trainer.max_epochs=50 \
    --data.root_dir=/mnt/data/5.625deg_npz \
    --data.predict_range=72 --data.out_variables=['z_500','t_850','t2m'] \
    --data.batch_size=16 \
    --model.pretrained_path='https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt' \
    --model.lr=5e-7 --model.beta_1="0.9" --model.beta_2="0.99" \
    --model.weight_decay=1e-5
```