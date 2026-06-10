# 数据目录约定

当前阶段不随仓库提交公开数据集原始文件。

## C-MAPSS

下载后放置到：

```text
data/raw/cmapss/
  train_FD001.txt
  test_FD001.txt
  RUL_FD001.txt
  train_FD002.txt
  test_FD002.txt
  RUL_FD002.txt
  train_FD003.txt
  test_FD003.txt
  RUL_FD003.txt
  train_FD004.txt
  test_FD004.txt
  RUL_FD004.txt
```

对应配置：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/train_baseline.py --config configs/cmapss_fd001_ptcn.yaml
```

## XJTU-SY Bearing

后续下载后放置到：

```text
data/raw/xjtu_sy/
```

第一版不直接迁移原始高频振动，而是通过 `extract_low_freq_hi(signal_window)` 提取低频健康指标。

## NASA Battery Aging

后续下载后放置到：

```text
data/raw/nasa_battery/
```

## 自建航天仿真数据集

后续生成后放置到：

```text
data/simulated/reaction_wheel/
data/simulated/satellite_battery/
```
