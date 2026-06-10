# XA-202608 数据放置说明

本目录只放本地数据，不提交到代码仓库。

## 1. C-MAPSS

放到：

```text
data/raw/cmapss/
```

需要文件直接放在该目录下，不要再套一层文件夹：

```text
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

你截图里的 C-MAPSS 文件就是我们需要的，可以直接复制这些 txt 文件。

## 2. XJTU-SY Bearing

放到：

```text
data/raw/xjtu_sy/
```

下载全部分卷：

```text
XJTU-SY_Bearing_Datasets.part01.rar
XJTU-SY_Bearing_Datasets.part02.rar
XJTU-SY_Bearing_Datasets.part03.rar
XJTU-SY_Bearing_Datasets.part04.rar
XJTU-SY_Bearing_Datasets.part05.rar
XJTU-SY_Bearing_Datasets.part06.rar
```

全部放在同一个临时文件夹后，用 WinRAR 从 `part01.rar` 解压。解压后的工况文件夹或 `Bearing*` 文件夹放到 `data/raw/xjtu_sy/` 下。

推荐最终结构类似：

```text
data/raw/xjtu_sy/
  35Hz12kN/
    Bearing1_1/
      1.csv
      2.csv
      ...
  37.5Hz11kN/
    Bearing2_1/
      1.csv
      2.csv
      ...
  40Hz10kN/
    Bearing3_1/
      1.csv
      2.csv
      ...
```

若解压后外面多一层 `XJTU-SY_Bearing_Datasets/` 或 `Data/` 也可以，当前预处理脚本会递归查找 `Bearing*` 文件夹。

预处理命令：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/prepare_xjtu_hi.py --root_dir data/raw/xjtu_sy --output_path data/processed/xjtu_sy_hi.npz
```

## 3. NASA Battery Aging

原始数据放到：

```text
data/raw/nasa_battery/
```

第一版训练脚本读取处理后的 CSV：

```text
data/processed/nasa_battery.csv
```

CSV 至少建议包含这些列：

```text
unit_id, cycle, voltage, current, temperature, capacity, internal_resistance
```

如果暂时没有 `internal_resistance`，后续可以先调整配置或补预处理脚本。

## 4. 自建航天仿真数据

这一阶段不用下载，后续由我们的 Python 脚本生成：

```text
data/simulated/reaction_wheel/
data/simulated/satellite_battery/
```

## 5. 检查命令

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/check_data.py --all
```
