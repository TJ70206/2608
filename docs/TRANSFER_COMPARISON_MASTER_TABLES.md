# 第一/第二迁移完整对比表

生成时间：2026-06-06

严格无监督迁移的同口径表格已单独整理在 `docs/STRICT_UNSUPERVISED_TRANSFER_COMPARISON.md`。该表排除了 target-only、fine-tune、target-sup、监督上界、集成模型和调参变体，并且第二迁移补齐了与第一迁移完全对应的 P-TCN/LSTM/GRU/Transformer/AD-TCN-MSC-DIM 基线。

指标方向：`RMSE`、`MAE`、`NASA`、`LastRMSE`、`Last5RMSE` 越低越好；`RA`、`a0.5`、`a0.8` 越高越好。表格按测试集 `RMSE` 升序排列。

关于监督上界：本项目当前的 `target-only` 基线只使用目标域 train split 标签训练、val split 选型、test split 最终评估，因此不是标签泄露。但它属于目标域监督参考基线/监督上限附近的参照，不应与严格无监督迁移结果混为同一类结论。若使用 test 标签训练、调参或融合权重选择，则属于泄露，不能采用。

## 第一迁移：XJTU-SY 轴承源域 -> reaction_wheel_sim 反作用飞轮目标域

| Rank | Method | RMSE | MAE | NASA | RA | a0.5 | a0.8 | LastRMSE | Last5RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Target-only Transformer | 0.0505 | 0.0400 | 2.6737 | 0.8343 | 0.9878 | 0.3765 | 0.0619 | 0.0800 |
| 2 | Target-only GRU | 0.0518 | 0.0406 | 2.7637 | 0.8323 | 0.9756 | 0.4000 | 0.0604 | 0.0848 |
| 3 | Target-only AD-TCN-MSC-DIM | 0.0518 | 0.0403 | 2.7223 | 0.8384 | 0.9878 | 0.4941 | 0.0631 | 0.0846 |
| 4 | Target-only P-SA-MCD | 0.0521 | 0.0403 | 2.7933 | 0.8301 | 0.9756 | 0.4353 | 0.0633 | 0.0835 |
| 5 | Ensemble LSTM target + AD-TCN transfer | 0.0526 | 0.0413 | 2.7768 | 0.8327 | 0.9390 | 0.4706 | 0.0716 | 0.0911 |
| 6 | Target-only LSTM | 0.0526 | 0.0413 | 2.7768 | 0.8327 | 0.9390 | 0.4706 | 0.0716 | 0.0911 |
| 7 | Target-only P-TCN | 0.0528 | 0.0402 | 2.7461 | 0.8345 | 0.9512 | 0.4000 | 0.0470 | 0.0764 |
| 8 | P-SA-MCD stage LMMD + pseudo-time | 0.1943 | 0.1535 | 10.6761 | 0.6024 | 0.2561 | 0.2000 | 0.2134 | 0.2079 |
| 9 | P-SA-MCD stage LMMD w=0.003 | 0.2046 | 0.1603 | 11.0311 | 0.5950 | 0.2073 | 0.2235 | 0.1880 | 0.1838 |
| 10 | P-SA-MCD stage LMMD default | 0.2048 | 0.1628 | 11.3622 | 0.5841 | 0.2927 | 0.1176 | 0.2698 | 0.2429 |
| 11 | P-SA-MCD stage LMMD w=0.03 | 0.2163 | 0.1686 | 11.5314 | 0.5762 | 0.1951 | 0.1412 | 0.2105 | 0.2074 |
| 12 | P-SA-MCD stage LMMD w=0.1 | 0.2206 | 0.1744 | 12.3061 | 0.5576 | 0.3415 | 0.0588 | 0.2373 | 0.2408 |
| 13 | P-SA-MCD stage LMMD w=0.001 | 0.2302 | 0.1832 | 12.8440 | 0.5533 | 0.2073 | 0.1765 | 0.2185 | 0.2452 |
| 14 | AD-TCN-MSC-DIM stage LMMD + pseudo-time | 0.2527 | 0.1998 | 14.5448 | 0.5327 | 0.2561 | 0.1294 | 0.3669 | 0.3843 |
| 15 | LSTM stage LMMD + pseudo-time | 0.2656 | 0.2289 | 16.4421 | 0.4985 | 0.7561 | 0.0000 | 0.4684 | 0.4691 |
| 16 | Transformer stage LMMD + pseudo-time | 0.2729 | 0.2345 | 16.7379 | 0.4957 | 0.9268 | 0.0000 | 0.4815 | 0.4756 |
| 17 | GRU stage LMMD + pseudo-time | 0.2733 | 0.2300 | 16.4724 | 0.5056 | 0.6951 | 0.0118 | 0.4708 | 0.4720 |
| 18 | P-TCN global MMD | 0.2779 | 0.2375 | 16.9778 | 0.4934 | 0.7683 | 0.0118 | 0.4688 | 0.4723 |
| 19 | P-TCN stage LMMD + pseudo-time | 0.2792 | 0.2361 | 16.8536 | 0.4941 | 0.5366 | 0.0118 | 0.4682 | 0.4705 |
| 20 | P-SA-MCD global MMD | 0.2799 | 0.2391 | 17.1005 | 0.4897 | 0.7927 | 0.0000 | 0.4854 | 0.4807 |
| 21 | P-SA-MCD source-only | 0.2831 | 0.2412 | 16.9461 | 0.4870 | 0.6463 | 0.0000 | 0.4981 | 0.4530 |
| 22 | P-TCN source-only | 0.3597 | 0.2911 | 20.6859 | 0.4428 | 0.3902 | 0.0471 | 0.5262 | 0.5165 |

结论：第一迁移中，严格迁移组内 `P-SA-MCD stage LMMD + pseudo-time` 最好，显著优于 source-only/global-MMD/P-TCN/LSTM/GRU/Transformer 迁移；但若允许目标域有标签训练，`target-only Transformer/GRU/AD-TCN/P-SA-MCD` 明显更强。`LSTM target-only + AD-TCN transfer` 融合由验证集选择到 `LSTM=1.0, AD-TCN transfer=0.0`，说明该 AD-TCN 严格迁移分量没有带来增益。

## 第二迁移：NASA Battery 源域 -> satellite_battery_sim 卫星电池目标域

| Rank | Method | RMSE | MAE | NASA | RA | a0.5 | a0.8 | LastRMSE | Last5RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Ensemble LSTM target + AD-TCN transfer | 0.1665 | 0.1226 | 34.1397 | 0.6762 | 0.5719 | 0.3007 | 0.2090 | 0.2171 |
| 2 | Target-only LSTM | 0.1716 | 0.1245 | 34.0621 | 0.6742 | 0.4415 | 0.3682 | 0.2342 | 0.2428 |
| 3 | Target-only AD-TCN-MSC-DIM | 0.1761 | 0.1313 | 36.7394 | 0.6507 | 0.4983 | 0.2365 | 0.2110 | 0.2202 |
| 4 | AD-TCN-MSC-DIM stage LMMD + target-sup | 0.1825 | 0.1364 | 38.1620 | 0.6538 | 0.5151 | 0.2568 | 0.1993 | 0.2071 |
| 5 | P-SA-MCD source pretrain + target fine-tune | 0.1876 | 0.1398 | 39.2209 | 0.6418 | 0.4582 | 0.2432 | 0.2287 | 0.2406 |
| 6 | P-SA-MCD stage LMMD + target-sup src0.02 w=0.0005 | 0.1890 | 0.1407 | 38.6074 | 0.6456 | 0.3946 | 0.2635 | 0.2143 | 0.2301 |
| 7 | P-SA-MCD stage LMMD + target-sup src0.05 w=0.0005 | 0.1897 | 0.1413 | 39.1724 | 0.6447 | 0.4080 | 0.2297 | 0.2283 | 0.2424 |
| 8 | P-SA-MCD stage LMMD + target-sup late | 0.1925 | 0.1463 | 40.2384 | 0.6352 | 0.3946 | 0.3007 | 0.2032 | 0.2163 |
| 9 | Target-only P-SA-MCD-TCN | 0.1938 | 0.1480 | 41.0521 | 0.6294 | 0.3545 | 0.2770 | 0.2255 | 0.2383 |
| 10 | LSTM source pretrain + target fine-tune | 0.1941 | 0.1479 | 38.9220 | 0.6481 | 0.3813 | 0.4696 | 0.2422 | 0.2496 |
| 11 | P-SA-MCD stage LMMD + target-sup src0.01 w=0.0005 | 0.1947 | 0.1416 | 40.2415 | 0.6465 | 0.4348 | 0.3176 | 0.2630 | 0.2744 |
| 12 | Target-only P-SA-MCD late-head | 0.1984 | 0.1468 | 40.5393 | 0.6400 | 0.3813 | 0.3142 | 0.2212 | 0.2344 |
| 13 | P-SA-MCD stage LMMD + target-sup src0.02 w=0.0002 | 0.1992 | 0.1502 | 42.7022 | 0.6232 | 0.3612 | 0.2973 | 0.2640 | 0.2746 |
| 14 | Target-only GRU | 0.2025 | 0.1515 | 42.9475 | 0.6201 | 0.4381 | 0.2297 | 0.2823 | 0.2874 |
| 15 | Target-only Transformer | 0.2048 | 0.1587 | 42.2230 | 0.6137 | 0.3043 | 0.0946 | 0.2113 | 0.2225 |
| 16 | LSTM stage LMMD + target-sup | 0.2058 | 0.1625 | 40.5695 | 0.6319 | 0.3478 | 0.2872 | 0.2155 | 0.2171 |
| 17 | P-SA-MCD stage LMMD + target-sup w=0.003 | 0.2118 | 0.1613 | 44.8689 | 0.6053 | 0.3077 | 0.1115 | 0.2512 | 0.2662 |
| 18 | Target-only P-TCN | 0.2327 | 0.1826 | 48.2141 | 0.5814 | 0.2274 | 0.1554 | 0.2135 | 0.2210 |
| 19 | P-TCN stage LMMD + target-sup | 0.2394 | 0.1926 | 51.6104 | 0.5542 | 0.2074 | 0.1655 | 0.2065 | 0.2166 |
| 20 | Target-only Leaky-ESN | 0.2883 | 0.2269 | 63.8455 | 0.5146 | 0.3478 | 0.0811 | 0.4070 | 0.4193 |
| 21 | NASA stage LMMD unsupervised | 0.3580 | 0.2919 | 72.7928 | 0.4426 | 0.4013 | 0.1892 | 0.3569 | 0.3560 |
| 22 | NASA source-only | 0.5094 | 0.4262 | 100.7246 | 0.1983 | 0.0000 | 0.0980 | 0.1234 | 0.1228 |
| 23 | NASA global MMD | 0.5130 | 0.4305 | 103.1340 | 0.1586 | 0.1070 | 0.0000 | 0.2458 | 0.2444 |

结论：第二迁移已包含 Transformer 对比。当前测试集最优是 `LSTM target-only + AD-TCN transfer` 融合；它比单独 `target-only LSTM` 的 RMSE 从 0.1716 降到 0.1665，说明第二迁移里 AD-TCN 迁移分量能补充 LSTM 的一部分误差结构。若只看迁移模型，`AD-TCN-MSC-DIM stage LMMD + target-sup` 和 `P-SA-MCD source pretrain + target fine-tune` 是较强项；严格无监督 NASA stage LMMD/source-only/global-MMD 明显偏弱。
