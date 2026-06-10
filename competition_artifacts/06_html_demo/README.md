# XA-202608 动态 HTML Demo

本目录是可直接打开的离线动态 demo。建议双击 `index.html`，或用浏览器打开该文件。

## 数据来源

- `assets/demo-data.js` 由 `competition_artifacts/05_report_assets/demo_payload.json` 自动生成。
- Demo 只读已有实验结果，不现场训练模型，不重新拟合 TC，不使用目标测试标签做校准。
- `PG-STDA-SAC-RSPA` 是 strict raw UDA 迁移模型；`PG-STDA-SAC-RSPA-TC` 是 validation-only calibrated final pipeline。

## 重新生成

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/build_html_demo.py
```
