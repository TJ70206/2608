# XA-202608 动态 HTML Demo 源模板

本目录保存离线 HTML demo 的源模板。正式可打开产物由以下命令生成到 `competition_artifacts/06_html_demo/`：

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' scripts/build_html_demo.py
```

生成过程只读取 `competition_artifacts/05_report_assets/demo_payload.json`，并把它转换为 `assets/demo-data.js`。Demo 不现场训练模型，不重新拟合 TC，不使用目标测试标签做校准。
