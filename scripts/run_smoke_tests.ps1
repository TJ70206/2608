$Python = "D:\anaconda\envs\jiebang\python.exe"

& $Python scripts/check_project.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts/train_baseline.py --config configs/synthetic_debug_fast.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts/plot_predictions.py --predictions outputs/synthetic_debug_fast/predictions_test.csv --output_dir outputs/synthetic_debug_fast/figures --max_units 2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts/train_transfer.py --config configs/synthetic_transfer_fast.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts/plot_predictions.py --predictions outputs/synthetic_transfer_fast/predictions_test.csv --output_dir outputs/synthetic_transfer_fast/figures --max_units 2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m compileall src scripts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts/check_demo_inputs.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts/check_html_demo.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts/check_pre_demo_readiness.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Smoke tests passed."
