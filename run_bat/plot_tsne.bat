@REM setlocal
@REM cd /d "%~dp0.."
@REM set "PYTHON=C:\Users\pc\.conda\envs\ISRA\python.exe"
@REM set "SOURCE_LIST=data\LongSig_50\t-SNE\19-27-47-5-38\class_5\source\SNR_20_200_list.txt"
@REM set "TARGET_LIST=data\LongSig_50\t-SNE\19-27-47-5-38\class_5\target\SNR_0_200_list.txt"
@REM set "CHECKPOINT=model\RepVGG_B1g2\TD_15class_SNR_0_best_model_en.pt"

@REM for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TIMESTAMP=%%i"
@REM set "PLOTS_DIR=t-SNE_results\trained_repvgg\19-27-47-5-38\%RUN_TIMESTAMP%"

@REM echo Drawing t-SNE figures
@REM echo Figures will be saved to: %PLOTS_DIR%

@REM "%PYTHON%" -u good_tool\create_tsne.py ^
@REM     --source-list "%SOURCE_LIST%" ^
@REM     --target-list "%TARGET_LIST%" ^
@REM     --feature-extractor trained_repvgg ^
@REM     --checkpoint "%CHECKPOINT%" ^
@REM     --plots-dir "%PLOTS_DIR%" ^
@REM     --workers 4

@REM setlocal
@REM cd /d "%~dp0.."
@REM set "PYTHON=C:\Users\pc\.conda\envs\ISRA\python.exe"
@REM set "SOURCE_LIST=data\LongSig_50\t-SNE\19-27-47-5-38\class_5\source\SNR_20_200_list.txt"
@REM set "TARGET_LIST=data\LongSig_50\t-SNE\19-27-47-5-38\class_5\target\SNR_0_200_list.txt"
@REM set "CHECKPOINT=model\ResNet50\TD_15class_SNR_0_best_model_en.pt"

@REM for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TIMESTAMP=%%i"
@REM set "PLOTS_DIR=t-SNE_results\trained_resnet50\19-27-47-5-38\%RUN_TIMESTAMP%"

@REM echo Drawing t-SNE figures
@REM echo Figures will be saved to: %PLOTS_DIR%

@REM "%PYTHON%" -u good_tool\create_tsne.py ^
@REM     --source-list "%SOURCE_LIST%" ^
@REM     --target-list "%TARGET_LIST%" ^
@REM     --feature-extractor trained_resnet50 ^
@REM     --checkpoint "%CHECKPOINT%" ^
@REM     --plots-dir "%PLOTS_DIR%" ^
@REM     --workers 4

@REM setlocal
@REM cd /d "%~dp0.."
@REM set "PYTHON=C:\Users\pc\.conda\envs\ISRA\python.exe"
@REM set "SOURCE_LIST=data\LongSig_50\t-SNE\class_15\source\SNR_20_200_list.txt"
@REM set "TARGET_LIST=data\LongSig_50\t-SNE\class_15\target\SNR_0_200_list.txt"
@REM set "CHECKPOINT=model\RepVGG_B1g2\TD_15class_SNR_0_best_model_en.pt"

@REM for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TIMESTAMP=%%i"
@REM set "PLOTS_DIR=t-SNE_results\trained_repvgg\class_15\%RUN_TIMESTAMP%"

@REM echo Drawing t-SNE figures
@REM echo Figures will be saved to: %PLOTS_DIR%

@REM "%PYTHON%" -u good_tool\create_tsne-class15.py ^
@REM     --source-list "%SOURCE_LIST%" ^
@REM     --target-list "%TARGET_LIST%" ^
@REM     --feature-extractor trained_repvgg ^
@REM     --checkpoint "%CHECKPOINT%" ^
@REM     --plots-dir "%PLOTS_DIR%" ^
@REM     --workers 4

@REM setlocal
@REM cd /d "%~dp0.."
@REM set "PYTHON=C:\Users\pc\.conda\envs\ISRA\python.exe"
@REM set "SOURCE_LIST=data\LongSig_50\t-SNE\class_15\source\SNR_20_200_list.txt"
@REM set "TARGET_LIST=data\LongSig_50\t-SNE\class_15\target\SNR_0_200_list.txt"
@REM set "CHECKPOINT=model\ResNet50\TD_15class_SNR_0_best_model_en.pt"

@REM for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TIMESTAMP=%%i"
@REM set "PLOTS_DIR=t-SNE_results\trained_resnet50\class_15\%RUN_TIMESTAMP%"

@REM echo Drawing t-SNE figures
@REM echo Figures will be saved to: %PLOTS_DIR%

@REM "%PYTHON%" -u good_tool\create_tsne-class15.py ^
@REM     --source-list "%SOURCE_LIST%" ^
@REM     --target-list "%TARGET_LIST%" ^
@REM     --feature-extractor trained_resnet50 ^
@REM     --checkpoint "%CHECKPOINT%" ^
@REM     --plots-dir "%PLOTS_DIR%" ^
@REM     --workers 4


@REM setlocal
@REM cd /d "%~dp0.."
@REM set "PYTHON=C:\Users\pc\.conda\envs\ISRA\python.exe"
@REM set "SOURCE_LIST=data\LongSig_50\t-SNE\class_15\source\SNR_20_200_list.txt"
@REM set "TARGET_LIST=data\LongSig_50\t-SNE\class_15\target\SNR_0_200_list.txt"
@REM set "CHECKPOINT=model\RepVGG_B1g2\TD_15class_SNR_0_best_model_en.pt"

@REM for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TIMESTAMP=%%i"
@REM set "PLOTS_DIR=t-SNE_results\trained_repvgg\class_15\%RUN_TIMESTAMP%"

@REM echo Drawing t-SNE figures
@REM echo Figures will be saved to: %PLOTS_DIR%

@REM "%PYTHON%" -u good_tool\create_tsne-class15-no_legand.py ^
@REM     --source-list "%SOURCE_LIST%" ^
@REM     --target-list "%TARGET_LIST%" ^
@REM     --feature-extractor trained_repvgg ^
@REM     --checkpoint "%CHECKPOINT%" ^
@REM     --plots-dir "%PLOTS_DIR%" ^
@REM     --workers 4


setlocal
cd /d "%~dp0.."
set "PYTHON=C:\Users\pc\.conda\envs\ISRA\python.exe"
set "SOURCE_LIST=data\LongSig_50\t-SNE\class_15\source\SNR_20_200_list.txt"
set "TARGET_LIST=data\LongSig_50\t-SNE\class_15\target\SNR_0_200_list.txt"
set "CHECKPOINT=model\ResNet50\TD_15class_SNR_0_best_model_en.pt"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TIMESTAMP=%%i"
set "PLOTS_DIR=t-SNE_results\trained_resnet50\class_15\%RUN_TIMESTAMP%"

echo Drawing t-SNE figures
echo Figures will be saved to: %PLOTS_DIR%

"%PYTHON%" -u good_tool\create_tsne-class15-no_legand.py ^
    --source-list "%SOURCE_LIST%" ^
    --target-list "%TARGET_LIST%" ^
    --feature-extractor trained_resnet50 ^
    --checkpoint "%CHECKPOINT%" ^
    --plots-dir "%PLOTS_DIR%" ^
    --workers 4