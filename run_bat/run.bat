@REM for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "timestamp=%%t"

@REM C:\Users\pc\.conda\envs\ISRA\python.exe -u run_version-26-7-14.py ^
@REM     --s 5 ^
@REM     --t 1 ^
@REM     --dset LongSig_50 ^
@REM     --cls_type Multi_class ^
@REM     --net RepVGG_B1g2 ^
@REM     --multi_class 30 ^
@REM     --gpu_id 0 ^
@REM     --seed 2026 ^
@REM     > results\RepVGG\TD_Multi_class\SNR_0_class_30_%timestamp%.log


for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "timestamp=%%t"

@REM set "net=RepVGG_B1g2"
@REM set "log_dir=results\"%net%"\Pretrain"
@REM if not exist "%log_dir%" mkdir "%log_dir%"
@REM C:\Users\pc\.conda\envs\ISRA\python.exe -u run_version-26-7-17-pretrain.py ^
@REM     --net "%net%"^
@REM     --gpu_id 0 ^
@REM     --seed 2026 ^
@REM     --worker 4 ^
@REM     > "%log_dir%"\SD_SNR_20_class_50_%timestamp%.log


set "net=RepVGG_B1g2"
set "log_dir=results\"%net%"\Train"
set "s=20"
set "t=0"
if not exist "%log_dir%" mkdir "%log_dir%"
C:\Users\pc\.conda\envs\ISRA\python.exe -u run_version-26-7-18.py ^
    --s %s% ^
    --t %t% ^
    --net "%net%"^
    --gpu_id 0 ^
    --seed 2026 ^
    --worker 4 ^
    --no-amp^
    > "%log_dir%"\SD_SNR_20_class_50_%timestamp%.log