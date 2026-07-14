@REM 训练repvgg模型，目标域class_15, SNR_0
@REM C:\Users\pc\.conda\envs\ISRA\python.exe -u run_final_version.py ^
@REM     --s 5 ^
@REM     --t 1 ^
@REM     --dset LongSig_50 ^
@REM     --prepro GASF_old ^
@REM     --cls_type Multi_class ^
@REM     --net RepVGG_B1g2 ^
@REM     --multi_class 15 ^
@REM     --gpu_id 0 ^
@REM     --seed 2025 ^
@REM     > results\RepVGG\Multi_class\SNR_0_class_15-save-model.txt

@REM 训练resnet50模型，目标域class_15, SNR_0
@REM C:\Users\pc\.conda\envs\ISRA\python.exe -u run_final_version.py ^
@REM     --s 5 ^
@REM     --t 1 ^
@REM     --dset LongSig_50 ^
@REM     --prepro GASF_old ^
@REM     --cls_type Multi_class ^
@REM     --net ResNet50 ^
@REM     --multi_class 15 ^
@REM     --gpu_id 0 ^
@REM     --seed 2025 ^
@REM     > results\ResNet50\Multi_class\SNR_0_15_class-save-model.txt

@REM @REM 训练repvgg模型，目标域class_15, SNR_0
@REM C:\Users\pc\.conda\envs\ISRA\python.exe -u run_final_version.py ^
@REM     --s 5 ^
@REM     --t 0 ^
@REM     --dset LongSig_50 ^
@REM     --prepro GASF_old ^
@REM     --cls_type Multi_class ^
@REM     --net RepVGG_B1g2 ^
@REM     --multi_class 30 ^
@REM     --gpu_id 0 ^
@REM     --seed 2025 ^
@REM     > results\RepVGG\Multi_class\SNR_-5_class_30-save-model.txt


@REM C:\Users\pc\.conda\envs\ISRA\python.exe -u run_final_version.py ^
@REM     --s 5 ^
@REM     --t 1 ^
@REM     --dset LongSig_50 ^
@REM     --prepro GASF_old ^
@REM     --cls_type Multi_class ^
@REM     --net RepVGG_B1g2 ^
@REM     --multi_class 30 ^
@REM     --gpu_id 0 ^
@REM     --seed 2025 ^
@REM     > results\RepVGG\Multi_class\SNR_0_class_30-save-model.txt

@REM C:\Users\pc\.conda\envs\ISRA\python.exe -u run_final_version.py ^
@REM     --s 5 ^
@REM     --t 2 ^
@REM     --dset LongSig_50 ^
@REM     --prepro GASF_old ^
@REM     --cls_type Multi_class ^
@REM     --net RepVGG_B1g2 ^
@REM     --multi_class 30 ^
@REM     --gpu_id 0 ^
@REM     --seed 2025 ^
@REM     > results\RepVGG\Multi_class\SNR_5_class_30-save-model.txt


C:\Users\pc\.conda\envs\ISRA\python.exe -u run_final_version.py ^
    --s 5 ^
    --t 3 ^
    --dset LongSig_50 ^
    --prepro GASF_old ^
    --cls_type Multi_class ^
    --net RepVGG_B1g2 ^
    --multi_class 30 ^
    --gpu_id 0 ^
    --seed 2025 ^
    > results\RepVGG\Multi_class\SNR_10_class_30-save-model.txt

C:\Users\pc\.conda\envs\ISRA\python.exe -u run_final_version.py ^
    --s 5 ^
    --t 4 ^
    --dset LongSig_50 ^
    --prepro GASF_old ^
    --cls_type Multi_class ^
    --net RepVGG_B1g2 ^
    --multi_class 30 ^
    --gpu_id 0 ^
    --seed 2025 ^
    > results\RepVGG\Multi_class\SNR_15_class_30-save-model.txt


C:\Users\pc\.conda\envs\ISRA\python.exe -u run_final_version.py ^
    --s 5 ^
    --t 5 ^
    --dset LongSig_50 ^
    --prepro GASF_old ^
    --cls_type Multi_class ^
    --net RepVGG_B1g2 ^
    --multi_class 30 ^
    --gpu_id 0 ^
    --seed 2025 ^
    > results\RepVGG\Multi_class\SNR_20_class_30-save-model.txt