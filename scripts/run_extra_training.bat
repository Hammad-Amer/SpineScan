@echo off
REM ================================================================
REM SpineSCAN FYP — Train Missing Crop-Ratio Variants
REM ================================================================
REM Trains 9 additional classification models to enable full ensemble.
REM Each model gets train + predict (predict generates OOF CSV for noise detection).
REM
REM Run from: SpineSCAN_FYP\scripts directory
REM Prereqs:  conda activate rsna
REM           Base 5 models already trained
REM           All preprocessing (YOLO boxes, sagittal CSVs) already done
REM
REM Estimated time: ~12-15h on RTX 4060 8GB
REM ================================================================

setlocal enabledelayedexpansion
set FOLD=0
set REPO=..\spine_model
set LOGFILE=extra_training_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%.log

echo ================================================================
echo  SpineSCAN — Extra Crop-Variant Training
echo  Started: %date% %time%
echo  Log: %LOGFILE%
echo ================================================================

pushd %REPO%
if errorlevel 1 (
    echo [FATAL] Cannot cd into %REPO%. Aborting.
    exit /b 1
)

REM ── Group 1: Sagittal MIL models (fastest, image_size 96-128) ──────

echo.
echo [1/9] Sagittal Spinal crop_x03_y05
echo ────────────────────────────────────
if not exist "results\rsna_saggital_mil_spinal_crop_x03_y05\oof_fold0.csv" (
    powershell -Command "python train_one_fold.py -c rsna_saggital_mil_spinal_crop_x03_y05 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
    if errorlevel 1 echo [WARN] Training failed for rsna_saggital_mil_spinal_crop_x03_y05
    powershell -Command "python predict.py -c rsna_saggital_mil_spinal_crop_x03_y05 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
) else (
    echo [SKIP] rsna_saggital_mil_spinal_crop_x03_y05 already has OOF, skipping
)

echo.
echo [2/9] Sagittal NFN crop_x15_y1_v2
echo ────────────────────────────────────
if not exist "results\rsna_saggital_mil_nfn_crop_x15_y1_v2\oof_fold0.csv" (
    powershell -Command "python train_one_fold.py -c rsna_saggital_mil_nfn_crop_x15_y1_v2 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
    if errorlevel 1 echo [WARN] Training failed for rsna_saggital_mil_nfn_crop_x15_y1_v2
    powershell -Command "python predict.py -c rsna_saggital_mil_nfn_crop_x15_y1_v2 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
) else (
    echo [SKIP] rsna_saggital_mil_nfn_crop_x15_y1_v2 already has OOF, skipping
)

echo.
echo [3/9] Sagittal NFN crop_x03_y1_v2
echo ────────────────────────────────────
if not exist "results\rsna_saggital_mil_nfn_crop_x03_y1_v2\oof_fold0.csv" (
    powershell -Command "python train_one_fold.py -c rsna_saggital_mil_nfn_crop_x03_y1_v2 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
    if errorlevel 1 echo [WARN] Training failed for rsna_saggital_mil_nfn_crop_x03_y1_v2
    powershell -Command "python predict.py -c rsna_saggital_mil_nfn_crop_x03_y1_v2 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
) else (
    echo [SKIP] rsna_saggital_mil_nfn_crop_x03_y1_v2 already has OOF, skipping
)

echo.
echo [4/9] Sagittal SS crop_x03_y05_96
echo ────────────────────────────────────
if not exist "results\rsna_saggital_mil_ss_crop_x03_y05_96\oof_fold0.csv" (
    powershell -Command "python train_one_fold.py -c rsna_saggital_mil_ss_crop_x03_y05_96 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
    if errorlevel 1 echo [WARN] Training failed for rsna_saggital_mil_ss_crop_x03_y05_96
    powershell -Command "python predict.py -c rsna_saggital_mil_ss_crop_x03_y05_96 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
) else (
    echo [SKIP] rsna_saggital_mil_ss_crop_x03_y05_96 already has OOF, skipping
)

echo.
echo [5/9] Sagittal SS crop_x03_y2_96
echo ────────────────────────────────────
if not exist "results\rsna_saggital_mil_ss_crop_x03_y2_96\oof_fold0.csv" (
    powershell -Command "python train_one_fold.py -c rsna_saggital_mil_ss_crop_x03_y2_96 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
    if errorlevel 1 echo [WARN] Training failed for rsna_saggital_mil_ss_crop_x03_y2_96
    powershell -Command "python predict.py -c rsna_saggital_mil_ss_crop_x03_y2_96 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
) else (
    echo [SKIP] rsna_saggital_mil_ss_crop_x03_y2_96 already has OOF, skipping
)

echo.
echo [6/9] Sagittal SS crop_x1_y07_96
echo ────────────────────────────────────
if not exist "results\rsna_saggital_mil_ss_crop_x1_y07_96\oof_fold0.csv" (
    powershell -Command "python train_one_fold.py -c rsna_saggital_mil_ss_crop_x1_y07_96 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
    if errorlevel 1 echo [WARN] Training failed for rsna_saggital_mil_ss_crop_x1_y07_96
    powershell -Command "python predict.py -c rsna_saggital_mil_ss_crop_x1_y07_96 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
) else (
    echo [SKIP] rsna_saggital_mil_ss_crop_x1_y07_96 already has OOF, skipping
)

REM ── Group 2: Axial models (slower, image_size 384) ─────────────────

echo.
echo [7/9] Axial Spinal crop_x05_y6
echo ────────────────────────────────────
if not exist "results\rsna_axial_spinal_dis3_crop_x05_y6\oof_fold0.csv" (
    powershell -Command "python train_one_fold.py -c rsna_axial_spinal_dis3_crop_x05_y6 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
    if errorlevel 1 echo [WARN] Training failed for rsna_axial_spinal_dis3_crop_x05_y6
    powershell -Command "python predict.py -c rsna_axial_spinal_dis3_crop_x05_y6 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
) else (
    echo [SKIP] rsna_axial_spinal_dis3_crop_x05_y6 already has OOF, skipping
)

echo.
echo [8/9] Axial NFN+SS x2_y6_center_pad0
echo ────────────────────────────────────
if not exist "results\rsna_axial_ss_nfn_x2_y6_center_pad0\oof_fold0.csv" (
    powershell -Command "python train_one_fold.py -c rsna_axial_ss_nfn_x2_y6_center_pad0 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
    if errorlevel 1 echo [WARN] Training failed for rsna_axial_ss_nfn_x2_y6_center_pad0
    powershell -Command "python predict.py -c rsna_axial_ss_nfn_x2_y6_center_pad0 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
) else (
    echo [SKIP] rsna_axial_ss_nfn_x2_y6_center_pad0 already has OOF, skipping
)

echo.
echo [9/9] Axial NFN+SS x2_y8_center_pad10
echo ────────────────────────────────────
if not exist "results\rsna_axial_ss_nfn_x2_y8_center_pad10\oof_fold0.csv" (
    powershell -Command "python train_one_fold.py -c rsna_axial_ss_nfn_x2_y8_center_pad10 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
    if errorlevel 1 echo [WARN] Training failed for rsna_axial_ss_nfn_x2_y8_center_pad10
    powershell -Command "python predict.py -c rsna_axial_ss_nfn_x2_y8_center_pad10 -f %FOLD% 2>&1 | Tee-Object -FilePath '../%LOGFILE%' -Append"
) else (
    echo [SKIP] rsna_axial_ss_nfn_x2_y8_center_pad10 already has OOF, skipping
)

popd

echo.
echo ================================================================
echo  Extra training complete!  %date% %time%
echo  Check OOF CSVs:
echo    dir %REPO%\results\*\oof_fold0.csv
echo  Next: python src/noise_detection.py
echo ================================================================
