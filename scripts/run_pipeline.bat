@echo off
REM ============================================================
REM  SpineSCAN FYP — Full Training Pipeline (Windows)
REM  RTX 4060 8GB VRAM adaptation (single fold, FP16, small BS)
REM ============================================================
REM
REM  BEFORE running this script:
REM   1. conda activate rsna
REM   2. python src/setup_check.py   (fix any FAIL items first)
REM   3. python src/gpu_patch.py
REM
REM  Run from SpineSCAN_FYP\ directory:
REM   cd C:\Users\hamma\Desktop\SpineSCAN_FYP
REM   run_pipeline.bat
REM
REM  Estimated wall-clock time on RTX 4060 (single fold):
REM   - DICOM conversion:         1-3 h
REM   - Preprocessing:            30 min
REM   - Sagittal slice est. s1:   3-5 h
REM   - Sagittal slice est. s2:   2-4 h
REM   - YOLO training:            4-8 h
REM   - Axial classification:     4-8 h
REM   - Sagittal classification:  4-8 h
REM   Total:                    ~20-40 h  (leave running overnight)
REM ============================================================

setlocal enabledelayedexpansion

REM ── Paths ────────────────────────────────────────────────────
set "BASE_DIR=%~dp0"
set "REPO_DIR=%BASE_DIR%..\spine_model"
set "INPUT_DIR=%REPO_DIR%\input"
set "CHKPT_DIR=%BASE_DIR%..\checkpoints"
set "OUT_DIR=%BASE_DIR%..\outputs"
set "LOG_DIR=%BASE_DIR%..\logs"

REM Fold to train (0 = first fold only, saves time on RTX 4060)
set "FOLD=0"

REM ── Setup ────────────────────────────────────────────────────
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%CHKPT_DIR%" mkdir "%CHKPT_DIR%"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

REM Timestamp for log file
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DT=%%I
set "TIMESTAMP=%DT:~0,8%_%DT:~8,6%"
set "LOGFILE=%LOG_DIR%\pipeline_%TIMESTAMP%.log"

echo Pipeline started: %DATE% %TIME% | tee "%LOGFILE%"
echo Repo dir: %REPO_DIR%          | tee -a "%LOGFILE%"
echo Training fold: %FOLD%          | tee -a "%LOGFILE%"
echo.

REM Check conda env
conda info --envs | findstr /C:"rsna" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Make sure you have run: conda activate rsna
)

REM ── Guard: patch must be applied ─────────────────────────────
if not exist "%REPO_DIR%\.rtx4060_patch_applied" (
    echo [ERROR] RTX 4060 patch not applied. Run first:
    echo         python src/gpu_patch.py
    pause
    exit /b 1
)

REM ── Change into repo dir for all commands ────────────────────
pushd "%REPO_DIR%"

REM =============================================================
REM  STEP 0: Verify CUDA
REM =============================================================
echo.
echo [STEP 0] Verifying CUDA...
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('CUDA OK:', torch.cuda.get_device_name(0))"
if errorlevel 1 (
    echo [ERROR] CUDA check failed. Exiting.
    popd & exit /b 1
)

REM =============================================================
REM  STEP 1: DICOM to PNG
REM  Output: input/axial_all_images/, input/sagittal_all_images/
REM =============================================================
echo.
echo [STEP 1] Converting DICOMs to PNG...
if exist "%INPUT_DIR%\sagittal_all_images" (
    echo   Already done — skipping DICOM conversion.
    echo   (Delete input\sagittal_all_images to redo)
) else (
    python dcm_to_png.py 2>&1 | tee -a "%LOGFILE%"
    if errorlevel 1 (
        echo [ERROR] dcm_to_png.py failed. Check log: %LOGFILE%
        popd & exit /b 1
    )
    echo [DONE] STEP 1 complete.
)

REM =============================================================
REM  STEP 2: General preprocessing (fold CSV)
REM  Output: input/train_with_fold.csv
REM =============================================================
echo.
echo [STEP 2] Running preprocess.py...
if exist "%INPUT_DIR%\train_with_fold.csv" (
    echo   Already done — skipping preprocessing.
) else (
    python preprocess.py 2>&1 | tee -a "%LOGFILE%"
    if errorlevel 1 (
        echo [ERROR] preprocess.py failed.
        popd & exit /b 1
    )
    echo [DONE] STEP 2 complete.
)

REM =============================================================
REM  STEP 3: Sagittal Slice Estimation — Stage 1
REM  Trains: rsna_sagittal_level_cl_spinal_v1 (fold 0)
REM          rsna_sagittal_level_cl_nfn_v1    (fold 0)
REM =============================================================
echo.
echo [STEP 3] Sagittal slice estimation — Stage 1...

python preprocess_for_sagittal_stage1.py 2>&1 | tee -a "%LOGFILE%"
if errorlevel 1 (echo [ERROR] preprocess_for_sagittal_stage1.py failed & popd & exit /b 1)

for %%C in (rsna_sagittal_level_cl_spinal_v1 rsna_sagittal_level_cl_nfn_v1) do (
    echo   Training config: %%C  fold: %FOLD%
    python train_one_fold.py -c %%C -f %FOLD% 2>&1 | tee -a "%LOGFILE%"
    if errorlevel 1 (echo [ERROR] Training %%C failed & popd & exit /b 1)
    python predict.py -c %%C -f %FOLD% 2>&1 | tee -a "%LOGFILE%"
    if errorlevel 1 (echo [ERROR] Prediction %%C failed & popd & exit /b 1)
    echo   [DONE] %%C
)
echo [DONE] STEP 3 complete.

REM =============================================================
REM  STEP 4: Sagittal Slice Estimation — Stage 2
REM  Trains: rsna_sagittal_cl (fold 0)
REM =============================================================
echo.
echo [STEP 4] Sagittal slice estimation — Stage 2...

python preprocess_for_sagittal_stage2.py 2>&1 | tee -a "%LOGFILE%"
if errorlevel 1 (echo [ERROR] preprocess_for_sagittal_stage2.py failed & popd & exit /b 1)

python train_one_fold.py -c rsna_sagittal_cl -f %FOLD% 2>&1 | tee -a "%LOGFILE%"
if errorlevel 1 (echo [ERROR] Training rsna_sagittal_cl failed & popd & exit /b 1)
python predict.py -c rsna_sagittal_cl -f %FOLD% 2>&1 | tee -a "%LOGFILE%"
if errorlevel 1 (echo [ERROR] Prediction rsna_sagittal_cl failed & popd & exit /b 1)
echo [DONE] STEP 4 complete.

REM =============================================================
REM  STEP 5: Axial YOLO — Region estimation
REM  NOTE: Inspect axial_yolo.sh to confirm config names,
REM        then update the list below.
REM =============================================================
echo.
echo [STEP 5] Axial YOLO region estimation...

python preprocess_for_axial_yolo.py 2>&1 | tee -a "%LOGFILE%"
if errorlevel 1 (echo [ERROR] preprocess_for_axial_yolo.py failed & popd & exit /b 1)

REM TODO: Replace CONFIG_AXIAL_YOLO with actual config name from axial_yolo.sh
REM       Check: type axial_yolo.sh  and look at the configs=(...) line
set "CONFIG_AXIAL_YOLO=rsna_axial_yolox_x"
echo   Training YOLO config: %CONFIG_AXIAL_YOLO%  fold: %FOLD%
python yolox_train_one_fold.py -c %CONFIG_AXIAL_YOLO% -f %FOLD% 2>&1 | tee -a "%LOGFILE%"
if errorlevel 1 (echo [ERROR] YOLO training failed & popd & exit /b 1)
echo [DONE] STEP 5 complete.

REM =============================================================
REM  STEP 6: Sagittal YOLO — Region estimation
REM  Downloads: improved coordinate annotations dataset
REM =============================================================
echo.
echo [STEP 6] Sagittal YOLO region estimation...

REM Download improved coordinate annotations if not present
if not exist "%INPUT_DIR%\bartley-coords-rsna-improved-csv" (
    pushd "%INPUT_DIR%"
    kaggle datasets download -d hammadamer/bartley-coords-rsna-improved-csv
    if errorlevel 1 (
        echo [WARN] Could not download Bartley coords — continuing without them.
    ) else (
        python -c "import zipfile; zipfile.ZipFile('bartley-coords-rsna-improved-csv.zip').extractall('.')"
    )
    popd
)

python preprocess_for_sagittal_yolo.py 2>&1 | tee -a "%LOGFILE%"
if errorlevel 1 (echo [ERROR] preprocess_for_sagittal_yolo.py failed & popd & exit /b 1)

python yolox_train_one_fold.py -c rsna_10classes_yolox_x -f %FOLD% 2>&1 | tee -a "%LOGFILE%"
if errorlevel 1 (echo [ERROR] Sagittal YOLO training failed & popd & exit /b 1)
echo [DONE] STEP 6 complete.

REM =============================================================
REM  STEP 7: Axial Classification
REM  NOTE: Inspect axial_classification.sh to confirm config names.
REM =============================================================
echo.
echo [STEP 7] Axial classification...

python preprocess_for_axial_classification.py 2>&1 | tee -a "%LOGFILE%"
if errorlevel 1 (echo [ERROR] preprocess_for_axial_classification.py failed & popd & exit /b 1)

REM TODO: Replace with actual config names from axial_classification.sh
REM       Check: type axial_classification.sh  and look at configs=(...) line
for %%C in (rsna_axial_spinal_dis3_crop_x05_y6 rsna_axial_spinal_dis3_crop_x1_y2) do (
    echo   Training config: %%C  fold: %FOLD%
    python train_one_fold.py -c %%C -f %FOLD% 2>&1 | tee -a "%LOGFILE%"
    if errorlevel 1 (echo [ERROR] Training %%C failed & popd & exit /b 1)
    python predict.py -c %%C -f %FOLD% 2>&1 | tee -a "%LOGFILE%"
    if errorlevel 1 (echo [ERROR] Prediction %%C failed & popd & exit /b 1)
    echo   [DONE] %%C
)
echo [DONE] STEP 7 complete.

REM =============================================================
REM  STEP 8: Sagittal Classification
REM =============================================================
echo.
echo [STEP 8] Sagittal classification...

python preprocess_for_sagittal_classification.py 2>&1 | tee -a "%LOGFILE%"
if errorlevel 1 (echo [ERROR] preprocess_for_sagittal_classification.py failed & popd & exit /b 1)

REM TODO: Replace with actual config names from sagittal_classification.sh
for %%C in (rsna_sagittal_spinal_dis3 rsna_sagittal_nfn_dis3) do (
    echo   Training config: %%C  fold: %FOLD%
    python train_one_fold.py -c %%C -f %FOLD% 2>&1 | tee -a "%LOGFILE%"
    if errorlevel 1 (echo [ERROR] Training %%C failed & popd & exit /b 1)
    python predict.py -c %%C -f %FOLD% 2>&1 | tee -a "%LOGFILE%"
    if errorlevel 1 (echo [ERROR] Prediction %%C failed & popd & exit /b 1)
    echo   [DONE] %%C
)
echo [DONE] STEP 8 complete.

REM =============================================================
REM  STEP 9: Noise reduction (uses OOF predictions from fold 0)
REM =============================================================
echo.
echo [STEP 9] Noise reduction...
python find_noisy_label.py 2>&1 | tee -a "%LOGFILE%"
if errorlevel 1 (
    echo [WARN] find_noisy_label.py failed — skipping noise reduction.
    echo        This is non-critical; continue with the results so far.
) else (
    echo [DONE] STEP 9 complete.
)

REM =============================================================
REM  STEP 10: Retrain with clean labels (optional but recommended)
REM =============================================================
echo.
echo [STEP 10] Retraining with clean labels...

for %%C in (rsna_axial_spinal_dis3_crop_x05_y6_reduce_noise rsna_axial_spinal_dis3_crop_x1_y2_reduce_noise) do (
    echo   Training config: %%C  fold: %FOLD%
    python train_one_fold.py -c %%C -f %FOLD% 2>&1 | tee -a "%LOGFILE%"
    if errorlevel 1 (echo [WARN] %%C failed — config may not exist, skipping)
    python predict.py -c %%C -f %FOLD% 2>&1 | tee -a "%LOGFILE%"
)
echo [DONE] STEP 10 complete.

REM =============================================================
REM  PIPELINE COMPLETE
REM =============================================================
popd
echo.
echo ============================================================
echo   Pipeline COMPLETE!  %DATE% %TIME%
echo   Log saved to: %LOGFILE%
echo.
echo   Next steps:
echo   1. Review results in: %REPO_DIR%\results\
echo   2. Run inference:    python src/inference.py --help
echo   3. Launch web app:   streamlit run app.py
echo ============================================================
pause
