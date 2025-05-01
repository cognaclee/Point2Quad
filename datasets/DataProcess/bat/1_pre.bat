@echo off
setlocal enabledelayedexpansion

set input_dir=..\data\1_m\
set output_dir1=..\data\2_pre\faces\
set output_dir2=..\data\2_pre\points\
set exe_path=..\bin\pre.exe

if not exist %output_dir1% (
    mkdir %output_dir1%
)

if not exist %output_dir2% (
    mkdir %output_dir2%
)

for %%f in (%input_dir%*.m) do (
    %exe_path% %%f %output_dir1% %output_dir2%
)

echo Done.
pause
