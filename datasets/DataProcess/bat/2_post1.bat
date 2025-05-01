@echo off
setlocal enabledelayedexpansion

set input_dir=..\data\3_pred\
set output_dir=..\data\4_post\post1\
set exe_path=..\bin\post1.exe

if not exist %output_dir% (
    mkdir %output_dir%
)

for %%f in (%input_dir%*.m) do (
    %exe_path% %%f %output_dir%
)

echo Done.
pause
