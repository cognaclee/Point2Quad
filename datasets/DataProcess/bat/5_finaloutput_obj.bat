@echo off
setlocal enabledelayedexpansion

set input_dir=..\data\5_finaloutput_m\
set output_dir=..\data\6_finaloutput_obj\
set exe_path=..\bin\m2obj_.exe

if not exist %output_dir% (
    mkdir %output_dir%
)

for %%f in (%input_dir%*.m) do (
    %exe_path% %%f %output_dir%
)

echo Done.
pause
