@echo off
setlocal enabledelayedexpansion

set input_dir=..\data\0_obj\
set output_dir=..\data\1_m\
set exe_path=..\bin\obj2m_.exe

if not exist %output_dir% (
    mkdir %output_dir%
)

for %%f in (%input_dir%*.obj) do (
    %exe_path% %%f %output_dir%
)

echo Done.
pause
