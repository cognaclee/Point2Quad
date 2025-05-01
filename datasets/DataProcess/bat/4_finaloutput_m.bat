@echo off

set source_dir=..\data\4_post\post2\
set target_dir=..\data\5_finaloutput_m\

if not exist "%target_dir%" (
    mkdir "%target_dir%"
)

copy "%source_dir%*.m" "%target_dir%"

echo .m files copied.
pause
