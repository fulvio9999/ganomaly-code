@echo off

for %%P in (0 5 25 50 100) do (
    for %%C in (0 1 2 3 4 5) do (
        python test.py --dataset mnist --isize 32 --nc 1 --niter 15 --abnormal_class 2 --manualseed 0 --class_test 2 --perc_pullation %%P --count_test %%C
    )
)

exit /b 0
