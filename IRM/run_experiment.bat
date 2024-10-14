@echo off
for %%h in (0 1) do (
    for %%r in (0 1) do (
        for %%s in (0 1) do (
            echo Running with --setup_hidden=%%h, --setup_hetero=%%r, --setup_scramble=%%s
            python main.py --setup_hidden %%h --setup_hetero %%r --setup_scramble %%s
        )
    )
)
