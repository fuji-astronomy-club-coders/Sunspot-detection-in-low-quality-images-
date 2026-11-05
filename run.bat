@echo off
setlocal

rem Always run from the repository root.
cd /d "%~dp0"
if errorlevel 1 goto :error

set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_CMD="
set "APP_ARGS="

rem The GUI re-enters this batch file with a private marker so the selected
rem command is parsed without opening the GUI a second time.
set "FROM_GUI="
if /i "%~1"=="--_from-gui" (
    set "FROM_GUI=1"
    shift
)
if "%~1"=="" if not defined FROM_GUI goto :launch_gui

rem Parse supported options.
if /i "%~1"=="--help" (
    if not "%~2"=="" goto :invalid_option
    goto :help
)
if /i "%~1"=="--setup" (
    if /i "%~2"=="--global" (
        if not "%~3"=="" goto :invalid_option
        goto :setup_global
    )
    if not "%~2"=="" goto :invalid_option
    goto :setup_venv
)
if /i "%~1"=="--venv" (
    if /i "%~2"=="--compare" (
        if not "%~4"=="" goto :invalid_option
        set "APP_ARGS=--compare"
        if not "%~3"=="" set "APP_ARGS=--compare %3"
        goto :run_venv
    )
    if not "%~2"=="" goto :invalid_option
    goto :run_venv
)
if /i "%~1"=="--compare" (
    if not "%~3"=="" goto :invalid_option
    set "APP_ARGS=--compare"
    if not "%~2"=="" set "APP_ARGS=--compare %2"
    if exist "%VENV_DIR%\." goto :run_venv
    goto :run_system
)
if not "%~1"=="" goto :invalid_option

rem Prefer an existing .venv during normal startup.
rem If the directory exists but is incomplete, report it instead of silently
rem falling back to the system Python.
if exist "%VENV_DIR%\." goto :run_venv
goto :run_system


:launch_gui
rem Prefer the existing venv for the GUI when it has Tkinter. If it does not,
rem use a system Python so an incomplete venv can still be repaired from GUI.
set "GUI_USE_VENV="
if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" -c "import sys, tkinter; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>nul
    if not errorlevel 1 set "GUI_USE_VENV=1"
)

echo [INFO] Starting launcher GUI...
if defined GUI_USE_VENV goto :launch_gui_venv

rem Invoke the system interpreter directly; do not store a quoted executable
rem path in another variable, because nested cmd.exe quoting is error-prone.
call :find_python
if errorlevel 1 goto :error
%PYTHON_CMD% "%~dp0launcher_gui.py"
goto :gui_finished

:launch_gui_venv
"%VENV_PYTHON%" "%~dp0launcher_gui.py"

:gui_finished
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] The launcher GUI could not be started or exited with an error.
    echo [HINT] Use Python 3.13+ with Tkinter (Tcl/Tk) installed.
    pause
)
exit /b %EXIT_CODE%


:run_system
call :find_python
if errorlevel 1 goto :error
goto :run_application


:run_venv
if not exist "%VENV_PYTHON%" goto :venv_not_ready

set PYTHON_CMD="%VENV_PYTHON%"

rem Check the interpreter version.
%PYTHON_CMD% -c "import sys; print('[INFO] Python', sys.version.split()[0]); raise SystemExit(0 if sys.version_info >= (3, 13) else 1)"
if errorlevel 1 goto :venv_not_ready

rem Check that this project was installed in the venv and its dependencies are intact.
%PYTHON_CMD% -c "import tomllib; from pathlib import Path; from importlib.metadata import distribution; distribution(tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))['project']['name'])"
if errorlevel 1 goto :venv_not_ready

%PYTHON_CMD% -m pip check
if errorlevel 1 goto :venv_not_ready

goto :run_application


:setup_venv
rem Reuse an existing venv interpreter; otherwise create or repair .venv.
if exist "%VENV_PYTHON%" goto :setup_existing_venv

call :find_python
if errorlevel 1 goto :error

echo [INFO] Creating the virtual environment...
if exist "%VENV_DIR%\." (
    %PYTHON_CMD% -m venv --clear "%VENV_DIR%"
) else (
    %PYTHON_CMD% -m venv "%VENV_DIR%"
)
if errorlevel 1 goto :error

set PYTHON_CMD="%VENV_PYTHON%"
goto :setup_venv_check


:setup_existing_venv
set PYTHON_CMD="%VENV_PYTHON%"


:setup_venv_check
%PYTHON_CMD% -c "import sys; print('[INFO] Python', sys.version.split()[0]); raise SystemExit(0 if sys.version_info >= (3, 13) else 1)"
if errorlevel 1 goto :python_version_error

rem Ensure pip exists even if the existing virtual environment is incomplete.
%PYTHON_CMD% -m ensurepip --upgrade
if errorlevel 1 goto :install_error

rem pip reads runtime dependencies from [project].dependencies in pyproject.toml.
rem Development extras from [project.optional-dependencies] are intentionally excluded.
echo [INFO] Installing dependencies declared in [project].dependencies...
echo [INFO] Using prebuilt scikit-image wheels to avoid local compilation.
%PYTHON_CMD% -m pip install --only-binary=scikit-image -e .
if errorlevel 1 goto :install_error

echo.
echo [INFO] Virtual environment setup completed.
echo [INFO] Double-clicking run.bat will now use this .venv automatically.
echo [INFO] You can also force it with "run.bat --venv".
pause
exit /b 0


:setup_global
rem Select a Python outside any currently activated virtual environment.
call :find_python
if errorlevel 1 goto :error

%PYTHON_CMD% -c "import sys; print('[INFO] Global Python', sys.version.split()[0])"
if errorlevel 1 goto :error

rem This mode intentionally does not create or modify .venv.
%PYTHON_CMD% -m ensurepip --upgrade
if errorlevel 1 goto :install_error

echo [INFO] Installing dependencies into the system Python environment...
echo [INFO] Using prebuilt scikit-image wheels to avoid local compilation.
%PYTHON_CMD% -m pip install --only-binary=scikit-image -e .
if errorlevel 1 goto :install_error

echo.
echo [INFO] Global setup completed. The .venv directory was not changed.
if exist "%VENV_DIR%\." (
    echo [INFO] Normal run.bat startup will still prioritize the existing .venv.
    echo [INFO] To use the global Python directly, run "python run.py" with the same system interpreter.
)
pause
exit /b 0


:run_application
echo [INFO] Starting run.py...
%PYTHON_CMD% "%~dp0run.py" %APP_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] run.py exited with code %EXIT_CODE%.
    echo [HINT] Run "run.bat --setup" to prepare or update .venv.
    echo [HINT] Or run "run.bat --setup --global" to install dependencies in the system Python.
) else (
    echo.
    echo [INFO] Application finished.
)

pause
exit /b %EXIT_CODE%


:find_python
rem Use a Python command available on PATH, excluding active virtual environments.
where python >nul 2>nul
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 13) and sys.prefix == sys.base_prefix else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
        exit /b 0
    )
)

rem Fall back to the Python Launcher.
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 13) and sys.prefix == sys.base_prefix else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
        exit /b 0
    )
)

echo [ERROR] Python 3.13 or later was not found outside a virtual environment.
echo         Install Python 3.13+ and make sure python or py is available in PATH.
exit /b 1


:venv_not_ready
echo.
echo [ERROR] The existing virtual environment is missing or incomplete.
echo [HINT] No packages were installed or modified during this launch.
echo [HINT] Run "run.bat --setup" to prepare or repair .venv, then retry.
echo [HINT] Normal startup will not fall back to the system Python while .venv exists.
goto :error


:python_version_error
echo.
echo [ERROR] Python 3.13 or later is required in .venv.
echo [HINT] Delete the existing .venv folder and run "run.bat --setup" again.
goto :error


:install_error
echo [ERROR] Dependency installation failed.
echo [HINT] Check your internet connection and confirm that Python 3.13+ is installed.
echo [HINT] This setup requires a compatible scikit-image wheel for your Python version and architecture.
goto :error


:invalid_option
echo [ERROR] Invalid option or argument: %~1 %~2 %~3
echo Usage: run.bat [--compare [DATE_FOLDER]^|--venv [--compare [DATE_FOLDER]]^|--setup [--global]^|--help]
goto :error


:help
echo Usage:
echo   run.bat                  Open the startup GUI to select an operation and options.
echo   run.bat --venv           Force the existing .venv environment.
echo   run.bat --compare        Start comparison mode using a folder-selection dialog.
echo   run.bat --compare PATH   Start comparison mode for the specified date folder.
echo   run.bat --venv --compare [PATH]  Run comparison mode with the existing .venv.
echo   run.bat --setup          Create or update .venv with project dependencies.
echo   run.bat --setup --global Install project dependencies into system Python without creating .venv.
echo   run.bat --help           Show this help.
echo.
echo The --global option is valid only together with --setup.
echo The startup GUI requires Python 3.13+ with Tkinter available.
if defined FROM_GUI pause
exit /b 0


:error
echo.
echo [ERROR] Setup or execution failed.
pause
exit /b 1
