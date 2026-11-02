@echo off
setlocal

rem Run from the repository root, regardless of the caller's working directory.
cd /d "%~dp0"
if errorlevel 1 goto :error

set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_CMD="
set "USE_VENV=0"

rem Use the system/current Python by default. Pass --venv to create/use .venv.
if /i "%~1"=="--venv" set "USE_VENV=1"
if /i "%~1"=="-v" set "USE_VENV=1"

if not "%~1"=="" if "%USE_VENV%"=="0" (
    echo [ERROR] Unknown option: %~1
    echo Usage: run.bat [--venv]
    goto :error
)

if "%USE_VENV%"=="1" goto :use_venv

rem Default: do not create or activate a virtual environment.
call :find_python
if errorlevel 1 goto :error
goto :python_ready

:use_venv
rem Optional: create .venv only when explicitly requested.
if not exist "%VENV_PYTHON%" (
    call :find_python
    if errorlevel 1 goto :error

    echo [INFO] Creating the virtual environment...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
)

set PYTHON_CMD="%VENV_PYTHON%"

:python_ready
rem Verify the selected interpreter version.
%PYTHON_CMD% -c "import sys; print('[INFO] Python', sys.version.split()[0]); raise SystemExit(0 if sys.version_info >= (3, 13) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.13 or later is required.
    if "%USE_VENV%"=="1" echo         Delete the .venv folder and run with a Python 3.13+ installation.
    goto :error
)

rem Install or update dependencies declared in pyproject.toml.
echo [INFO] Installing project dependencies...
echo [INFO] Using prebuilt scikit-image wheels to avoid local compilation.
%PYTHON_CMD% -m pip install --only-binary=scikit-image -e .
if errorlevel 1 goto :install_error

rem Start the application.
echo [INFO] Starting run.py...
%PYTHON_CMD% "%~dp0run.py"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] run.py exited with code %EXIT_CODE%.
) else (
    echo.
    echo [INFO] Application finished.
)

pause
exit /b %EXIT_CODE%


:find_python
rem Prefer the current PATH interpreter (including an activated environment).
where python >nul 2>nul
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
        exit /b 0
    )
)

rem Fall back to the Python Launcher if the python command is unavailable/too old.
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
        exit /b 0
    )
)

echo [ERROR] Python 3.13 or later was not found.
echo         Install Python 3.13+ and make sure python or py is available in PATH.
exit /b 1


:install_error
echo [ERROR] Dependency installation failed.
echo [HINT] Check your internet connection and confirm that Python 3.13+ is installed.
echo [HINT] This launcher requires a compatible scikit-image wheel for your Python version and architecture.
goto :error

:error
echo.
echo [ERROR] Setup or execution failed.
pause
exit /b 1
