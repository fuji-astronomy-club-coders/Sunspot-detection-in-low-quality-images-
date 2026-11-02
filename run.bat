@echo off
setlocal

rem Run from the repository root, regardless of the caller's working directory.
cd /d "%~dp0"
if errorlevel 1 goto :error

set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

rem Create the project virtual environment on the first launch.
if not exist "%VENV_PYTHON%" (
    echo [INFO] Creating the virtual environment...
    call :create_venv
    if errorlevel 1 goto :error
)

rem The project requires Python 3.13 or later.
"%VENV_PYTHON%" -c "import sys; print('[INFO] Python', sys.version.split()[0]); raise SystemExit(0 if sys.version_info >= (3, 13) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.13 or later is required.
    echo         Delete the .venv folder and run this file again with Python 3.13+ installed.
    goto :error
)

rem Install or update dependencies declared in pyproject.toml.
echo [INFO] Installing project dependencies...
"%VENV_PYTHON%" -m pip install -e .
if errorlevel 1 goto :error

rem Start the application with the project's virtual environment.
echo [INFO] Starting run.py...
"%VENV_PYTHON%" "%~dp0run.py"
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


:create_venv
rem Prefer the Python Launcher, then fall back to the python command.
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%VENV_DIR%"
        if errorlevel 1 exit /b 1
        exit /b 0
    )
)

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.13 or later was not found.
    echo         Install Python and make sure the py or python command is available.
    exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.13 or later is required.
    exit /b 1
)

python -m venv "%VENV_DIR%"
if errorlevel 1 exit /b 1
exit /b 0


:error
echo.
echo [ERROR] Setup or execution failed.
pause
exit /b 1
