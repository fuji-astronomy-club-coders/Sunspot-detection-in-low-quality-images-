@echo off
setlocal

rem Always run from the repository root.
cd /d "%~dp0"
if errorlevel 1 goto :error

set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_CMD="

rem Environment setup is opt-in. A normal launch never installs packages.
if /i "%~1"=="--help" goto :help
if /i "%~1"=="--setup" goto :setup
if not "%~1"=="" goto :invalid_option

goto :run


:setup
if not "%~2"=="" goto :invalid_option

rem Reuse an existing environment, or create it with an available Python.
if exist "%VENV_PYTHON%" goto :setup_existing

call :find_python
if errorlevel 1 goto :error

echo [INFO] Creating the virtual environment...
%PYTHON_CMD% -m venv "%VENV_DIR%"
if errorlevel 1 goto :error
set PYTHON_CMD="%VENV_PYTHON%"
goto :setup_check

:setup_existing
set PYTHON_CMD="%VENV_PYTHON%"

:setup_check
%PYTHON_CMD% -c "import sys; print('[INFO] Python', sys.version.split()[0]); raise SystemExit(0 if sys.version_info >= (3, 13) else 1)"
if errorlevel 1 goto :python_version_error

echo [INFO] Installing project dependencies...
echo [INFO] Using prebuilt scikit-image wheels to avoid local compilation.
%PYTHON_CMD% -m pip install --only-binary=scikit-image -e .
if errorlevel 1 goto :install_error

echo.
echo [INFO] Environment setup completed.
echo [INFO] From now on, double-click run.bat to start the application.
pause
exit /b 0


:run
rem Prefer the prepared project environment. Otherwise use an existing Python.
if exist "%VENV_PYTHON%" goto :run_with_venv

call :find_python
if errorlevel 1 goto :error
goto :run_application

:run_with_venv
set PYTHON_CMD="%VENV_PYTHON%"

:run_application
%PYTHON_CMD% -c "import sys; print('[INFO] Python', sys.version.split()[0]); raise SystemExit(0 if sys.version_info >= (3, 13) else 1)"
if errorlevel 1 goto :python_version_error

echo [INFO] Starting run.py...
%PYTHON_CMD% "%~dp0run.py"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] run.py exited with code %EXIT_CODE%.
    echo [HINT] If required packages are missing, run "run.bat --setup" once.
) else (
    echo.
    echo [INFO] Application finished.
)

pause
exit /b %EXIT_CODE%


:find_python
rem Prefer the Python command available on PATH.
where python >nul 2>nul
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
        exit /b 0
    )
)

rem Fall back to the Python Launcher.
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


:python_version_error
echo.
echo [ERROR] Python 3.13 or later is required.
if exist "%VENV_PYTHON%" (
    echo [HINT] The existing .venv may use an older Python version.
    echo        Delete the .venv folder and run "run.bat --setup" again.
)
goto :error


:install_error
echo [ERROR] Dependency installation failed.
echo [HINT] Check your internet connection and confirm that Python 3.13+ is installed.
echo [HINT] This setup requires a compatible scikit-image wheel for your Python version and architecture.
goto :error


:invalid_option
echo [ERROR] Unknown option: %~1
echo Usage: run.bat [--setup]
goto :error


:help
echo Usage:
echo   run.bat          Start the application using an existing Python environment.
echo   run.bat --setup  Create or update .venv and install project dependencies.
exit /b 0


:error
echo.
echo [ERROR] Setup or execution failed.
pause
exit /b 1
