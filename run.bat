@echo off
setlocal

rem Always run from the repository root.
cd /d "%~dp0"
if errorlevel 1 goto :error

set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_CMD="

rem The default is the system Python. Virtual environments are opt-in.
if /i "%~1"=="--help" goto :help
if /i "%~1"=="--setup" goto :setup
if /i "%~1"=="--venv" goto :run_venv
if not "%~1"=="" goto :invalid_option

goto :run_system


:run_system
call :find_python
if errorlevel 1 goto :error
goto :run_application


:run_venv
if not "%~2"=="" goto :invalid_option
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


:setup
if not "%~2"=="" goto :invalid_option

rem Create a virtual environment only when setup is explicitly requested.
if exist "%VENV_PYTHON%" goto :setup_existing

call :find_python
if errorlevel 1 goto :error

echo [INFO] Creating the virtual environment...
if exist "%VENV_DIR%" (
    %PYTHON_CMD% -m venv --clear "%VENV_DIR%"
) else (
    %PYTHON_CMD% -m venv "%VENV_DIR%"
)
if errorlevel 1 goto :error
set PYTHON_CMD="%VENV_PYTHON%"
goto :setup_check

:setup_existing
set PYTHON_CMD="%VENV_PYTHON%"

:setup_check
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
echo [INFO] Run "run.bat --venv" to use this environment.
echo [INFO] Double-clicking run.bat continues to use the system Python.
pause
exit /b 0


:run_application
echo [INFO] Starting run.py...
%PYTHON_CMD% "%~dp0run.py"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] run.py exited with code %EXIT_CODE%.
    echo [HINT] If you want to use the project virtual environment, run "run.bat --setup" first.
    echo [HINT] Then launch it with "run.bat --venv".
) else (
    echo.
    echo [INFO] Application finished.
)

pause
exit /b %EXIT_CODE%


:find_python
rem Use the Python command available on PATH. Do not activate or select .venv here.
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

echo [ERROR] Python 3.13 or later was not found.
echo         Install Python 3.13+ and make sure python or py is available in PATH.
exit /b 1


:venv_not_ready
echo.
echo [ERROR] The virtual environment is missing or incomplete.
echo [HINT] No packages were installed or modified during this launch.
echo [HINT] Run "run.bat --setup" to prepare or repair .venv, then retry "run.bat --venv".
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
echo [ERROR] Unknown option: %~1
echo Usage: run.bat [--venv^|--setup^|--help]
goto :error


:help
echo Usage:
echo   run.bat          Run with the system Python (default).
echo   run.bat --venv   Run with the prepared .venv environment.
echo   run.bat --setup  Create or update .venv with project dependencies.
exit /b 0


:error
echo.
echo [ERROR] Setup or execution failed.
pause
exit /b 1
