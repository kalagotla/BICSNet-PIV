@echo off
REM BICSNet-PIV Windows Installation Script
REM This script sets up BICSNet-PIV on Windows with automatic PyTorch GPU/CPU detection

echo.
echo ========================================
echo  BICSNet-PIV Windows Installation
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.12 first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if uv is available
uv --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ❌ uv package manager not found.
    echo Installing uv...
    echo.
    
    REM Try PowerShell installation
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex" 2>nul
    if errorlevel 1 (
        echo ❌ Failed to install uv via PowerShell.
        echo Please install uv manually from: https://github.com/astral-sh/uv/releases
        pause
        exit /b 1
    )
    
    REM Refresh PATH
    call refreshenv.cmd 2>nul || (
        echo Please restart your command prompt or add uv to your PATH manually.
        echo uv should be installed in: %USERPROFILE%\.cargo\bin
    )
)

echo ✅ uv found
uv --version

REM Create virtual environment
echo.
echo 📦 Creating Python 3.12 virtual environment...
uv venv --python 3.12 .venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment.
    pause
    exit /b 1
)

REM Activate virtual environment
echo.
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install PyTorch with automatic detection
echo.
echo 🚀 Installing PyTorch with automatic GPU/CPU detection...
python scripts\install_pytorch.py
if errorlevel 1 (
    echo ❌ PyTorch installation failed.
    pause
    exit /b 1
)

REM Install remaining dependencies
echo.
echo 📚 Installing remaining dependencies...
uv sync
if errorlevel 1 (
    echo ❌ Failed to install dependencies.
    pause
    exit /b 1
)

REM Test installation
echo.
echo 🧪 Testing installation...
python scripts\test_pytorch_install.py
if errorlevel 1 (
    echo ⚠️  Some tests failed, but installation may still work.
    echo You can continue with the setup.
) else (
    echo ✅ All tests passed!
)

echo.
echo ========================================
echo  Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Activate the virtual environment: .venv\Scripts\activate
echo 2. Run inference: python src\pivnet_image_gen.py
echo 3. Launch Jupyter: jupyter lab
echo.
echo For troubleshooting, see the README.md file.
echo.
pause
