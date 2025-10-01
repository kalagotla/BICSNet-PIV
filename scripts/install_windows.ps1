# BICSNet-PIV Windows PowerShell Installation Script
# This script sets up BICSNet-PIV on Windows with automatic PyTorch GPU/CPU detection

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " BICSNet-PIV Windows Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.12 first." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if uv is available
try {
    $uvVersion = uv --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "uv not found"
    }
    Write-Host "✅ uv found: $uvVersion" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "❌ uv package manager not found." -ForegroundColor Red
    Write-Host "Installing uv..." -ForegroundColor Yellow
    Write-Host ""
    
    try {
        # Install uv using the official installer
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        Write-Host "✅ uv installed successfully" -ForegroundColor Green
        
        # Refresh PATH for current session
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
        
        # Verify installation
        $uvVersion = uv --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "uv installation verification failed"
        }
        Write-Host "✅ uv verified: $uvVersion" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to install uv." -ForegroundColor Red
        Write-Host "Please install uv manually from: https://github.com/astral-sh/uv/releases" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Create virtual environment
Write-Host ""
Write-Host "📦 Creating Python 3.12 virtual environment..." -ForegroundColor Blue
try {
    uv venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment"
    }
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create virtual environment." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host ""
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Blue
& .\.venv\Scripts\Activate.ps1

# Install PyTorch with automatic detection
Write-Host ""
Write-Host "🚀 Installing PyTorch with automatic GPU/CPU detection..." -ForegroundColor Blue
try {
    python scripts\install_pytorch.py
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch installation failed"
    }
    Write-Host "✅ PyTorch installation completed" -ForegroundColor Green
} catch {
    Write-Host "❌ PyTorch installation failed." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install remaining dependencies
Write-Host ""
Write-Host "📚 Installing remaining dependencies..." -ForegroundColor Blue
try {
    uv sync
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install dependencies"
    }
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install dependencies." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Test installation
Write-Host ""
Write-Host "🧪 Testing installation..." -ForegroundColor Blue
try {
    python scripts\test_pytorch_install.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ All tests passed!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Some tests failed, but installation may still work." -ForegroundColor Yellow
        Write-Host "You can continue with the setup." -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Test script encountered an error, but installation may still work." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Installation Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Activate the virtual environment: .venv\Scripts\activate" -ForegroundColor Gray
Write-Host "2. Run inference: python src\pivnet_image_gen.py" -ForegroundColor Gray
Write-Host "3. Launch Jupyter: jupyter lab" -ForegroundColor Gray
Write-Host ""
Write-Host "For troubleshooting, see the README.md file." -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to exit"
