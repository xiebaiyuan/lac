@echo off
setlocal

echo Checking Paddle Inference DLL...

set DLL_PATH=D:\workspace\paddle_inference_win_x86_mkl\paddle\lib\paddle_inference.dll

if not exist "%DLL_PATH%" (
    echo ERROR: DLL file does not exist at path: %DLL_PATH%
    exit /b 1
)

echo DLL exists. File information:
dir "%DLL_PATH%"

echo.
echo Running dumpbin to check DLL validity...
where dumpbin > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: dumpbin not found in PATH. Make sure Visual Studio tools are installed.
    echo You need Visual Studio with C++ tools installed.
) else (
    echo DLL headers:
    dumpbin /HEADERS "%DLL_PATH%" | findstr /C:"machine" /C:"Magic" /C:"linker version" /C:"size of code"
    
    echo.
    echo DLL exports:
    dumpbin /EXPORTS "%DLL_PATH%" | findstr /C:"ordinal" /C:"Function" /C:"paddle" /B
)

echo.
echo Checking if DLL is 32-bit or 64-bit...
call :CheckArch "%DLL_PATH%"

echo.
echo Testing to open the DLL with LoadLibrary...
powershell -Command "$ErrorActionPreference = 'SilentlyContinue'; Add-Type -AssemblyName Microsoft.VisualBasic; [Microsoft.VisualBasic.Interaction]::MsgBox('Testing to load DLL - press OK', 'OkOnly,SystemModal', 'DLL Test'); $handle = [System.Runtime.InteropServices.LoadLibrary]::LoadLibrary('$DLL_PATH'); if ($handle -eq [System.IntPtr]::Zero) { 'Failed to load DLL' } else { 'Successfully loaded DLL'; [System.Runtime.InteropServices.LoadLibrary]::FreeLibrary($handle) }"

goto :EOF

:CheckArch
powershell -Command "$sig = New-Object byte[] 2; $fs = New-Object System.IO.FileStream('%~1', [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read); $fs.Read($sig, 0, 2) | Out-Null; $fs.Close(); if ($sig[0] -eq 0x4d -and $sig[1] -eq 0x5a) { $fs = New-Object System.IO.FileStream('%~1', [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read); $fs.Seek(0x3c, [System.IO.SeekOrigin]::Begin) | Out-Null; $byte = $fs.ReadByte(); $byte2 = $fs.ReadByte(); $offset = $byte + $byte2*256; $fs.Seek($offset + 4, [System.IO.SeekOrigin]::Begin) | Out-Null; $arch = $fs.ReadByte() + $fs.ReadByte()*256; $fs.Close(); if ($arch -eq 0x014c) { 'DLL is 32-bit' } elseif ($arch -eq 0x8664) { 'DLL is 64-bit' } else { 'Unknown architecture: 0x' + $arch.ToString('X4') } } else { 'Not a valid PE file' }"
exit /b