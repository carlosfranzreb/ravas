@ECHO OFF
REM store current working dir in CWD:
FOR /F "tokens=*" %%g IN ('cd') do (SET CWD=%%g)

set EXT_PATH="%CWD%\dist\chrome-extension"
set PEM_PATH="%CWD%\resources\chrome-extension-packing\privkey.pem"
@ECHO ON

chrome.exe --pack-extension=%EXT_PATH% --pack-extension-key=%PEM_PATH%
