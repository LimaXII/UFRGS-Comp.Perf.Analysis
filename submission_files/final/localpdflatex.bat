@echo off
:: Change to script directory so relative paths in the .tex file resolve correctly
pushd "%~dp0"
:: Set TEXINPUTS so acmart.cls in the local template/ folder is found
set TEXINPUTS=%~dp0\..\..\template;;
:: Determine basename of first argument (if provided)
if "%~1"=="" (
	set TEXFILE=relatorio_final.tex
) else (
	set TEXFILE=%~nx1
)
"C:\Users\lucca\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" "%TEXFILE%"
popd
