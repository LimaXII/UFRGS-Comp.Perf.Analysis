# Build script for the final report
# Usage: .\build-relatorio.ps1
try {
    $root = (Get-Location).ProviderPath
    $wrapper = Join-Path $root "submission_files\final\localpdflatex.bat"
    $tex = Join-Path $root "submission_files\final\relatorio_final.tex"
    $outpdf = Join-Path $root "submission_files\final\relatorio_final.pdf"

    Write-Host "Removing previous PDF if unlocked..."
    Remove-Item $outpdf -Force -ErrorAction SilentlyContinue

    if (-not (Test-Path $wrapper)) {
        Write-Error "Wrapper not found: $wrapper"
        exit 2
    }

    Write-Host "Compiling LaTeX via wrapper: $wrapper"
    & $wrapper $tex
    if ($LASTEXITCODE -ne 0) {
        Write-Host "pdflatex returned non-zero exit code ($LASTEXITCODE). Check submission_files/final/relatorio_final.log for details." -ForegroundColor Yellow
        exit $LASTEXITCODE
    }
    Write-Host "Compilation finished. Output: submission_files/final/relatorio_final.pdf"
} catch {
    Write-Error "Build failed: $_"
    exit 1
}
