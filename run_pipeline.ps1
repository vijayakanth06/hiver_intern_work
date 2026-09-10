<#
.SYNOPSIS
    Windows PowerShell Pipeline Runner for Hiver AI Customer Support Agent.
.DESCRIPTION
    Executes the end-to-end ingestion, golden dataset curation, hybrid indexing,
    ablation evaluation, and test suite on Windows systems.
#>

param (
    [string]$Brand = "amazonhelp",
    [int]$SampleSize = 200,
    [switch]$SkipIngest,
    [switch]$SkipEval,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "🚀 HIVER AI CUSTOMER SUPPORT AGENT — WINDOWS POWERSHELL RUNNER" -ForegroundColor Cyan
Write-Host "Target Brand: $Brand | Sample Size: $SampleSize" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

# Detect Python
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    $PythonExe = (Get-Command py -ErrorAction SilentlyContinue).Source
}
if (-not $PythonExe) {
    Write-Error "Python was not found in PATH. Please install Python 3.10+ and add it to PATH."
}

Write-Host "Using Python: $PythonExe" -ForegroundColor Green

# 1. Ingestion
if (-not $SkipIngest) {
    Write-Host "`n[Step 1/5] Ingesting TWCS and reconstructing conversation threads..." -ForegroundColor Yellow
    & $PythonExe -m src.ingest --brand $Brand
}

# 2. Golden Set Curation
Write-Host "`n[Step 2/5] Curating verified Golden Set with audit trail..." -ForegroundColor Yellow
& $PythonExe -m src.label_golden --brand $Brand --sample-size $SampleSize

# 3. Hybrid Indexing
Write-Host "`n[Step 3/5] Building Hybrid Vector Index (FAISS + BM25)..." -ForegroundColor Yellow
& $PythonExe -c "from src.retrieve import HybridRetriever; from src.config import PROJECT_ROOT; r = HybridRetriever('$Brand'); r.build_index(PROJECT_ROOT / 'data' / 'processed' / '$Brand' / 'rag_corpus.jsonl', PROJECT_ROOT / 'indices' / '$Brand')"

# 4. Evaluation
if (-not $SkipEval) {
    Write-Host "`n[Step 4/5] Running 10-Way Ablation Benchmark..." -ForegroundColor Yellow
    & $PythonExe -m eval.ablation_study --brand $Brand --sample-size ([Math]::Min(50, $SampleSize))
}

# 5. Tests
if (-not $SkipTests) {
    Write-Host "`n[Step 5/5] Running Pytest test suite..." -ForegroundColor Yellow
    & $PythonExe -m pytest tests/ -v
}

Write-Host "`n==============================================================================" -ForegroundColor Cyan
Write-Host "✅ PIPELINE RUN COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan
