# Start Qdrant
$qdrant = docker ps --filter "name=qdrant" --format "{{.Names}}"
if (-not $qdrant) {
    Write-Host "Starting Qdrant..."
    docker start qdrant
} else {
    Write-Host "Qdrant already running"
}

# Start FastAPI
Write-Host "Starting FastAPI..."
Start-Process -NoNewWindow python -ArgumentList "-m uvicorn api.main:app --port 8001 --env-file .env" -WorkingDirectory "D:\stockrag_system\backend"

# Start Frontend
Write-Host "Starting Frontend..."
Start-Process -NoNewWindow cmd -ArgumentList "/c npm run dev" -WorkingDirectory "D:\stockrag_system\frontend"

Write-Host "All services started. Open http://localhost:3000"