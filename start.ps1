# Start Qdrant if not already running
$qdrant = docker ps --filter "name=qdrant" --format "{{.Names}}"
if (-not $qdrant) {
    Write-Host "Starting Qdrant..."
    docker start qdrant
} else {
    Write-Host "Qdrant already running"
}

# Start FastAPI
Write-Host "Starting FastAPI..."
Start-Process -NoNewWindow python -ArgumentList "-m uvicorn api.main:app --host 0.0.0.0 --port 8001 --env-file .env" -WorkingDirectory "D:\astrorag\backend"

# Start Frontend
Write-Host "Starting Frontend..."
Start-Process -NoNewWindow cmd -ArgumentList "/c npm run dev" -WorkingDirectory "D:\astrorag\frontend"

Write-Host "All services started. Open http://localhost:3000"
