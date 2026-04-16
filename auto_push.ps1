# ============================
# AUTO GITHUB PUSH SCRIPT
# ============================

Write-Host "🔄 Starting auto push..."

# Add all changes
git add .

# Commit with timestamp
$time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
git commit -m "Auto update: $time"

# Push to GitHub
git push origin ultimate-upgrade

Write-Host "✅ Push completed!"