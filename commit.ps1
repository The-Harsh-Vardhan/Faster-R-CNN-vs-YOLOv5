$proj = 'c:\Users\harsh\OneDrive - Indian Institute of Information Technology, Nagpur\IIIT Nagpur\6th Semester\2. Computer Vision and Deep Learning (CVDL) [CSL 313]\Project'
Set-Location -LiteralPath $proj

# Remove cleanup.ps1 from git tracking and disk
git rm --cached cleanup.ps1
Remove-Item -LiteralPath (Join-Path $proj 'cleanup.ps1') -Force
Write-Host "Removed cleanup.ps1"

# Stage everything
git add -A

# Commit
$msg = @"
chore: GitHub-ready repo structure

- Add comprehensive README.md with badges, results tables, architecture overview, and team info
- Update .gitignore (venv, Jupyter checkpoints, LaTeX build artifacts, zips, model weights, Extras/)
- Create Extras/ folder and archive intermediate report versions (v2-v6 PDFs + .tex), zip archives, raw notebook output cache (__results___files), and early experimental notebook
- Remove empty 'zip files' directory
- Keep only final report (v7.tex + CVDL_Report_6th_Semester.pdf) and clean notebook set
"@

git commit -m $msg
Write-Host "--- Commit done ---"
git log --oneline -3
