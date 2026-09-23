@echo off
rem Example launcher for the merged Qwen3.8-Flash-Next IQ2 DS4 pack on Windows + ROCm (gfx1151).
rem Edit the three paths. Keep the model on an NVMe drive: -lzm on-direct reads the 29.8 GiB PLE table from disk.
rem See README.md in this folder for how each flag was chosen and what was measured.

set "SERVER=C:\path\to\build\bin\llama-server.exe"
set "MODEL=C:\path\to\Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-DenseQ4K-MTP-PLEmerged.gguf"
set "TEMPLATE=C:\path\to\sharp-v22.5.0.jinja"

"%SERVER%" ^
  -m "%MODEL%" ^
  -lzm on-direct ^
  --jinja ^
  --chat-template-file "%TEMPLATE%" ^
  --reasoning-format deepseek ^
  --reasoning-preserve ^
  --n-predict 16384 ^
  --temp 0.60 ^
  --top-k 20 ^
  --top-p 0.95 ^
  --min-p 0.00 ^
  --repeat-penalty 1.05 ^
  -ngl 999 ^
  -c 262144 ^
  --ctx-checkpoints 8 ^
  --cache-ram 16384 ^
  -np 1 ^
  -fa on ^
  -b 8192 ^
  -ub 8192 ^
  --spec-type draft-mtp ^
  --spec-draft-n-max 2 ^
  --spec-draft-p-min 0.40 ^
  --host 127.0.0.1 ^
  --port 8080

pause
