param(
  [string]$AppDir = (Get-Location).Path,
  [string]$Models = "realesr-general-x4v3,realesrgan-x4plus,realesrgan-x4plus-anime,swinir-4x,waifu2x-anime"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = 'SilentlyContinue'

$modelsDir = Join-Path $AppDir "models"
if (!(Test-Path $modelsDir)) { New-Item -ItemType Directory -Path $modelsDir | Out-Null }

function Download-WithRetry($url, $outPath, $retries=3) {
  for ($i=1; $i -le $retries; $i++) {
    try {
      Invoke-WebRequest -Uri $url -OutFile $outPath -UseBasicParsing
      return
    } catch {
      if ($i -eq $retries) { throw }
      Start-Sleep -Seconds (2 * $i)
    }
  }
}

function Extract-First($zipPath) {
  $tmp = New-Item -ItemType Directory -Path ([System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), [System.IO.Path]::GetRandomFileName()))
  Expand-Archive -Path $zipPath -DestinationPath $tmp.FullName -Force
  return $tmp.FullName
}

# Known sources (multiple fallbacks per model)
$sources = @{
  "realesr-general-x4v3" = @(
    @{ url = "https://github.com/nihui/realesr-ncnn-vulkan/releases/download/20220416/realesr-ncnn-vulkan-20220416-windows.zip"; pick="realesr-general-x4v3" },
    @{ url = "https://github.com/nihui/realesr-ncnn-vulkan/releases/latest/download/realesr-ncnn-vulkan-20220416-windows.zip"; pick="realesr-general-x4v3" }
  );
  "realesrgan-x4plus" = @(
    @{ url = "https://github.com/nihui/realesrgan-ncnn-vulkan/releases/download/20220424/realesrgan-ncnn-vulkan-20220424-windows.zip"; pick="realesrgan-x4plus" },
    @{ url = "https://github.com/nihui/realesrgan-ncnn-vulkan/releases/latest/download/realesrgan-ncnn-vulkan-20220424-windows.zip"; pick="realesrgan-x4plus" }
  );
  "realesrgan-x4plus-anime" = @(
    @{ url = "https://github.com/nihui/realesrgan-ncnn-vulkan/releases/download/20220424/realesrgan-ncnn-vulkan-20220424-windows.zip"; pick="realesrgan-x4plus-anime" },
    @{ url = "https://github.com/nihui/realesrgan-ncnn-vulkan/releases/latest/download/realesrgan-ncnn-vulkan-20220424-windows.zip"; pick="realesrgan-x4plus-anime" }
  );
  "swinir-4x" = @(
    @{ url = "https://github.com/nihui/swinir-ncnn-vulkan/releases/download/20220728/swinir-ncnn-vulkan-20220728-windows.zip"; pick="models-4x" },
    @{ url = "https://github.com/nihui/swinir-ncnn-vulkan/releases/latest/download/swinir-ncnn-vulkan-20220728-windows.zip"; pick="models-4x" }
  );
  "waifu2x-anime" = @(
    @{ url = "https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20220728/waifu2x-ncnn-vulkan-20220728-windows.zip"; pick="models-cunet" },
    @{ url = "https://github.com/nihui/waifu2x-ncnn-vulkan/releases/latest/download/waifu2x-ncnn-vulkan-20220728-windows.zip"; pick="models-cunet" }
  )
}

$requested = $Models.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }

foreach ($name in $requested) {
  $target = Join-Path $modelsDir $name
  if (Test-Path $target) {
    Write-Host "[OK] $name already present at $target"
    continue
  }

  if (-not $sources.ContainsKey($name)) {
    Write-Warning "No source mapping for $name. Skipping."
    continue
  }

  $got = $false
  foreach ($entry in $sources[$name]) {
    try {
      $url = $entry.url
      $pick = $entry.pick
      Write-Host "Fetching $name from $url ..."
      $zip = Join-Path $env:TEMP ("fv_mod_" + [guid]::NewGuid().ToString() + ".zip")
      Download-WithRetry $url $zip 3
      $root = Extract-First $zip

      # find the 'models' folder under extracted root
      $modelsRoot = Get-ChildItem -Recurse -Directory -Path $root | Where-Object { $_.Name -eq "models" } | Select-Object -First 1
      if (-not $modelsRoot) { throw "No models directory in archive." }

      # find subdir that matches pick
      $pickDir = Get-ChildItem -Recurse -Directory -Path $modelsRoot.FullName | Where-Object { $_.Name -eq $pick } | Select-Object -First 1
      if (-not $pickDir) { throw "Wanted subdir '$pick' not found." }

      # copy to target (rename if necessary)
      if ($pick -ne $name) {
        New-Item -ItemType Directory -Path $target | Out-Null
        Copy-Item -Path (Join-Path $pickDir.FullName "*") -Destination $target -Recurse -Force
      } else {
        Copy-Item -Path $pickDir.FullName -Destination $target -Recurse -Force
      }

      # compute size
      $size = (Get-ChildItem -Recurse -File $target | Measure-Object -Property Length -Sum).Sum
      $gb = [Math]::Round($size/1GB, 3)
      Write-Host "[OK] $name ready at $target ($gb GB)."
      $got = $true
      break
    } catch {
      Write-Warning "Attempt failed for $name: $($_.Exception.Message)"
    } finally {
      if ($root -and (Test-Path $root)) { Remove-Item -Recurse -Force $root }
      if (Test-Path $zip) { Remove-Item -Force $zip }
    }
  }

  if (-not $got) {
    Write-Warning "[SKIP] Could not fetch $name from known sources."
  }
}

Write-Host "`nDone fetching models."
