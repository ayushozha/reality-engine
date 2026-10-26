$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$base = $PSScriptRoot
$project = (Resolve-Path (Join-Path $base '../..')).Path
$assets = Join-Path $base 'assets'
$segments = Join-Path $base 'segments'
New-Item -ItemType Directory -Path $segments -Force | Out-Null
$ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
$ffprobe = (Get-Command ffprobe -ErrorAction Stop).Source
$fontName = 'Segoe UI'
$narration = Join-Path $base 'narration-elevenlabs.mp3'
if (-not (Test-Path -LiteralPath $narration)) {
    throw 'Missing approved ElevenLabs narration: narration-elevenlabs.mp3'
}

function Paint-Image([string]$Path, [string]$Title, [string[]]$Lines, [bool]$Overlay = $false) {
    $bitmap = New-Object System.Drawing.Bitmap 1280,720
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $white = [System.Drawing.Brushes]::White
    $muted = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(190,202,214))
    $accent = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(133,229,187))
    $background = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(230,16,23,32))
    $titleFont = New-Object System.Drawing.Font $fontName,44,([System.Drawing.FontStyle]::Bold),([System.Drawing.GraphicsUnit]::Pixel)
    $bodyFont = New-Object System.Drawing.Font $fontName,28,([System.Drawing.FontStyle]::Regular),([System.Drawing.GraphicsUnit]::Pixel)
    $smallFont = New-Object System.Drawing.Font $fontName,21,([System.Drawing.FontStyle]::Regular),([System.Drawing.GraphicsUnit]::Pixel)
    try {
        if ($Overlay) {
            $graphics.Clear([System.Drawing.Color]::Transparent)
            $graphics.FillRectangle($background,0,0,1280,94)
            $graphics.DrawString($Title,$bodyFont,$white,38,29)
        } else {
            $graphics.Clear([System.Drawing.Color]::FromArgb(16,23,32))
            $graphics.FillRectangle($accent,64,115,84,5)
            $graphics.DrawString($Title,$titleFont,$white,60,155)
            $y = 260
            foreach ($line in $Lines) {
                $graphics.DrawString($line,$bodyFont,$muted,64,$y)
                $y += 65
            }
            $graphics.DrawString('Gemini 4 Argon | More Dimensions',$smallFont,$accent,64,585)
        }
        $graphics.FillRectangle($background,0,665,1280,55)
        $graphics.DrawString('Prototype: coarse geometry; measured cameras and points; editable objects.',$smallFont,$muted,38,682)
        $bitmap.Save($Path,[System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $titleFont.Dispose(); $bodyFont.Dispose(); $smallFont.Dispose()
        $muted.Dispose(); $accent.Dispose(); $background.Dispose()
        $graphics.Dispose(); $bitmap.Dispose()
    }
}

Paint-Image (Join-Path $base 'intro.png') 'Reality Engine' @('Ayush + Kush','Video to a measured, editable 3D scene.','A real walkthrough. A real processing run.')
Paint-Image (Join-Path $base 'pipeline.png') 'From walkthrough to editable space' @('1  Stable frames + measured camera poses','2  Gemini reads the video and builds zones','3  Render, compare, critique and edit','Completed real-video run — processing condensed')

$capture = Join-Path $base 'capture/editor.webm'
$deadline = (Get-Date).AddSeconds(30)
while (-not (Test-Path -LiteralPath $capture) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 2 }
$hasCapture = Test-Path -LiteralPath $capture
$editor = if ($hasCapture) { $capture } else { Join-Path $assets 'real-overview-03.png' }
$exportOffset = 0.0
if ($hasCapture) {
    $seconds = & $ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $capture
    if ($LASTEXITCODE -ne 0) { throw 'Editor capture is not a readable video yet.' }
    $exportOffset = [Math]::Max(0,[double]::Parse(($seconds -join '').Trim(),[Globalization.CultureInfo]::InvariantCulture)-9)
}
$scenes = @(
    @{Name='01-intro'; Input=(Join-Path $base 'intro.png'); Seconds=8; Still=$true; Label='Reality Engine | Ayush + Kush'},
    @{Name='02-phone'; Input=(Join-Path $project 'runs/room-walkthrough/input.mov'); Seconds=14; Still=$false; Label='Original phone walkthrough | real input video'},
    @{Name='03-pipeline'; Input=(Join-Path $base 'pipeline.png'); Seconds=16; Still=$true; Label='Completed real-video run — processing condensed'},
    @{Name='04-comparison'; Input=(Join-Path $assets 'real-comparison-02.png'); Seconds=17; Still=$true; Label='Real image vs rendered replica | verification output'},
    @{Name='05-overview'; Input=(Join-Path $assets 'real-overview-01.png'); Seconds=20; Still=$true; Label='Reconstructed room | 16 editable objects | ceiling hidden'},
    @{Name='06-editor'; Input=$editor; Seconds=24; Still=(-not $hasCapture); Label=$(if($hasCapture){'Actual editor | select, edit, undo and compare'}else{'Editable scene overview | captured real run'})},
    @{Name='07-exports'; Input=$(if($hasCapture){$capture}else{Join-Path $assets 'real-overview-02.png'}); Seconds=9; Still=(-not $hasCapture); Offset=$exportOffset; Label='Actual editor | glTF and MJCF export controls'},
    @{Name='08-outro'; Input=(Join-Path $assets 'real-comparison-03.png'); Seconds=12; Still=$true; Label='Reality Engine | Gemini 4 Argon | More Dimensions'}
)

$manifest = @()
foreach ($scene in $scenes) {
    if (-not (Test-Path -LiteralPath $scene.Input)) { throw "Missing input: $($scene.Input)" }
    $overlay = Join-Path $segments ($scene.Name + '-label.png')
    Paint-Image $overlay $scene.Label @() $true
    $output = Join-Path $segments ($scene.Name + '.mp4')
    $arguments = @('-hide_banner','-loglevel','error','-y')
    if ($scene.Still) { $arguments += @('-loop','1','-framerate','24') }
    if ($scene.Offset -and -not $scene.Still) { $arguments += @('-ss',([double]$scene.Offset).ToString('0.###',[Globalization.CultureInfo]::InvariantCulture)) }
    $arguments += @('-i',$scene.Input,'-loop','1','-framerate','24','-i',$overlay)
    $filter = '[0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x101720,setsar=1,fps=24,tpad=stop_mode=clone:stop_duration=24[base];[base][1:v]overlay=0:0,format=yuv420p[v]'
    $arguments += @('-filter_complex',$filter,'-map','[v]','-an','-frames:v',([int]$scene.Seconds*24).ToString(),'-r','24','-c:v','libx264','-preset','ultrafast','-crf','20','-pix_fmt','yuv420p','-threads','4',$output)
    & $ffmpeg @arguments
    if ($LASTEXITCODE -ne 0) { throw "Encoding failed: $($scene.Name)" }
    $manifest += "file '$($scene.Name).mp4'"
    Write-Host "Rendered $($scene.Name) ($($scene.Seconds)s)"
}
$list = Join-Path $segments 'concat.txt'
[IO.File]::WriteAllLines($list,$manifest,[Text.UTF8Encoding]::new($false))
$final = Join-Path $base 'Reality_Engine_2min.mp4'
& $ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i $list -i $narration -map 0:v:0 -map 1:a:0 -c:v copy -af 'apad=whole_dur=120,atrim=duration=120,asetpts=PTS-STARTPTS' -c:a aac -b:a 192k -t 120 -movflags +faststart $final
if ($LASTEXITCODE -ne 0) { throw 'Final audio/video mux failed.' }
& $ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate -of json $final
Write-Host "Ready: $final"
