$ErrorActionPreference = 'Stop'
$base = $PSScriptRoot
$source = Join-Path $base 'Reality_Engine_2min.mp4'
$narration = Join-Path $base 'narration-1min-elevenlabs.mp3'
$segments = Join-Path $base 'segments/one-minute'
New-Item -ItemType Directory -Path $segments -Force | Out-Null
foreach ($inputFile in @($source, $narration)) {
    if (-not (Test-Path -LiteralPath $inputFile)) { throw "Missing input: $inputFile" }
}
$ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
$ffprobe = (Get-Command ffprobe -ErrorAction Stop).Source

# Select existing footage and labels; preserve the original two-minute video.
$shots = @(
    @{ Name='01-intro'; Start=0; Seconds=4 },
    @{ Name='02-phone'; Start=8; Seconds=8 },
    @{ Name='03-starting-point'; Start=55; Seconds=6 },
    @{ Name='04-pipeline'; Start=22; Seconds=8 },
    @{ Name='05-comparison'; Start=38; Seconds=8 },
    @{ Name='06-reconstruction'; Start=61; Seconds=9 },
    @{ Name='07-editor'; Start=78; Seconds=10 },
    @{ Name='08-outro'; Start=108; Seconds=7 }
)
$manifest = @()
foreach ($shot in $shots) {
    $fileName = $shot.Name + '.mp4'
    & $ffmpeg -hide_banner -loglevel error -y -ss $shot.Start -i $source -map 0:v:0 -an -vf 'setpts=PTS-STARTPTS,fps=24,format=yuv420p' -frames:v ($shot.Seconds * 24) -c:v libx264 -preset ultrafast -crf 20 -threads 4 (Join-Path $segments $fileName)
    if ($LASTEXITCODE -ne 0) { throw "Encoding failed: $($shot.Name)" }
    $manifest += "file '$fileName'"
}
$list = Join-Path $segments 'concat.txt'
[IO.File]::WriteAllLines($list, $manifest, [Text.UTF8Encoding]::new($false))
$final = Join-Path $base 'Reality_Engine_1min.mp4'
& $ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i $list -i $narration -map 0:v:0 -map 1:a:0 -c:v copy -af 'apad=whole_dur=60,atrim=duration=60,asetpts=PTS-STARTPTS' -c:a aac -b:a 192k -t 60 -movflags +faststart $final
if ($LASTEXITCODE -ne 0) { throw 'Final one-minute audio/video mux failed.' }
& $ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate -of json $final
if ($LASTEXITCODE -ne 0) { throw 'Final video verification failed.' }
Write-Host "Ready: $final"
