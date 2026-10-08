# Block time at 5M keys for TILE x KNN_SLICE settings (knn_scale_check QUICK=1; copy target/release/examples/knn_scale_check.exe to runs/ first). Run from the repo root.
$env:QUICK = '1'
foreach ($c in @(@(16384,4096), @(16384,8192), @(16384,16384), @(65536,4096), @(65536,16384), @(65536,65536), @(16384,4096))) {
  $env:TILE = $c[0]; $env:KNN_SLICE = $c[1]
  $cpu = [int](Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 2 -MaxSamples 1).CounterSamples[0].CookedValue
  $out = & .\runs\knn_scale_check.exe 2>&1 | Select-String 'ladder' | % { $_.Line }
  "tile $($c[0]) slice $($c[1]) cpu $cpu : $out"
}
