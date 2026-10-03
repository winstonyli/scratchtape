# Samples machine load every 30 s while a process runs: load_log.ps1 <process name> <log>. CPU %, and the processes with
# nonzero engine activity on the 9060 XT (all engines), per LONG_RUNS.md.
param($Name, $Log)
$luid = Get-ChildItem 'HKLM:\SOFTWARE\Microsoft\DirectX' | % { Get-ItemProperty $_.PSPath } | ? { $_.Description -like '*9060 XT*' -and ($_.AdapterLuid -band 0xffffffff) } | sort LastSeen -Desc | select -First 1 | % { '0x{0:x8}' -f ($_.AdapterLuid -band 0xffffffff) }
"defender realtime: $((Get-MpComputerStatus).RealTimeProtectionEnabled)" | Out-File $Log -Encoding utf8
while (Get-Process $Name -EA 0) {
  $cpu = [int](Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 2 -MaxSamples 1).CounterSamples[0].CookedValue
  $pids = (Get-Counter "\GPU Engine(*$luid*)\Utilization Percentage" -SampleInterval 2 -MaxSamples 1 -EA 0).CounterSamples | ? CookedValue -gt 0 | % { [int]($_.InstanceName -split '_')[1] } | sort -Unique | % { (Get-Process -Id $_ -EA 0).ProcessName }
  "$(Get-Date -Format HH:mm:ss) cpu $cpu gpu-users $($pids -join ',')" | Out-File $Log -Append -Encoding utf8
  Start-Sleep 26
}
