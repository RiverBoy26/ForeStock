$ErrorActionPreference = 'Stop'

$serviceRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $serviceRoot 'model_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

foreach ($sectionName in @('artifact_sha256', 'forecast_sha256')) {
    foreach ($property in $manifest.$sectionName.PSObject.Properties) {
        $artifactPath = Join-Path $serviceRoot ($property.Name -replace '/', '\')
        if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
            throw "Missing release artifact: $($property.Name)"
        }

        $property.Value = (Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash.ToLower()
    }
}

$json = $manifest | ConvertTo-Json -Depth 20
$utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($manifestPath, $json + [Environment]::NewLine, $utf8WithoutBom)

Write-Output 'model_manifest.json hashes updated'
