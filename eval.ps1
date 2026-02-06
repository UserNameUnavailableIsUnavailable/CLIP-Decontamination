param(
    [Parameter(Mandatory=$false, Position=0)]
    [String[]]$Datasets,
    [String]$WorkDir = "work-dirs",
    [String]$ShowDir = "show-dirs",
    [Parameter(Mandatory=$false)]
    [Switch]$All
)

enum DatasetType {
    LoveDA
    Potsdam
    Vaihingen
    iSAID
    UDD5
    VDD
    xBD
    WHU_Building
    Inria
}

$ErrorActionPreference = "Stop"

if ($All) {
    $Datasets = [DatasetType].GetEnumNames()
}

if ($Datasets.Count -eq 0) {
    Write-Error "Please specify at least one dataset or use the -All flag."
    exit 1
}

$Datasets = $Datasets | ForEach-Object { [DatasetType]::Parse([DatasetType], $_) }

Write-Host "Evaluating datasets: $($Datasets -join ', ')."

function EvaluateDataset {
    param([DatasetType]$DatasetName)
    $cfg = GetDatasetConfig -Name $DatasetName.ToString()
    $work_dir = Join-Path $WorkDir $DatasetName.ToString()
    $show_dir = Join-Path $ShowDir $DatasetName.ToString()
    python eval.py --config $cfg --work-dir $work_dir --save-seg-dir "$show_dir/segs" --save-heatmap-dir "$show_dir/heatmaps"
}
function GetDatasetConfig {
    param(
        [String]$Name
    )
    return (Join-Path "configs" ("cfg_" + $Name.ToLower() + ".py")).ToString()
}

$Datasets | ForEach-Object {
    EvaluateDataset -DatasetName $_
}
