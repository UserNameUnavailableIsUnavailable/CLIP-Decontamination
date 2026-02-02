$config_dir_base = "configs"
$work_dir_base = "work-dir"
$show_dir_base = "show-dir"

$configs = @(
    "cfg_isaid",
    "cfg_loveda",
    "cfg_potsdam",
    "cfg_vaihingen",
    "cfg_uavid",
    "cfg_udd5",
    "cfg_vdd"
)

$configs | ForEach-Object {
    $config = Join-Path $config_dir_base ($_ + ".py")
    $work_dir = Join-Path $work_dir_base $_
    $show_dir = Join-Path $show_dir_base $_
    python eval.py --config $config --work-dir $work_dir --save-seg-dir "$show_dir/segs" --save-heatmap-dir "$show_dir/heatmaps"
}