$config = "configs/Vaihingen.py"
$work_dir = "work-dir/Vaihingen"
$show_dir = "show-dir/Vaihingen"

python eval.py --config $config --work-dir $work_dir --save-seg-dir "$show_dir/segs" --save-heatmap-dir "$show_dir/heatmaps"