$config = "configs/LoveDA.py"
$work_dir = "work-dir/LoveDA"
$show_dir = "show-dir/LoveDA"

python eval.py --config $config --work-dir $work_dir --save-seg-dir "$show_dir/segs" --save-heatmap-dir "$show_dir/heatmaps"