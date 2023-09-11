# python main.py -ld logs_dir/5_20 -fd --num_layers 5 --layer_size 20
# python main.py -ld logs_dir/5_30 -fd --num_layers 5 --layer_size 30
# python main.py -ld logs_dir/10_28 -fd --num_layers 10 --layer_size 28
# python main.py -ld logs_dir/10_40 -fd --num_layers 10 --layer_size 40
# python main.py -ld logs_dir/13_49 -fd --num_layers 13 --layer_size 49

for image_id in {1..24}
do
    for config_id in "5_20" "5_30" "10_28" "10_40" "13_49"
    do
        python3 test.py \
            $image_id \
            $config_id \
            results/fp/$config_id/$image_id/ \
            -fp
    done
done