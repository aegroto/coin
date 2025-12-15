INPUT_IMAGE=$1
OUTPUT_FOLDER=$2

rm -rf $OUTPUT_FOLDER
mkdir -p $OUTPUT_FOLDER

uv run -m main \
    --num_layers $NUM_LAYERS --layer_size $LAYER_SIZE -ni 50000 \
    -ip $INPUT_IMAGE \
    -op $OUTPUT_FOLDER > $OUTPUT_FOLDER/execution.log 2>&1

uv run -m export_stats \
    $INPUT_IMAGE \
    $OUTPUT_FOLDER/fp_reconstruction.png \
    $OUTPUT_FOLDER/results.json \
    $OUTPUT_FOLDER/stats.json 
