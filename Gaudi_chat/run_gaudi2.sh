python -m fastchat.serve.cli --model-path databricks/dolly-v2-12b --device hpu  --use_cache --temperature 0.7 --use_graphs --static_shapes  --output_tps --seed 12345 --conv-template dolly

