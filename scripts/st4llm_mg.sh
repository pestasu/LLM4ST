num_process=4
master_port=9091

accelerate launch --multi_gpu --dynamo_backend inductor --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port main_mg.py \
--data "AIR-SZ" \
--model "st4llm" \
--is_training 1 \
--version 2
