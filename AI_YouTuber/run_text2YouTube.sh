#!/bin/bash

failed_indices=()

for i in {0..59}
do
    python AI_YouTuber/text2YouTube_HunYuan.py -i $i
    if [ $? -ne 0 ]; then
        echo "Error encountered with i=$i. Skipping..."
        failed_indices+=($i)
    fi
done
trap 'echo "Failed iterations so far: ${failed_indices[@]}"; exit' INT TERM
echo "Failed iterations: ${failed_indices[@]}"