#!/bin/bash

failed_indices=()

for i in {17..59}
do
    python examples/text2YouTube.py -i $i
    if [ $? -ne 0 ]; then
        echo "Error encountered with i=$i. Skipping..."
        failed_indices+=($i)
    fi
done

echo "Failed iterations: ${failed_indices[@]}"