#!/bin/bash

# Check if input and output directories are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

input_dir="$1"
output_dir="$2"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Define the array of pipeline options
# pipelines=("img2img" "canny" "depth" "redux" "depth_img2img" "canny_img2img" "depth_redux" "canny_redux")
pipelines=("img2img" "depth_img2img" "canny_img2img")

# Set the base prompt
prompt="full of furniture, interior design, clean, tidy, ambient lighting, realistic, natural, high-quality"

# Loop through the pipeline options
for pipeline in "${pipelines[@]}"; do
    echo "Processing with pipeline: $pipeline"
    # Loop through all image files in the input directory
    for image in "$input_dir"/*.{jpg,jpeg,png,bmp,gif}; do
        # Check if file exists (this is necessary to handle the case when no files match the pattern)
        [ -e "$image" ] || continue

        # Get the filename without extension
        filename=$(basename -- "$image")
        filename="${filename%.*}"

        # Set the prompt with respect to room type
        current_prompt="$prompt"
        if [ "$filename" = "a0" ] || [ "$filename" = "a5" ]; then
            current_prompt="$prompt, bedroom"
        elif [ "$filename" = "a1" ] || [ "$filename" = "a3" ] || [ "$filename" = "a4" ]; then
            current_prompt="$prompt, kitchen"
        elif [ "$filename" = "a2" ]; then
            current_prompt="$prompt, living room"
        elif [ "$filename" = "a9" ]; then
            current_prompt="$prompt, dining room"
        elif [ "$filename" = "a6" ] || [ "$filename" = "a7" ] || [ "$filename" = "a8" ] || [ "$filename" = "b0" ]; then
            current_prompt="$prompt, empty room"
        fi

        # Process the image
        python /home/user/mount/datadisk/Michael/flux/examples/flux_unified.py "$pipeline" --input_image "$image" --prompt "$current_prompt" --output_dir "$output_dir"
    done
done

echo "All images processed."