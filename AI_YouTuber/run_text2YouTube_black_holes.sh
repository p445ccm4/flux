#!/bin/bash

json_file="inputs/AI_YouTuber/scripts/black_holes.json"
working_dir="outputs/HunYuan/black_holes_20250207"

mkdir -p "$working_dir"
failed_indices=()
jq -c '.[]' $json_file | while read element; do
    # Check if the element starts with '{', if not, add it
    if [[ ! "$element" =~ ^\{.* ]] ; then
        element="{${element}"
    fi
    index=$(jq -r '.index' <<< "$element")
    caption=$(jq -r '.caption' <<< "$element")
    prompt=$(jq -r '.prompt' <<< "$element")

    # # Skip if index is not 0, 3, or 5
    # if [[ ! "$index" =~ ^(0|6|16|18)$ ]]; then
    #     echo "Skipping index $index"
    #     continue
    # fi

    echo "Element: $element"
    echo "Index: $index"
    echo "Cantonese Caption: $caption"
    echo "English Prompt: $prompt"

    python AI_YouTuber/gen_audio.py -i $index -c "$caption" -o $working_dir
    python AI_YouTuber/gen_video.py -i $index -e "$prompt" -o $working_dir
    python AI_YouTuber/interpolate.py --input_video_path $working_dir/$index.mp4 --output_video_path $working_dir/${index}_interpolated.mp4
    python AI_YouTuber/audio_caption.py --audio_path $working_dir/$index.mp3 --caption "$caption" --input_video_path $working_dir/${index}_interpolated.mp4 --output_video_path $working_dir/${index}_captioned.mp4
    
    if [ $? -ne 0 ]; then
        echo "Error encountered with index=$index. Skipping..."
        failed_indices+=($index)
    else
        echo "Successfully processed iteration with index=$index"
    fi
done

trap 'echo "Failed iterations so far: ${failed_indices[@]}"; exit' INT TERM

if [ ${#failed_indices[@]} -eq 0 ]; then
    echo "All iterations completed successfully!"
    python AI_YouTuber/concat.py -o $working_dir
    python AI_YouTuber/bg_music.py --input_video_path $working_dir/concat.mp4 --music_path inputs/AI_YouTuber/music/black_holes.m4a --output_video_path $working_dir/final.mp4
    echo "concat.py completed successfully!"
else
    echo "Some iterations failed. concat.py will not be run."
    echo "Failed iterations: ${failed_indices[@]}"
fi