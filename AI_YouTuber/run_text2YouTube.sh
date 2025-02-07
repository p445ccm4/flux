#!/bin/bash

json_file="inputs/AI_YouTuber_scripts/Jupiter.json"
working_dir="outputs/HunYuan/20250207"

failed_indices=()
jq -c '.[]' $json_file | while read element; do
    # Check if the element starts with '{', if not, add it
    if [[ ! "$element" =~ ^\{.* ]] ; then
        element="{${element}"
    fi
    index=$(jq -r '.index' <<< "$element")
    cantonese_caption=$(jq -r '.cantonese_caption' <<< "$element")
    english_prompt=$(jq -r '.english_prompt' <<< "$element")

    echo "Element: $element"
    echo "Index: $index"
    echo "Cantonese Caption: $cantonese_caption"
    echo "English Prompt: $english_prompt"

    python AI_YouTuber/gen_audio.py -i $index -c "$cantonese_caption" -o $working_dir
    # python AI_YouTuber/gen_video.py -i $index -e "$english_prompt" -o $working_dir
    python AI_YouTuber/interpolate.py --input_video_path $working_dir/$index.mp4 --output_video_path $working_dir/${index}_interpolated.mp4
    python AI_YouTuber/audio_caption.py --audio_path $working_dir/$index.mp3 --caption "$cantonese_caption" --input_video_path $working_dir/${index}_interpolated.mp4 --output_video_path $working_dir/${index}_captioned.mp4

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
    echo "concat.py completed successfully!"
else
    echo "Some iterations failed. concat.py will not be run."
    echo "Failed iterations: ${failed_indices[@]}"
fi