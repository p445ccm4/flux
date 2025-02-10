#!/bin/bash
source activate base
conda activate flux

Topics=("Zodiac_Taurus" "Zodiac_Gemini" "Zodiac_Cancer" "Zodiac_Leo" "Zodiac_Virgo" "Zodiac_Libra" "Zodiac_Scorpio" "Zodiac_Sagittarius" "Zodiac_Capricorn" "Zodiac_Aquarius" "Zodiac_Pisces" "deep_sea" "black_holes")

for topic in "${Topics[@]}"; do
    json_file="inputs/AI_YouTuber/scripts/${topic}.json"
    working_dir="outputs/HunYuan/${topic}_20250210"
    mkdir -p "$working_dir"
    if [[ $topic == Zodiac* ]]; then
        music_path="inputs/AI_YouTuber/music/Zodiac.m4a"
    else
        music_path="inputs/AI_YouTuber/music/${topic}.m4a"
    fi
    log_file="${working_dir}/${topic}.log"

    if [ ! -f "$log_file" ]; then
    touch "$log_file"
    fi

    bash AI_YouTuber/run_text2YouTube.sh --json_file "$json_file" --working_dir "$working_dir" --music_path "$music_path" &> "$log_file"
done
