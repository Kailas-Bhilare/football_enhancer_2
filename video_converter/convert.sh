#!/bin/bash

# Set colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TARGET_FPS=30
AUDIO_BITRATE="128k"

# Create converted directory if it doesn't exist
mkdir -p clips/converted

echo -e "${GREEN}=== Video Converter ===${NC}"
echo -e "${BLUE}Target Format: H.264 MP4, ${TARGET_FPS} fps${NC}"
echo ""

# Check if there are videos to convert
if [ ! "$(ls -A clips/original/ 2>/dev/null)" ]; then
    echo -e "${RED}No videos found in clips/original/${NC}"
    exit 1
fi

# Show available videos
echo -e "${YELLOW}Available videos in original/:${NC}"
ls -1 clips/original/ | nl -w2 -s') '

echo ""
read -p "Enter video number to convert (or 'all' for all videos): " selection

# Function to convert a single video
convert_video() {
    input_file="$1"
    filename=$(basename -- "$input_file")
    name_without_ext="${filename%.*}"
    
    echo -e "${GREEN}Converting: ${filename}${NC}"
    
    # Ask for resolution
    echo -e "${YELLOW}Choose output resolution:${NC}"
    echo "1) 480p (854x480) - Smaller file"
    echo "2) 720p (1280x720) - Better quality"
    read -p "Enter choice (1 or 2): " res_choice
    
    case $res_choice in
        1)
            resolution="854:480"
            res_name="480p"
            # Bitrate range for 480p: 1-1.5 Mbps
            min_bitrate="1000k"
            max_bitrate="1500k"
            ;;
        2)
            resolution="1280:720"
            res_name="720p"
            # Bitrate range for 720p: 1.5-2 Mbps
            min_bitrate="1500k"
            max_bitrate="2000k"
            ;;
        *)
            echo -e "${RED}Invalid choice. Using 480p.${NC}"
            resolution="854:480"
            res_name="480p"
            min_bitrate="1000k"
            max_bitrate="1500k"
            ;;
    esac
    
    # Ask for bitrate preference
    echo -e "${YELLOW}Choose bitrate:${NC}"
    echo "1) Lower (${min_bitrate}) - Smaller file"
    echo "2) Medium (${max_bitrate}) - Better quality"
    echo "3) Custom"
    read -p "Enter choice (1-3): " bitrate_choice
    
    case $bitrate_choice in
        1) video_bitrate="$min_bitrate" ;;
        2) video_bitrate="$max_bitrate" ;;
        3) 
            read -p "Enter custom bitrate (e.g., 1200k): " video_bitrate
            if [ -z "$video_bitrate" ]; then
                video_bitrate="$min_bitrate"
            fi
            ;;
        *)
            video_bitrate="$min_bitrate"
            ;;
    esac
    
    # Ask for duration
    echo -e "${YELLOW}Select clip duration:${NC}"
    echo "1) 1 minute"
    echo "2) 1.5 minutes (90 seconds)"
    echo "3) 2 minutes"
    echo "4) Full video (no trimming)"
    read -p "Enter choice (1-4): " duration_choice
    
    case $duration_choice in
        1) duration="60" ;;
        2) duration="90" ;;
        3) duration="120" ;;
        4) duration="0" ;;
        *) duration="90" ;;
    esac
    
    # Ask for start time if trimming
    start_time="0"
    if [ "$duration" != "0" ]; then
        echo -e "${YELLOW}Enter start time (format: HH:MM:SS or seconds):${NC}"
        read -p "Start time [0]: " start_time
        if [ -z "$start_time" ]; then
            start_time="0"
        fi
    fi
    
    # Output filename
    if [ "$duration" = "0" ]; then
        output_file="clips/converted/${name_without_ext}_${res_name}_${video_bitrate}.mp4"
    else
        output_file="clips/converted/${name_without_ext}_${res_name}_${duration}s_${video_bitrate}.mp4"
    fi
    
    echo -e "${BLUE}Converting with settings:${NC}"
    echo "  Resolution: ${resolution}"
    echo "  Video Bitrate: ${video_bitrate}"
    echo "  Audio Bitrate: ${AUDIO_BITRATE}"
    echo "  FPS: ${TARGET_FPS}"
    if [ "$duration" != "0" ]; then
        echo "  Duration: ${duration}s (from ${start_time})"
    else
        echo "  Duration: Full video"
    fi
    
    # Build ffmpeg command
    ffmpeg_cmd="ffmpeg -y"
    
    # Add start time if trimming
    if [ "$duration" != "0" ] && [ "$start_time" != "0" ]; then
        ffmpeg_cmd="$ffmpeg_cmd -ss $start_time"
    fi
    
    ffmpeg_cmd="$ffmpeg_cmd -i \"$input_file\""
    
    # Add duration if trimming
    if [ "$duration" != "0" ]; then
        ffmpeg_cmd="$ffmpeg_cmd -t $duration"
    fi
    
    # Add video filters and encoding options
    ffmpeg_cmd="$ffmpeg_cmd -vf \"scale=$resolution,fps=$TARGET_FPS\""
    ffmpeg_cmd="$ffmpeg_cmd -c:v libx264 -b:v $video_bitrate -preset medium"
    ffmpeg_cmd="$ffmpeg_cmd -c:a aac -b:a $AUDIO_BITRATE"
    ffmpeg_cmd="$ffmpeg_cmd -pix_fmt yuv420p -movflags +faststart"
    ffmpeg_cmd="$ffmpeg_cmd \"$output_file\""
    
    # Execute conversion
    echo -e "${BLUE}Running:${NC} $ffmpeg_cmd"
    eval $ffmpeg_cmd
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully converted: $(basename "$output_file")${NC}"
        
        # Show file size
        if [ -f "$output_file" ]; then
            filesize=$(du -h "$output_file" | cut -f1)
            echo -e "${BLUE}File size: ${filesize}${NC}"
        fi
    else
        echo -e "${RED}✗ Conversion failed for ${filename}${NC}"
    fi
    
    echo "----------------------------------------"
}

# Convert selected videos
if [ "$selection" = "all" ]; then
    for video in clips/original/*.mp4; do
        [ -f "$video" ] && convert_video "$video"
    done
else
    # Convert single video
    video_file=$(ls -1 clips/original/ | sed -n "${selection}p")
    if [ -n "$video_file" ]; then
        convert_video "clips/original/$video_file"
    else
        echo -e "${RED}Invalid selection${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}All conversions complete!${NC}"
echo -e "${YELLOW}Converted files are in: clips/converted/${NC}"
