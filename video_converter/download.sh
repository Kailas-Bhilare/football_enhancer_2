#!/bin/bash

# Set colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create directories if they don't exist
mkdir -p clips/original
mkdir -p clips/converted

echo -e "${GREEN}=== YouTube Video Downloader ===${NC}"

# Ask for video URL
read -p "Enter YouTube URL: " video_url

if [ -z "$video_url" ]; then
    echo -e "${RED}Error: No URL provided${NC}"
    exit 1
fi

# Ask for preferred quality
echo -e "${YELLOW}Choose quality:${NC}"
echo "1) 480p (854x480)"
echo "2) 720p (1280x720)"
read -p "Enter choice (1 or 2): " quality_choice

case $quality_choice in
    1)
        quality="480"
        resolution="854x480"
        format="best[height<=480]"
        ;;
    2)
        quality="720"
        resolution="1280x720"
        format="best[height<=720]"
        ;;
    *)
        echo -e "${RED}Invalid choice. Using 480p by default.${NC}"
        quality="480"
        resolution="854x480"
        format="best[height<=480]"
        ;;
esac

echo -e "${GREEN}Downloading video in ${resolution}...${NC}"

# Download the video
cd clips/original
yt-dlp -f "${format}+bestaudio" \
       --merge-output-format mp4 \
       --output "%(title)s.%(ext)s" \
       --restrict-filenames \
       "$video_url"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Download completed successfully!${NC}"
    echo -e "${YELLOW}Video saved in: clips/original/${NC}"
    
    # Show downloaded file
    latest_file=$(ls -t | head -n1)
    echo -e "${GREEN}Downloaded: ${latest_file}${NC}"
else
    echo -e "${RED}✗ Download failed${NC}"
    exit 1
fi

cd ../..
