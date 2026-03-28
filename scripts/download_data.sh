#!/bin/bash
# Download datasets from the Autoformer/ETDataset collection
set -e

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"/{ETT-small,weather,electricity,traffic}

echo "Downloading ETT datasets..."
for f in ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv; do
    wget -q -nc -O "$DATA_DIR/ETT-small/$f" \
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/$f" || true
    echo "  $f done"
done

echo "Downloading Weather dataset..."
wget -q -nc -O "$DATA_DIR/weather/weather.csv" \
    "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/weather.csv" 2>/dev/null || \
wget -q -nc -O "$DATA_DIR/weather/weather.csv" \
    "https://raw.githubusercontent.com/cure-lab/LTSF-Linear/main/dataset/weather.csv" 2>/dev/null || \
echo "  Weather: manual download may be needed"

echo "Downloading Electricity dataset..."
wget -q -nc -O "$DATA_DIR/electricity/electricity.csv" \
    "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/electricity.csv" 2>/dev/null || \
wget -q -nc -O "$DATA_DIR/electricity/electricity.csv" \
    "https://raw.githubusercontent.com/cure-lab/LTSF-Linear/main/dataset/electricity.csv" 2>/dev/null || \
echo "  Electricity: manual download may be needed"

echo "Downloading Traffic dataset..."
wget -q -nc -O "$DATA_DIR/traffic/traffic.csv" \
    "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/traffic.csv" 2>/dev/null || \
wget -q -nc -O "$DATA_DIR/traffic/traffic.csv" \
    "https://raw.githubusercontent.com/cure-lab/LTSF-Linear/main/dataset/traffic.csv" 2>/dev/null || \
echo "  Traffic: manual download may be needed"

echo ""
echo "Checking downloaded files:"
for f in "$DATA_DIR"/ETT-small/*.csv "$DATA_DIR"/weather/*.csv "$DATA_DIR"/electricity/*.csv "$DATA_DIR"/traffic/*.csv; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "  $f: $lines lines"
    else
        echo "  MISSING: $f"
    fi
done
