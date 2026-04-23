#!/bin/bash
echo "=============================================="
echo "Starting Intent Classification Fine-tuning"
echo "=============================================="
python scripts/train.py --config configs/train.yaml
