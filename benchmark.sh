#!/bin/bash
# Run as root/sudo

# Set GPU to persistent mode
sudo nvidia-smi -pm 1

# Wait for settings to apply
sleep 2

# Lock clocks to prevent frequency scaling
sudo nvidia-smi -lgc 1000,1000

# Memory clock
sudo nvidia-smi -lmc 5000,5000

sudo nvidia-smi --auto-boost-default=0  # Disable auto-boost

# Run your benchmark
echo "GPU ready for benchmarking with consistent power state"