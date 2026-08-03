# Intention-Guided Reward Shaping for Sample-Efficient Reinforcement Learning

## 📌 Overview
This project investigates whether a **single directional intention signal** can accelerate learning in sparse-reward environments.

## 🔬 Key Results
- **Intention reaches 50% success 5× faster** than sparse baseline (10k vs 50k steps)
- **Final reward: 109.84 vs 0.84** (130× improvement)

## 🏗️ Project Structure
- `train.py` - Main training script
- `analyze_results.py` - Result analysis & plotting
- `run_experiments.py` - Batch experiment runner
- `wrappers/` - Intention and dense reward wrappers

## 🚀 How to Run
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
python train.py --wrapper intention --seed 42 --debug