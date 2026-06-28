# Rank or Die: Candidate Ranking System

This repository contains the candidate ranking pipeline for the Redrob AI Hackathon. The system evaluates 100,000 candidate profiles against a specific Senior AI Engineer Job Description using a multi-layered approach involving hard filters, rule-based keyword scoring, TF-IDF text matching, and semantic embedding evaluation.

## Project Structure

```text
Redrob_AI/
├── main.py              # Main entry point to run the pipeline
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── data/                # Data directory (place candidates.jsonl here)
├── ranker/              # Core ranking engine module
│   ├── __init__.py      
│   ├── constants.py     # Skill taxonomies, tiers, weights, and rules
│   ├── parser.py        # Raw JSON to feature dictionary extractor
│   ├── filters.py       # Hard discard filters and honeypot detection
│   ├── scorer.py        # Keyword and TF-IDF scoring engines
│   ├── signals.py       # Embeddings and deep semantic scoring
│   ├── fusion.py        # Score fusion and normalization logic
│   └── output.py        # Final CSV generation logic
└── venv/                # Python Virtual Environment
```

## How to Start (Windows)

1. **Activate the Virtual Environment:**
   Ensure you are using the provided virtual environment which contains all the required dependencies.
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies (if not already installed):**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Data Preparation:**
   Ensure the candidate data file (`candidates.jsonl`) is located in the root of the project or inside the `data/` directory (depending on how `main.py` is configured to read it).

4. **Run the Pipeline:**
   Execute the main entry point to start the filtering and ranking process.
   ```powershell
   python main.py
   ```

## Key Features

- **Honeypot Detection:** Identifies fake profiles using 7 distinct rules (e.g., tenure mismatch, inverted salary, impossible skill durations).
- **Stuffer Resistance:** Uses TF-IDF on career descriptions and Title Coherence checks to heavily penalize keyword stuffing.
- **Multi-Signal Fusion:** Combines rule-based scoring (Signal A), TF-IDF matching (Signal B), and planned deep semantic bi-encoder/cross-encoder matching for maximum precision on the top 100 candidates.
