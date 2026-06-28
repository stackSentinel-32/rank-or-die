# 🏆 Redrob AI: Candidate Ranking Pipeline

A production-grade, highly performant candidate ranking pipeline designed to evaluate **100,000+ profiles** against a "Senior AI Engineer" job description. The system processes the candidate pool in **under 2.5 minutes** (on a 4-CPU environment), filtering out noise, detecting honeypot (fraudulent) resumes, and outputting the top 100 candidates with verified audit trails.

---

## 🏗️ Architecture & Core Modules

The pipeline uses a multi-stage approach, splitting candidate profiles into clean data structures, applying high-recall filters, generating independent signal scores, and fusing them together using **Reciprocal Rank Fusion (RRF)**.

```text
Redrob_AI/
├── main.py              # Main CLI entry point to run the pipeline
├── requirements.txt     # Python dependency configuration
├── Dockerfile           # Dockerized build for isolated sandbox execution
├── build_and_test.sh    # Script to build & execute container under sandboxed limits
├── verify_output.py     # Independent post-run assertion verification utility
├── data/                # Default inputs directory (data/candidates.jsonl)
└── ranker/              # Core evaluation engine
    ├── constants.py     # Skill taxonomies (Tiers 1-3), city lists, weights
    ├── parser.py        # Extracts raw profile lines into clean feature dicts
    ├── filters.py       # Hard discard filters and honeypot flag evaluation
    ├── scorer.py        # Independent Keyword (Sig A), TF-IDF (Sig B), Semantic (Sig C) Scorers
    ├── signals.py       # Computes availability multipliers and geo bonuses
    ├── fusion.py        # Fuses scores using RRF with a two-signal fallback
    └── output.py        # Handles dynamic format rendering and output CSV generation
```

---

## ⚡ Pipeline Execution Flow

The CLI processes candidates in **six distinct phases**:

```text
Phase 0: Load Model & Encode JD
   ↓
Phase 1: Stream & Discard Filters (Early Discard)
   ↓
Phase 2: Compute Signals A, B, C & RRF Fusion
   ↓
Phase 3: Apply Availability & Geo Bonus
   ↓
Phase 4: Select Top-100 with Honeypot Budget ≤10
   ↓
Phase 5: Validate Assertions & Write output.csv
```

### 1. Hard Filters (`ranker/filters.py`)
To optimize compute, candidates are streamed line-by-line and passed through four early discard filters:
*   **WITCH-only Career:** Discards candidates with >4 years of experience whose entire careers have been spent at IT services companies (WITCH) without significant ML contributions.
*   **Zero AI Signal:** Discards profiles that do not share any skills or description text in common with the target JD.
*   **Wrong Domain:** Discards profiles with mismatched roles (e.g., Accountant, HR Manager) unless they hold a technical degree or possess at least two core AI skills.
*   **Computer Vision/Speech Specialist:** Discards CV/Speech specialists if they lack NLP/IR backgrounds (the target role prioritizes search and retrieval).

### 2. Honeypot Detection (`ranker/parser.py`)
Identifies fake profiles by monitoring inconsistencies:
*   **Tenure Mismatch:** Discrepancy between calculated dates and claimed duration months > 6 months.
*   **Future Start Date:** Jobs starting in the future.
*   **YoE Sum Mismatch:** Claimed overall years of experience mismatching the sum of individual roles by > 3 years.
*   **Instant Expert:** "Advanced" skill proficiency declared with <= 1 month of experience.
*   **Education Timeline Inversion:** School end year preceding the start year.
*   **Salary Inversion:** Expected salary minimum exceeding the maximum.

Profiles matching **>= 2 flags** or a **salary inversion** are flagged as honeypots. The pipeline restricts the final output to a maximum of **10 honeypots** to avoid polluting the shortlists.

### 3. Fused Scoring & Normalization (`ranker/scorer.py` & `ranker/fusion.py`)
Survivors of Phase 1 are scored using three independent ranking signals:
*   **Signal A (Keyword Scoring):** Weighted matching based on tier relevancy, duration months, self-reported proficiency, skill assessments, ML role tenure, and a target YOE Gaussian curve peaking at 7 years.
*   **Signal B (TF-IDF Cosine Similarity):** N-gram TF-IDF matching on the combined skills and career description text against the JD text.
*   **Signal C (Semantic Similarity):** Deep semantic encoding using `TaylorAI/bge-micro-v2`. A keyword pre-filter gate (`keyword_score > 0.10`) prevents encoding zero-signal profiles, saving CPU cycles.
*   **Reciprocal Rank Fusion (RRF):** Fuses the rankings. If the semantic model fails to load, RRF gracefully falls back to a two-signal weighted RRF (Keyword: 0.65, TF-IDF: 0.35).
*   **Min-Max Normalization:** Rescales final scores in the output column to a full `[0.0, 1.0]` range (Rank 1 score = `1.000000`, Rank 100 score = `0.000000`).

---

## 🚀 How to Run

### Setup Environment
Using the pre-configured virtual environment is recommended:
```powershell
# Windows
.\venv\Scripts\activate

# Unix/macOS
source venv/bin/activate
```
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### Run the Pipeline
Run the main script using the command-line interface:
```bash
python main.py --input data/candidates.jsonl --output data/output.csv --verbose
```
*   `--input`: Path to input candidates file (defaults to `data/candidates.jsonl`).
*   `--output`: Path to save output rankings (defaults to `output.csv`).
*   `--verbose`: Enables step-by-step progress logging and metrics.

### Verify Output
Run the validation script to verify formatting consistency, score monotonicity, and honeypot counts:
```bash
python verify_output.py
```

---

## 🐳 Dockerized Build (Offline Environment)

To package and run the pipeline inside an isolated environment resembling the hackathon grading sandbox (no internet access, CPU-only execution, restricted resources):

1.  **Build and Download Model:**
    The Docker image downloads and caches the model during the image build phase.
    ```bash
    docker build -t redrob-ranker .
    ```

2.  **Run Under Constraints:**
    The container runs with the `--network none` flag (forcing zero network connections), limited to **14 GB of RAM** and **4 CPU cores**:
    ```bash
    docker run --network none --memory 14g --cpus 4 \
      -v "$(pwd)/data":/data \
      redrob-ranker \
      --input /data/candidates.jsonl \
      --output /data/output.csv \
      --verbose
    ```
    Alternatively, execution can be automated using:
    ```bash
    bash build_and_test.sh
    ```

---

## 📈 Performance Summary
On a 100k candidate pool:
*   **Early Filters:** Discards **~59.3%** of candidates instantly, bypassing heavy batch scoring.
*   **Execution Time:** **~150 seconds** (on CPU only).
*   **Memory Footprint:** Well below the **16 GB** limits.
*   **Honeypots Checked:** Exactly **10** honeypots allowed in the final top-100.
