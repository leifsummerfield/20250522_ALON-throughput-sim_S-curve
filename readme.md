# ALON Throughput Simulation Toolkit

## Overview

This project provides a modular, extensible, CLI-driven simulation toolkit for LA-ICP-MS imaging stage throughput analysis. It supports:

* **Single line scan debugging:** Motion profile and phase breakdown for given stage configs
* **Parametric sweeps:** Analyze total experiment time as a function of stage velocity and acceleration, including 2D surface plots
* **Vendor trade study:** Compare all potential vendor stage options using real part numbers in a single stacked-bar phase chart

All experiment parameters are configured via JSON files in the `config/` directory. Results (charts and PDF reports) are auto-saved to the `results/` directory.

---

## Project Structure

```
20250522_ALON-throughput-sim_S-curve/
│
├── config/
│   └── tv3_baseline.json      # Example config file
├── data/
│   └── stage_data.csv         # Vendor/part number parameter table
├── results/
│   ├── parametric/
│   └── single_cycle/
├── scripts/
│   ├── main.py
│   ├── plotter.py
│   ├── analysis.py
│   ├── simulator.py
│   └── motion_profile.py
├── requirements.txt
```

---

## Getting Started

### 1. **Set up your environment**
# On Windows: venv\Scripts\activate
```bash
python -m venv venv
.\venv\Scripts\Activate 
pip install -r requirements.txt
```

---

### 2. **Edit or add a configuration JSON file**

Edit `config/tv3_baseline.json` or create a new config variant as needed.

---

### 3. **Run a simulation**

You can also simply run `python scripts/main.py` with no arguments to execute the `single_cycle` scenario using the default parameters defined at the top of `scripts/main.py`.

**Basic example (run all use-cases with baseline config):**

```bash
python scripts/main.py --base-dir "C:/Users/lsummerfield/Documents/ESL Data Analytics/20250522_ALON-throughput-sim_S-curve"

python scripts/main.py --base-dir "C:\Users\Leif\Documents\ESL_Analytics\20250522_ALON-throughput-sim_S-curve" --usecase single_cycle
```

**Run only a specific use-case:**

```bash
python scripts/main.py --base-dir "..." --usecase single_cycle
python scripts/main.py --base-dir "..." --usecase vendor_table
python scripts/main.py --base-dir "..." --usecase parametric
```

**Use a different config:**

```bash
python scripts/main.py --base-dir "..." --config tv3_highres.json
```

**Generate a PDF report:**

```bash
python scripts/main.py --base-dir "..." --pdf-report
```

**Combine options (e.g., run parametric sweep and generate report):**

```bash
python scripts/main.py --base-dir "..." --usecase parametric --pdf-report
```

**Change log level (DEBUG, INFO, WARNING, ERROR):**

```bash
python scripts/main.py --base-dir "..." --loglevel DEBUG
```

---

## Extending the Toolkit

* **Add new experiment configurations:**
  Add or duplicate a JSON config file in the `config/` directory.

* **Add a new use-case:**
  Write a new function in `main.py`, add it to the `USE_CASES` dictionary, and document its CLI name.

* **Add new vendors/part numbers:**
  Update `data/stage_data.csv` and rerun the vendor table analysis.

---

## Dependencies

All required packages are listed in `requirements.txt`.

**Main dependencies:**

* numpy
* pandas
* matplotlib
* fpdf

---

## Outputs

All results and charts are saved in the `results/` directory.

PDF reports are generated if `--pdf-report` is specified, named after the config.

---

## Authors and Maintenance

**Primary:**
Leif Summerfield

**Contributors:**


---
