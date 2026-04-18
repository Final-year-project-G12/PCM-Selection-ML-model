"""
MASTER PREPROCESSING RUNNER
===========================
Orchestrate all preprocessing steps in sequence.

Execution flow:
  Step 0 → Audit initial files
  Step 1 → Unit conversions + derived features
  Step 2 → Temporal engineering (cyclical, lags, rolling)
  Step 3 → Temporal alignment & lag handling
  Step 4 → Outlier detection & cleaning
  Step 5 → Spatial joins (regional extraction)
  Step 6 → Normalization & scaling
  Step 8 → Final output & audit

Run this script to execute the entire pipeline sequentially.
"""

import os
import sys
import subprocess
from datetime import datetime

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSING_DIR = os.path.join(BASE_DIR, 'processing')

PYTHON_EXE = sys.executable

# Define steps in execution order
STEPS = [
    ('0', 'step_0_audit.py', 'Audit & Validate'),
    ('1', 'step_1_conversions.py', 'Unit Conversions'),
    ('2', 'step_2_temporal_features.py', 'Temporal Features'),
    ('3', 'step_3_alignment.py', 'Temporal Alignment'),
    ('4', 'step_4_outliers.py', 'Outlier Detection'),
    ('5', 'step_5_spatial.py', 'Spatial Joins'),
    ('6', 'step_6_scaling.py', 'Normalization & Scaling'),
    ('8', 'step_8_final_output.py', 'Final Output'),
]

def print_header(text):
    """Print a formatted header."""
    width = 80
    print(f"\n{'='*width}")
    print(f"  {text.center(width-4)}")
    print(f"{'='*width}\n")

def run_step(step_num, script_name, description):
    """Execute a single preprocessing step."""
    script_path = os.path.join(PROCESSING_DIR, script_name)
    
    print_header(f"EXECUTING STEP {step_num}: {description}")
    print(f"Script: {script_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [PYTHON_EXE, script_path],
            cwd=PROCESSING_DIR,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n[OK] STEP {step_num} COMPLETE")
            return True
        else:
            print(f"\n[ERROR] STEP {step_num} FAILED (exit code {result.returncode})")
            return False
    
    except Exception as e:
        print(f"\n[ERROR] ERROR running step {step_num}: {e}")
        return False

def main():
    """Main orchestration function."""
    
    print_header("TAMIL NADU ERA5 PREPROCESSING PIPELINE")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing directory: {PROCESSING_DIR}\n")
    
    # Check if processing directory exists
    if not os.path.exists(PROCESSING_DIR):
        print(f"❌ ERROR: Processing directory not found: {PROCESSING_DIR}")
        sys.exit(1)
    
    # Summary of steps
    print("Pipeline structure:")
    for step_num, script, desc in STEPS:
        print(f"  Step {step_num}: {desc:<35} ({script})")
    
    print(f"\n{'─'*80}\n")
    
    # Execute steps
    completed_steps = []
    failed_step = None
    
    for step_num, script, description in STEPS:
        success = run_step(step_num, script, description)
        
        if success:
            completed_steps.append(step_num)
        else:
            failed_step = step_num
            print(f"\n[WARNING] Pipeline stopped at Step {step_num}")
            break
        
        print(f"\nCompleted: {len(completed_steps)}/{len(STEPS)}\n")
    
    # Final summary
    print_header("PIPELINE SUMMARY")
    
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nSteps completed: {len(completed_steps)}/{len(STEPS)}")
    for step_num, _, desc in STEPS:
        status = "[OK]" if step_num in completed_steps else "[--]"
        print(f"{status} Step {step_num}: {desc}")
    
    if failed_step:
        print(f"\n❌ FAILED at Step {failed_step}")
        print(f"Review the step's output above for error details.")
        sys.exit(1)
    else:
        print(f"\n[SUCCESS] PREPROCESSING PIPELINE COMPLETE!")
        print(f"\nOutput files:")
        
        data_dir = os.path.join(BASE_DIR, 'data')
        final_csv = os.path.join(data_dir, 'era5_tamilnadu_preprocessed_final.csv')
        if os.path.exists(final_csv):
            size_gb = os.path.getsize(final_csv) / 1e9
            print(f"  ✅ {final_csv} ({size_gb:.2f} GB)")
        
        scalers = [
            'scaler_minmax_tamilnadu.pkl',
            'scaler_standard_tamilnadu.pkl',
            'scaler_robust_tamilnadu.pkl'
        ]
        for scaler in scalers:
            scaler_path = os.path.join(data_dir, scaler)
            if os.path.exists(scaler_path):
                print(f"  ✅ {scaler}")
        
        print(f"\nNext steps:")
        print(f"  1. Load final CSV for XGBoost PCM classifier training")
        print(f"  2. Deploy scalers (PKL files) to RPi/TFLite pipeline")
        print(f"  3. Use regional climate CSVs for prototype simulation setup")
        
        sys.exit(0)

if __name__ == '__main__':
    main()
