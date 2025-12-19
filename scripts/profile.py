#!/usr/bin/env python3
"""Profile training with Nsight (run from project root). Example:
  nsys profile --stats=true python scripts/profile.py --epochs 5
  ncu --set full python scripts/profile.py --epochs 1
"""
import runpy
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
runpy.run_path("scripts/train_mixed_precision.py", run_name="__main__")
