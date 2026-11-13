"""Compatibility entry point for the existing session-learning-curve prototype.

The planned data_scaling study is separate and is not implemented by this script.
"""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from NM_TinyRNN.code.analyses import session_learning_curve

if __name__=='__main__':
    session_learning_curve.main()
else:
    sys.modules[__name__] = session_learning_curve
