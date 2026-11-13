"""Compatibility entry point; pooled study code now lives in code.analyses.pooled_ws."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from NM_TinyRNN.code.analyses import pooled_ws

if __name__ == '__main__':
    pooled_ws.main()
else:
    sys.modules[__name__] = pooled_ws
