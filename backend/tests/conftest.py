import os
import sys
# Add backend root to sys.path so `app` can be imported in tests
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
