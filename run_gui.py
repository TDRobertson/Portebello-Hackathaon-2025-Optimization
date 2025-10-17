#!/usr/bin/env python3
"""
Launcher script for the Portobello Optimization GUI
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pyqt_integration import main
    
    if __name__ == '__main__':
        print("Starting Portobello America Optimization Suite...")
        print("Make sure you have PyQt5 installed: pip install PyQt5")
        main()
        
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nPlease install the required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting application: {e}")
    sys.exit(1)
