import subprocess
import sys

# Run the main_analysis.py script
result = subprocess.run([sys.executable, "main_analysis.py"], capture_output=False)
sys.exit(result.returncode)
</parameter>
</create_file>
