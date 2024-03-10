import subprocess
import time
from datetime import datetime

if __name__=="__main__":
    for _ in range(100):
        for gov in ["bmfac_gov", "bmfac2_gov", "ippo_gov", "ippo2_gov", 
                    "maddpg_gov", "maddpg2_gov", "maddpgb_gov", "maddpgb2_gov"]:
            cmd = f"python3 main.py --gov {gov}"
            subprocess.run(["python3", "/root/Competition_TaxingAI/main.py"])