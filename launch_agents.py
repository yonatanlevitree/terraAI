import os
import subprocess
import time
import sys

# Set your sweep info here
ENTITY = "yonatanlevitree-levitree"
PROJECT = "terraAI"
SWEEP_ID = "51srywy9"  # <-- Replace with your actual sweep id

NUM_AGENTS = os.cpu_count()  # Use all logical cores
VENV_PATH = os.path.join(os.getcwd(), "venv", "Scripts", "activate")  # Adjust if your venv is elsewhere

print(f"Launching {NUM_AGENTS} wandb agents...")

processes = []
try:
    for i in range(NUM_AGENTS):
        print(f"Starting agent {i+1}/{NUM_AGENTS}")
        # Use shell=True for Windows PowerShell compatibility
        # Use 'start' to launch in a new window (optional)
        command = f"& '{VENV_PATH}'; wandb agent {ENTITY}/{PROJECT}/{SWEEP_ID}"

        p = subprocess.Popen(
            ["powershell.exe", "-NoExit", "-Command", command],
            cwd=os.getcwd(),  # Explicitly set working directory
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        processes.append(p)
        time.sleep(1)  # Stagger launches slightly

    print("All agents launched. Press Ctrl+C to stop all agents.")
    # Wait for all processes to finish
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("\nStopping all agents...")
    for p in processes:
        p.terminate()
    print("All agents stopped.") 