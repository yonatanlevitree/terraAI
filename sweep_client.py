import wandb
import requests
import time
import sys

API_URL = "http://192.168.4.91:5001/simulation"  # Update if your backend runs elsewhere
POLL_INTERVAL = 1  # seconds


def main():
    try:
        # 1. Initialize wandb and get sweep config
        run = wandb.init(project="terraAI")
        if not run:
            print("Failed to initialize wandb run")
            sys.exit(1)
            
        config = dict(wandb.config)
        
        # 2. Submit job to backend
        try:
            response = requests.post(API_URL, json=config)
            response.raise_for_status()
            job_id = response.json()["job_id"]
        except Exception as e:
            wandb.alert(title="Job submission failed", text=str(e))
            run.finish()
            return

        # 3. Poll for job completion
        while True:
            try:
                status_resp = requests.get(f"{API_URL}/{job_id}")
                status_resp.raise_for_status()
                job = status_resp.json()
            except Exception as e:
                wandb.alert(title="Job polling failed", text=str(e))
                run.finish()
                return
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(POLL_INTERVAL)

        # 4. Log results to wandb
        if job["status"] == "completed":
            summary = job["result"]["terrain_summary"]
            print("SUMMARY: ", summary)
            wandb.log({
                "finalMSE": summary.get("finalMSE", float("inf")),
                "finalFidelity": summary.get("finalFidelity", 0),
                "totalWells": summary.get("totalWells", 0),
                "monetaryCost": summary.get("monetaryCost", 0),
                "timeCost": summary.get("timeCost", 0),
            })
        else:
            wandb.log({"finalMSE": float("inf")})
            wandb.alert(title="Job failed", text=job.get("error", "Unknown error"))

        run.finish()
    except Exception as e:
        print(f"Error in wandb initialization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
