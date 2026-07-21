import ee
import time

EE_PROJECT = "fyp-flood-mapping"
ee.Initialize(project=EE_PROJECT)

POLL_SECONDS = 30

while True:
    tasks = ee.batch.Task.list()
    relevant = [
        t for t in tasks
        if "region_a" in t.config.get("description", "") or "region_b" in t.config.get("description", "")
    ]

    states = {}
    for t in relevant:
        s = t.status()
        states[s["description"]] = s["state"]

    done = sum(1 for v in states.values() if v == "COMPLETED")
    failed = [k for k, v in states.items() if v == "FAILED"]
    total = len(states)

    print(f"\n--- {time.strftime('%H:%M:%S')} --- {done}/{total} completed, {len(failed)} failed")
    for name, state in sorted(states.items()):
        print(f"  {name:40s} {state}")

    if failed:
        print("\nFAILED TASKS DETECTED:")
        for name in failed:
            print(f"  - {name}")
        break

    if done == total:
        print("\nAll tasks completed successfully.")
        break

    time.sleep(POLL_SECONDS)
