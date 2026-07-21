import ee
import time

EE_PROJECT = "fyp-flood-mapping"
ee.Initialize(project=EE_PROJECT)

TARGETS = ["region_a_kelani_open_buildings_v2", "region_b_dakar_open_buildings_v2"]
POLL_SECONDS = 30

while True:
    tasks = ee.batch.Task.list()
    states = {}
    for t in tasks:
        s = t.status()
        if s["description"] in TARGETS:
            states[s["description"]] = s["state"]

    print(f"\n--- {time.strftime('%H:%M:%S')} ---")
    for name in TARGETS:
        print(f"  {name:40s} {states.get(name, 'not found yet')}")

    if all(states.get(n) == "COMPLETED" for n in TARGETS):
        print("\nBoth building-footprint exports completed.")
        break

    if any(states.get(n) == "FAILED" for n in TARGETS):
        print("\nAt least one v2 export failed -- stopping.")
        for n in TARGETS:
            if states.get(n) == "FAILED":
                for t in tasks:
                    s = t.status()
                    if s["description"] == n:
                        print(f"  {n} error: {s.get('error_message')}")
        break

    time.sleep(POLL_SECONDS)
