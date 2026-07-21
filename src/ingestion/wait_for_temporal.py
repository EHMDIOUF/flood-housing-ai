import ee, time
ee.Initialize(project="fyp-flood-mapping")
REGIONS = ["region_a_kelani", "region_b_dakar"]
YEARS = [2019, 2021, 2023, 2025]
TARGETS = [f"{r}_s2_{y}" for r in REGIONS for y in YEARS]

while True:
    tasks = ee.batch.Task.list()
    states = {t.status()["description"]: t.status()["state"] for t in tasks if t.status()["description"] in TARGETS}
    done = sum(1 for v in states.values() if v == "COMPLETED")
    failed = [k for k, v in states.items() if v == "FAILED"]
    print(f"\n--- {time.strftime('%H:%M:%S')} --- {done}/{len(TARGETS)} completed, {len(failed)} failed")
    for n in sorted(TARGETS):
        print(f"  {n:30s} {states.get(n, 'not found yet')}")
    if failed:
        print("\nFAILED TASKS:", failed)
        break
    if done == len(TARGETS):
        print("\nAll temporal exports completed.")
        break
    time.sleep(30)
