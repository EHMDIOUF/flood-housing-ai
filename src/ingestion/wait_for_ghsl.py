import ee, time
ee.Initialize(project="fyp-flood-mapping")
REGIONS = ["region_a_kelani", "region_b_dakar"]
EPOCHS = [2000, 2010, 2015, 2020, 2025]
TARGETS = [f"{r}_ghsl_{e}" for r in REGIONS for e in EPOCHS]

while True:
    tasks = ee.batch.Task.list()
    states = {t.status()["description"]: t.status()["state"] for t in tasks if t.status()["description"] in TARGETS}
    done = sum(1 for v in states.values() if v == "COMPLETED")
    failed = [k for k, v in states.items() if v == "FAILED"]
    print(f"\n--- {time.strftime('%H:%M:%S')} --- {done}/{len(TARGETS)} completed, {len(failed)} failed")
    if failed:
        print("FAILED:", failed)
        break
    if done == len(TARGETS):
        print("All GHSL exports completed.")
        break
    time.sleep(20)
