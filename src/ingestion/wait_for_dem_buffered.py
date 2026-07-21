import ee, time
ee.Initialize(project="fyp-flood-mapping")
TARGETS = ["region_a_kelani_dem_buffered", "region_b_dakar_dem_buffered"]

while True:
    tasks = ee.batch.Task.list()
    states = {t.status()["description"]: t.status()["state"] for t in tasks if t.status()["description"] in TARGETS}
    print(f"\n--- {time.strftime('%H:%M:%S')} ---")
    for n in TARGETS:
        print(f"  {n:35s} {states.get(n, 'not found yet')}")
    if all(states.get(n) == "COMPLETED" for n in TARGETS):
        print("\nBoth buffered DEM exports completed.")
        break
    if any(states.get(n) == "FAILED" for n in TARGETS):
        print("\nA task failed.")
        break
    time.sleep(20)
