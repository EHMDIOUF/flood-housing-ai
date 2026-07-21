import ee, time
ee.Initialize(project="fyp-flood-mapping")
TARGETS = ["region_a_kelani_s2_full", "region_b_dakar_s2_full"]

while True:
    tasks = ee.batch.Task.list()
    states = {t.status()["description"]: t.status()["state"] for t in tasks if t.status()["description"] in TARGETS}
    print(f"\n--- {time.strftime('%H:%M:%S')} ---")
    for n in TARGETS:
        print(f"  {n:30s} {states.get(n, 'not found yet')}")
    if all(states.get(n) == "COMPLETED" for n in TARGETS):
        print("\nBoth S2 full-band exports completed.")
        break
    if any(states.get(n) == "FAILED" for n in TARGETS):
        print("\nA task failed.")
        break
    time.sleep(30)
