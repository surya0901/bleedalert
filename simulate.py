"""Simulate a day of head CTs arriving at one remote reader.

Compares two queues:
  FIFO     : scans are read in the order they arrive (today)
  TRIAGE   : scans BleedAlert flags as urgent are read first

All inputs are adjustable assumptions, NOT measured hospital data. Say so in the pitch.
Run standalone:  python simulate.py
"""
import heapq
import random
import statistics


def simulate(scans_per_day=60, bleed_rate=0.10, read_minutes=8.0,
             sensitivity=0.90, specificity=0.85, days=200, seed=0):
    rng = random.Random(seed)
    fifo_waits, triage_waits = [], []

    for _ in range(days):
        # Arrivals spread randomly over a 24 hour day (minutes)
        arrivals = sorted(rng.uniform(0, 1440) for _ in range(scans_per_day))
        scans = []
        for t in arrivals:
            bleed = rng.random() < bleed_rate
            flagged = rng.random() < (sensitivity if bleed else 1 - specificity)
            scans.append((t, bleed, flagged))

        for mode, out in (("fifo", fifo_waits), ("triage", triage_waits)):
            clock, i, queue = 0.0, 0, []
            while i < len(scans) or queue:
                if not queue and i < len(scans) and scans[i][0] > clock:
                    clock = scans[i][0]
                while i < len(scans) and scans[i][0] <= clock:
                    t, bleed, flagged = scans[i]
                    priority = 0 if (mode == "triage" and flagged) else 1
                    heapq.heappush(queue, (priority, t, bleed))
                    i += 1
                _, t, bleed = heapq.heappop(queue)
                if bleed:
                    out.append(clock - t)
                clock += rng.expovariate(1 / read_minutes)

    fifo_avg = statistics.mean(fifo_waits)
    triage_avg = statistics.mean(triage_waits)
    return {
        "assumptions": {
            "scans_per_day": scans_per_day, "bleed_rate": bleed_rate,
            "read_minutes": read_minutes, "sensitivity": sensitivity,
            "specificity": specificity, "days_simulated": days,
        },
        "fifo_avg_wait_min": round(fifo_avg, 1),
        "triage_avg_wait_min": round(triage_avg, 1),
        "minutes_saved": round(fifo_avg - triage_avg, 1),
        "fifo_p90_wait_min": round(sorted(fifo_waits)[int(0.9 * len(fifo_waits))], 1),
        "triage_p90_wait_min": round(sorted(triage_waits)[int(0.9 * len(triage_waits))], 1),
    }


if __name__ == "__main__":
    for load in (40, 80, 120, 160):
        r = simulate(scans_per_day=load)
        print(f"{load:>3} scans/day  FIFO {r['fifo_avg_wait_min']:>6} min  "
              f"BleedAlert {r['triage_avg_wait_min']:>5} min  saved {r['minutes_saved']} min")
