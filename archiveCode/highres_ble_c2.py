#!/usr/bin/env python3
# Concept2 PM5 BLE logger (hi-res). Compatible with bleak 1.1.1.
# - Subscribes to C2 Rowing Service notifies: 0x0031/0x0032/0x0033 (+ 0x0035/0x0036 raw)
# - Sets sample rate to fastest (~100 ms) via 0x0034
# - Logs parsed metrics and raw packets with nanosecond timestamps

import asyncio, csv, datetime, time
from bleak import BleakScanner, BleakClient

# Concept2 Rowing Service UUID base
BASE = "ce06{:04x}-43e5-11e4-916c-0800200c9a66".format

# Notify characteristics
CHAR_GEN    = BASE(0x0031)  # General Status
CHAR_ADD1   = BASE(0x0032)  # Additional Status 1
CHAR_ADD2   = BASE(0x0033)  # Additional Status 2
CHAR_EXTRA1 = BASE(0x0035)  # Extra stream (stroke/summary; firmware-dependent)
CHAR_EXTRA2 = BASE(0x0036)  # Extra stream (stroke/summary; firmware-dependent)

# Write characteristic: sample rate control (0x00=1s, 0x01=500ms, 0x02=250ms, 0x03≈100ms)
CHAR_RATE   = BASE(0x0034)

# --- helpers -----------------------------------------------------------------

def u16le(b, i): return b[i] | (b[i+1] << 8)
def u24le(b, i): return b[i] | (b[i+1] << 8) | (b[i+2] << 16)

WORKOUT_STATE = {
    0:"WaitToBegin",1:"WorkoutRow",2:"CountdownPause",3:"IntervalRest",
    4:"WorkTime",5:"WorkDistance",6:"Rest",7:"WorkCalories",
    8:"WorkWatts",9:"Finished",10:"Terminated"
}
ROWING_STATE = {0:"Idle",1:"Active",2:"Inactive"}
STROKE_STATE = {0:"Unknown",1:"Ready",2:"Drive",3:"Dwell",4:"Recovery"}

def sec_to_mmss(x):
    if x is None: return None
    m = int(x // 60); s = x - 60*m
    return f"{m}:{s:04.1f}"

def safe_get(b, idx):  # returns None if out of range
    return b[idx] if idx < len(b) else None

# --- parsers for 0x0031/0x0032/0x0033 ---------------------------------------

def parse_gen(b: bytes):
    # 0x0031: 19 bytes typical; guard by length
    out = {}
    if len(b) >= 6:
        out["elapsed_s"]  = u24le(b,0)/100.0
        out["distance_m"] = u24le(b,3)/10.0
    if len(b) >= 11:
        wt, it, ws, rs, ss = safe_get(b,6), safe_get(b,7), safe_get(b,8), safe_get(b,9), safe_get(b,10)
        if wt is not None: out["workout_type"] = wt
        if it is not None: out["interval_type"] = it
        if ws is not None: out["workout_state"] = WORKOUT_STATE.get(ws, ws)
        if rs is not None: out["rowing_state"]  = ROWING_STATE.get(rs, rs)
        if ss is not None: out["stroke_state"]  = STROKE_STATE.get(ss, ss)
    if len(b) >= 14:
        out["total_work_distance_m"] = u24le(b,11)
    return out

def parse_add1(b: bytes):
    # 0x0032: 17 bytes typical; guard by length
    out = {}
    if len(b) >= 3:  out["elapsed_s"] = u24le(b,0)/100.0
    if len(b) >= 6:
        out["speed_m_s"] = u16le(b,3)/1000.0
        out["spm"] = safe_get(b,5)
    if len(b) >= 7:
        hr = safe_get(b,6)
        out["hr_bpm"] = None if hr is None or hr == 255 else hr
    if len(b) >= 9:  out["pace_cur_s_per_500m"] = u16le(b,7)/100.0
    if len(b) >= 11: out["pace_avg_s_per_500m"] = u16le(b,9)/100.0
    if len(b) >= 13: out["rest_distance_m"] = u16le(b,11)
    if len(b) >= 16: out["rest_time_s"] = u24le(b,13)/100.0
    if len(b) >= 17: out["erg_machine_type"] = b[16]
    return out

def parse_add2(b: bytes):
    # 0x0033: ~20 bytes typical; guard by length
    out = {}
    if len(b) >= 3:  out["elapsed_s"] = u24le(b,0)/100.0
    if len(b) >= 4:  out["interval_count"] = b[3]
    if len(b) >= 6:  out["avg_power_w"] = u16le(b,4)
    if len(b) >= 8:  out["total_calories"] = u16le(b,6)
    if len(b) >= 10: out["split_avg_pace_s_per_500m"] = u16le(b,8)/100.0
    if len(b) >= 12: out["split_avg_power_w"] = u16le(b,10)
    if len(b) >= 14: out["split_avg_cal_hr"] = u16le(b,12)
    if len(b) >= 17: out["last_split_time_s"] = u24le(b,14)/10.0
    if len(b) >= 20: out["last_split_distance_m"] = u24le(b,17)
    return out

# --- main --------------------------------------------------------------------

async def main():
    print("Scanning for PM5…")
    devs = await BleakScanner.discover(timeout=5.0)
    pm = next((d for d in devs if (d.name or "").startswith("PM5")), None)
    if not pm:
        raise SystemExit("No PM5 found. On the PM5, open Bluetooth/Connect and try again.")

    print(f"Connecting to {pm.name} @ {pm.address} … (don’t use sudo for BLE)")
    async with BleakClient(pm) as cli:
        # Try to set fastest notify rate (~10 Hz)
        try:
            await cli.write_gatt_char(CHAR_RATE, bytes([3]), response=True)
        except Exception:
            pass  # some firmware ignores this

        # Prepare CSVs
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        parsed_name = f"pm5_ble_parsed_{ts}.csv"
        raw_name    = f"pm5_ble_raw_{ts}.csv"

        parsed_fields = [
            "timestamp_iso","elapsed_s","distance_m","spm","hr_bpm","speed_m_s",
            "pace_cur_s_per_500m","pace_cur_mmss",
            "pace_avg_s_per_500m","pace_avg_mmss",
            "avg_power_w","total_calories",
            "workout_type","interval_type","workout_state","rowing_state","stroke_state",
            "interval_count","rest_distance_m","rest_time_s",
            "last_split_time_s","last_split_distance_m",
            "total_work_distance_m","erg_machine_type",
            "split_avg_pace_s_per_500m","split_avg_power_w","split_avg_cal_hr",
        ]
        pf = open(parsed_name, "w", newline="")
        pw = csv.DictWriter(pf, fieldnames=parsed_fields); pw.writeheader()

        raw_fields = ["ts_ns","char","hex"]
        rf = open(raw_name, "w", newline="")
        rw = csv.DictWriter(rf, fieldnames=raw_fields); rw.writeheader()

        # hi-res timebase for raw packets
        t0 = time.perf_counter_ns()
        def ts_ns(): return time.perf_counter_ns() - t0

        latest = {"gen": {}, "add1": {}, "add2": {}}

        def log_raw(tag: str, data: bytes):
            rw.writerow({"ts_ns": ts_ns(), "char": tag, "hex": data.hex()})
            rf.flush()

        def merge_row():
            row = {}
            for k in ("gen","add1","add2"):
                for kk,vv in latest[k].items():
                    if vv is not None:
                        row[kk] = vv
            # friendly pace strings
            if "pace_cur_s_per_500m" in row:
                row["pace_cur_mmss"] = sec_to_mmss(row["pace_cur_s_per_500m"])
            if "pace_avg_s_per_500m" in row:
                row["pace_avg_mmss"] = sec_to_mmss(row["pace_avg_s_per_500m"])
            return row

        # notify handlers
        def on_gen(_, data: bytearray):
            b = bytes(data); log_raw("0031", b); latest["gen"] = parse_gen(b)
        def on_add1(_, data: bytearray):
            b = bytes(data); log_raw("0032", b); latest["add1"] = parse_add1(b)
        def on_add2(_, data: bytearray):
            b = bytes(data); log_raw("0033", b); latest["add2"] = parse_add2(b)
        def on_extra1(_, data: bytearray):
            log_raw("0035", bytes(data))  # raw capture; parse later if desired
        def on_extra2(_, data: bytearray):
            log_raw("0036", bytes(data))  # raw capture; parse later if desired

        # subscribe
        await cli.start_notify(CHAR_GEN,    on_gen)
        await cli.start_notify(CHAR_ADD1,   on_add1)
        await cli.start_notify(CHAR_ADD2,   on_add2)
        await cli.start_notify(CHAR_EXTRA1, on_extra1)
        await cli.start_notify(CHAR_EXTRA2, on_extra2)

        print("Subscribed to 0x0031/0x0032/0x0033/0x0035/0x0036.")
        print(f"Logging parsed rows → {parsed_name}")
        print(f"Logging raw packets → {raw_name}")
        print("Start ‘Just Row’ and pull… (Ctrl+C to stop)")

        # UI loop (capture happens in callbacks)
        try:
            while True:
                row = merge_row()
                if row:
                    row["timestamp_iso"] = datetime.datetime.now().isoformat(timespec="seconds")
                    pw.writerow({k: row.get(k) for k in parsed_fields}); pf.flush()
                    # Pretty console line (avg power from ADD2; instantaneous power isn’t in these packets)
                    et   = f"{row.get('elapsed_s',0.0):6.1f}s"
                    dist = f"{row.get('distance_m',0.0):7.1f}m"
                    spm  = f"{row.get('spm','--'):>2}spm"
                    hr   = f"{row.get('hr_bpm','--'):>3}bpm"
                    pace = row.get("pace_cur_mmss","-:--.-")
                    pwr  = f"{row.get('avg_power_w','--'):>3}W"
                    print(f"{et} {dist}  {spm}  {hr}  {pace}  {pwr}")
                await asyncio.sleep(0.03)  # UI refresh only; capture is event-driven
        except KeyboardInterrupt:
            pass
        finally:
            # stop notifications & close files
            for ch in (CHAR_GEN, CHAR_ADD1, CHAR_ADD2, CHAR_EXTRA1, CHAR_EXTRA2):
                try: await cli.stop_notify(ch)
                except Exception: pass
            pf.close(); rf.close()
            print("Saved:")
            print("  ", parsed_name)
            print("  ", raw_name)

if __name__ == "__main__":
    asyncio.run(main())
