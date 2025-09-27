#!/usr/bin/env python3
# Robust Concept2 PM5 BLE logger (bleak 1.1.1 compatible, no get_services)
import asyncio, csv, datetime
from bleak import BleakScanner, BleakClient

BASE = "ce06{:04x}-43e5-11e4-916c-0800200c9a66".format
CHAR_GEN   = BASE(0x0031)  # notify: General Status
CHAR_ADD1  = BASE(0x0032)  # notify: Additional Status 1
CHAR_ADD2  = BASE(0x0033)  # notify: Additional Status 2
CHAR_RATE  = BASE(0x0034)  # write: sample rate (0..3)

def u16le(b, i): return b[i] | (b[i+1] << 8)
def u24le(b, i): return b[i] | (b[i+1] << 8) | (b[i+2] << 16)

WORKOUT_STATE = {0:"WaitToBegin",1:"WorkoutRow",2:"CountdownPause",3:"IntervalRest",
                 4:"WorkTime",5:"WorkDistance",6:"Rest",7:"WorkCalories",
                 8:"WorkWatts",9:"Finished",10:"Terminated"}
ROWING_STATE = {0:"Idle",1:"Active",2:"Inactive"}
STROKE_STATE = {0:"Unknown",1:"Ready",2:"Drive",3:"Dwell",4:"Recovery"}

latest = {"gen": {}, "add1": {}, "add2": {}}

def safe_get(b, idx):  # returns None if out of range
    return b[idx] if idx < len(b) else None

def parse_gen(b: bytes):
    out = {}
    if len(b) >= 6:
        out["elapsed_s"]  = u24le(b,0)/100.0
        out["distance_m"] = u24le(b,3)/10.0
    if len(b) >= 11:
        wt = safe_get(b,6); it = safe_get(b,7); ws = safe_get(b,8); rs = safe_get(b,9); ss = safe_get(b,10)
        if wt is not None: out["workout_type"] = wt
        if it is not None: out["interval_type"] = it
        if ws is not None: out["workout_state"] = WORKOUT_STATE.get(ws, ws)
        if rs is not None: out["rowing_state"]  = ROWING_STATE.get(rs, rs)
        if ss is not None: out["stroke_state"]  = STROKE_STATE.get(ss, ss)
    if len(b) >= 14:
        out["total_work_distance_m"] = u24le(b,11)
    return out

def parse_add1(b: bytes):
    out = {}
    if len(b) >= 3:
        out["elapsed_s"] = u24le(b,0)/100.0
    if len(b) >= 6:
        out["speed_m_s"] = u16le(b,3)/1000.0
        out["spm"] = safe_get(b,5)
    if len(b) >= 7:
        hr = safe_get(b,6)
        out["hr_bpm"] = None if hr is None or hr == 255 else hr
    if len(b) >= 9:
        out["pace_cur_s_per_500m"] = u16le(b,7)/100.0
    if len(b) >= 11:
        out["pace_avg_s_per_500m"] = u16le(b,9)/100.0
    if len(b) >= 13:
        out["rest_distance_m"] = u16le(b,11)
    if len(b) >= 16:
        out["rest_time_s"] = u24le(b,13)/100.0
    if len(b) >= 17:
        out["erg_machine_type"] = b[16]
    return out

def parse_add2(b: bytes):
    out = {}
    if len(b) >= 3:
        out["elapsed_s"] = u24le(b,0)/100.0
    if len(b) >= 4:
        out["interval_count"] = b[3]
    if len(b) >= 6:
        out["avg_power_w"] = u16le(b,4)
    if len(b) >= 8:
        out["total_calories"] = u16le(b,6)
    if len(b) >= 10:
        out["split_avg_pace_s_per_500m"] = u16le(b,8)/100.0
    if len(b) >= 12:
        out["split_avg_power_w"] = u16le(b,10)
    if len(b) >= 14:
        out["split_avg_cal_hr"] = u16le(b,12)
    if len(b) >= 17:
        out["last_split_time_s"] = u24le(b,14)/10.0
    if len(b) >= 20:
        out["last_split_distance_m"] = u24le(b,17)
    return out

def sec_to_mmss(x):
    if x is None: return None
    m = int(x // 60); s = x - 60*m
    return f"{m}:{s:04.1f}"

def merge_row():
    row = {}
    for k in ("gen","add1","add2"):
        row.update({kk: vv for kk,vv in latest[k].items() if vv is not None})
    if "pace_cur_s_per_500m" in row:
        row["pace_cur_mmss"] = sec_to_mmss(row["pace_cur_s_per_500m"])
    if "pace_avg_s_per_500m" in row:
        row["pace_avg_mmss"] = sec_to_mmss(row["pace_avg_s_per_500m"])
    return row

async def main():
    print("Scanning for PM5…")
    devs = await BleakScanner.discover(timeout=5.0)
    pm = next((d for d in devs if (d.name or "").startswith("PM5")), None)
    if not pm:
        raise SystemExit("No PM5 found. Open Bluetooth/Connect on the PM5 and try again.")

    print(f"Connecting to {pm.name} @ {pm.address} … (don’t use sudo for BLE)")
    async with BleakClient(pm) as cli:
        # Try to speed up notifications (ignore if unsupported)
        try:
            await cli.write_gatt_char(CHAR_RATE, bytes([3]), response=True)  # 3 ≈ 100ms
        except Exception:
            pass

        # CSV: include all fields we might emit
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"pm5_ble_parsed_{ts}.csv"
        fields = [
            "timestamp_iso","elapsed_s","distance_m","spm","hr_bpm","speed_m_s",
            "pace_cur_s_per_500m","pace_cur_mmss","pace_avg_s_per_500m","pace_avg_mmss",
            "avg_power_w","total_calories","workout_type","interval_type","workout_state",
            "rowing_state","stroke_state","interval_count","rest_distance_m","rest_time_s",
            "last_split_time_s","last_split_distance_m","total_work_distance_m","erg_machine_type",
            "split_avg_pace_s_per_500m","split_avg_power_w","split_avg_cal_hr"
        ]
        f = open(fname, "w", newline="")
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()

        def on_gen(_, data: bytearray):  latest["gen"]  = parse_gen(bytes(data))
        def on_add1(_, data: bytearray): latest["add1"] = parse_add1(bytes(data))
        def on_add2(_, data: bytearray): latest["add2"] = parse_add2(bytes(data))

        await cli.start_notify(CHAR_GEN, on_gen)
        await cli.start_notify(CHAR_ADD1, on_add1)
        await cli.start_notify(CHAR_ADD2, on_add2)
        print("Subscribed to 0x0031/0x0032/0x0033. Start ‘Just Row’ and pull…")
        print(f"Logging to {fname} (Ctrl+C to stop)")

        try:
            while True:
                row = merge_row()
                if row:
                    row["timestamp_iso"] = datetime.datetime.now().isoformat(timespec="seconds")
                    # Only write keys we have headers for
                    w.writerow({k: row.get(k) for k in fields}); f.flush()
                    # Pretty console line
                    et   = row.get("elapsed_s", 0.0)
                    dist = row.get("distance_m", 0.0)
                    spm  = row.get("spm", None)
                    hr   = row.get("hr_bpm", None)
                    pace = row.get("pace_cur_mmss", "-:--.-")
                    pwr  = row.get("avg_power_w", None)  # avg power from ADD2
                    spm_str = f"{spm:>2}spm" if spm is not None else "--spm"
                    hr_str  = f"{hr:>3}bpm" if hr is not None else "--bpm"
                    pwr_str = f"{pwr:>3}W"   if pwr is not None else "--W"
                    print(f"{et:6.1f}s {dist:7.1f}m  {spm_str}  {hr_str}  {pace}  {pwr_str}")
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                await cli.stop_notify(CHAR_GEN)
                await cli.stop_notify(CHAR_ADD1)
                await cli.stop_notify(CHAR_ADD2)
            except Exception:
                pass
            f.close()
            print("Saved:", fname)

if __name__ == "__main__":
    asyncio.run(main())
