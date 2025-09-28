#!/usr/bin/env python3
# Enhanced Concept2 PM5 BLE logger with power curve parsing
# - Subscribes to all C2 Rowing Service characteristics
# - Parses 0x0035/0x0036 for detailed power data
# - Captures instantaneous power and stroke data

import asyncio, csv, datetime, time, struct
from bleak import BleakScanner, BleakClient

# Concept2 Rowing Service UUID base
BASE = "ce06{:04x}-43e5-11e4-916c-0800200c9a66".format

# Notify characteristics
CHAR_GEN    = BASE(0x0031)  # General Status
CHAR_ADD1   = BASE(0x0032)  # Additional Status 1
CHAR_ADD2   = BASE(0x0033)  # Additional Status 2
CHAR_EXTRA1 = BASE(0x0035)  # Extra stream (stroke data)
CHAR_EXTRA2 = BASE(0x0036)  # Extra stream (summary data)

# Write characteristic: sample rate control
CHAR_RATE   = BASE(0x0034)

# --- helpers -----------------------------------------------------------------

def u16le(b, i): return b[i] | (b[i+1] << 8)
def u24le(b, i): return b[i] | (b[i+1] << 8) | (b[i+2] << 16)
def u32le(b, i): return b[i] | (b[i+1] << 8) | (b[i+2] << 16) | (b[i+3] << 24)

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

def safe_get(b, idx):
    return b[idx] if idx < len(b) else None

# --- parsers for main characteristics ---------------------------------------

def parse_gen(b: bytes):
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

# --- parsers for extra characteristics (power curve data) -------------------

def parse_extra1(b: bytes):
    """Parse 0x0035 - Stroke data with power curve"""
    out = {}
    if len(b) < 4:
        return out
    
    # Common fields
    out["elapsed_s"] = u24le(b, 0) / 100.0
    
    if len(b) >= 7:
        out["stroke_count"] = u16le(b, 3)
        out["stroke_rate_spm"] = b[5] / 2.0
        out["stroke_state"] = STROKE_STATE.get(b[6], b[6])
    
    if len(b) >= 9:
        power_val = u16le(b, 7)
        # Sanity check: rowing power shouldn't exceed 1000W
        if power_val <= 1000:
            out["instantaneous_power_w"] = power_val
    
    if len(b) >= 11:
        out["stroke_distance_m"] = u16le(b, 9) / 100.0
    
    if len(b) >= 13:
        out["stroke_time_ms"] = u16le(b, 11)
    
    # Power curve data (if available)
    if len(b) >= 15:
        out["drive_time_ms"] = u16le(b, 13)
    
    if len(b) >= 17:
        out["recovery_time_ms"] = u16le(b, 15)
    
    # Peak power in stroke
    if len(b) >= 19:
        peak_power_val = u16le(b, 17)
        # Sanity check: rowing power shouldn't exceed 1000W
        if peak_power_val <= 1000:
            out["peak_power_w"] = peak_power_val

    # Force curve data (variable length after peak power)
    if len(b) > 19:
        force_data = []
        for i in range(19, len(b), 2):
            if i + 1 < len(b):
                force_val = u16le(b, i)
                # Only include reasonable force values
                if 0 <= force_val <= 2000:  # Typical rowing force range
                    force_data.append(force_val)
        if force_data:
            out["force_curve"] = force_data
            print(f"DEBUG: Captured force curve with {len(force_data)} points: {force_data[:10]}...")

    return out

def parse_extra2(b: bytes):
    """Parse 0x0036 - Summary data"""
    out = {}
    if len(b) < 4:
        return out
    
    out["elapsed_s"] = u24le(b, 0) / 100.0
    
    if len(b) >= 6:
        out["total_strokes"] = u16le(b, 3)
    
    if len(b) >= 8:
        out["avg_stroke_rate_spm"] = u16le(b, 5) / 2.0
    
    if len(b) >= 10:
        out["total_distance_m"] = u16le(b, 7)
    
    if len(b) >= 12:
        out["avg_power_w"] = u16le(b, 9)
    
    if len(b) >= 14:
        out["total_calories"] = u16le(b, 11)
    
    return out

# --- main --------------------------------------------------------------------

async def main():
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            print(f"Scanning for PM5… (attempt {retry_count + 1}/{max_retries})")
            devs = await BleakScanner.discover(timeout=5.0)
            pm = next((d for d in devs if (d.name or "").startswith("PM5")), None)
            if not pm:
                if retry_count < max_retries - 1:
                    print("No PM5 found. Make sure PM5 is powered on and Bluetooth enabled.")
                    print("Retrying in 3 seconds...")
                    await asyncio.sleep(3.0)
                    retry_count += 1
                    continue
                else:
                    raise SystemExit("No PM5 found after multiple attempts. On the PM5, open Bluetooth/Connect and try again.")

            print(f"Connecting to {pm.name} @ {pm.address} …")
            async with BleakClient(pm) as cli:
                # Set fastest notify rate
                try:
                    await cli.write_gatt_char(CHAR_RATE, bytes([3]), response=True)
                    print("Set sample rate to fastest (~100ms)")
                except Exception as e:
                    print(f"Could not set sample rate: {e}")

        # Prepare CSVs
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        parsed_name = f"pm5_enhanced_parsed_{ts}.csv"
        raw_name    = f"pm5_enhanced_raw_{ts}.csv"
        power_name  = f"pm5_power_curve_{ts}.csv"

        # Enhanced parsed fields including power curve data
        parsed_fields = [
            "timestamp_iso","elapsed_s","distance_m","spm","hr_bpm","speed_m_s",
            "pace_cur_s_per_500m","pace_cur_mmss","pace_avg_s_per_500m","pace_avg_mmss",
            "avg_power_w","instantaneous_power_w","peak_power_w","total_calories",
            "workout_type","interval_type","workout_state","rowing_state","stroke_state",
            "interval_count","rest_distance_m","rest_time_s",
            "last_split_time_s","last_split_distance_m","total_work_distance_m","erg_machine_type",
            "split_avg_pace_s_per_500m","split_avg_power_w","split_avg_cal_hr",
            "stroke_count","stroke_rate_spm","stroke_distance_m","stroke_time_ms",
            "drive_time_ms","recovery_time_ms","total_strokes","avg_stroke_rate_spm"
        ]
        
        pf = open(parsed_name, "w", newline="")
        pw = csv.DictWriter(pf, fieldnames=parsed_fields); pw.writeheader()

        raw_fields = ["ts_ns","char","hex"]
        rf = open(raw_name, "w", newline="")
        rw = csv.DictWriter(rf, fieldnames=raw_fields); rw.writeheader()

        # Power curve specific CSV
        power_fields = ["timestamp_iso","elapsed_s","instantaneous_power_w","peak_power_w","stroke_count","stroke_state"]
        powerf = open(power_name, "w", newline="")
        powerw = csv.DictWriter(powerf, fieldnames=power_fields); powerw.writeheader()

        # Hi-res timebase
        t0 = time.perf_counter_ns()
        def ts_ns(): return time.perf_counter_ns() - t0

        latest = {"gen": {}, "add1": {}, "add2": {}, "extra1": {}, "extra2": {}}

        def log_raw(tag: str, data: bytes):
            rw.writerow({"ts_ns": ts_ns(), "char": tag, "hex": data.hex()})
            rf.flush()

        def log_power_curve(data):
            if data.get("instantaneous_power_w") is not None:
                power_row = {
                    "timestamp_iso": datetime.datetime.now().isoformat(timespec="milliseconds"),
                    "elapsed_s": data.get("elapsed_s"),
                    "instantaneous_power_w": data.get("instantaneous_power_w"),
                    "peak_power_w": data.get("peak_power_w"),
                    "stroke_count": data.get("stroke_count"),
                    "stroke_state": data.get("stroke_state")
                }
                powerw.writerow(power_row)
                powerf.flush()

        def merge_row():
            row = {}
            for k in ("gen","add1","add2","extra1","extra2"):
                for kk,vv in latest[k].items():
                    if vv is not None:
                        row[kk] = vv
            # Friendly pace strings
            if "pace_cur_s_per_500m" in row:
                row["pace_cur_mmss"] = sec_to_mmss(row["pace_cur_s_per_500m"])
            if "pace_avg_s_per_500m" in row:
                row["pace_avg_mmss"] = sec_to_mmss(row["pace_avg_s_per_500m"])
            return row

        # Notify handlers
        def on_gen(_, data: bytearray):
            b = bytes(data); log_raw("0031", b); latest["gen"] = parse_gen(b)
        def on_add1(_, data: bytearray):
            b = bytes(data); log_raw("0032", b); latest["add1"] = parse_add1(b)
        def on_add2(_, data: bytearray):
            b = bytes(data); log_raw("0033", b); latest["add2"] = parse_add2(b)
        def on_extra1(_, data: bytearray):
            b = bytes(data)
            log_raw("0035", b)
            print(f"DEBUG: EXTRA1 received {len(b)} bytes: {b.hex()[:100]}...")
            parsed = parse_extra1(b)
            latest["extra1"] = parsed
            if "force_curve" in parsed:
                print(f"DEBUG: Force curve captured with {len(parsed['force_curve'])} points!")
            log_power_curve(parsed)  # Log power curve data separately
        def on_extra2(_, data: bytearray):
            b = bytes(data); log_raw("0036", b); latest["extra2"] = parse_extra2(b)

        # Subscribe to all characteristics
        await cli.start_notify(CHAR_GEN,    on_gen)
        await cli.start_notify(CHAR_ADD1,   on_add1)
        await cli.start_notify(CHAR_ADD2,   on_add2)
        await cli.start_notify(CHAR_EXTRA1, on_extra1)
        await cli.start_notify(CHAR_EXTRA2, on_extra2)

        print("Subscribed to all characteristics (0x0031-0x0036).")
        print(f"Logging enhanced parsed data → {parsed_name}")
        print(f"Logging raw packets → {raw_name}")
        print(f"Logging power curve data → {power_name}")
        print("Start 'Just Row' and pull… (Ctrl+C to stop)")

        # Connection monitoring
        last_data_time = time.time()
        connection_warnings = 0
        last_heartbeat = time.time()

        # UI loop
        try:
            while True:
                current_time = time.time()

                # Send heartbeat every 30 seconds to keep PM5 active
                if current_time - last_heartbeat > 30.0:
                    try:
                        # Try to read a characteristic to keep connection alive
                        await cli.read_gatt_char(CHAR_GEN)
                        last_heartbeat = current_time
                        print("💓 Heartbeat sent to keep PM5 active")
                    except Exception as hb_error:
                        print(f"⚠️  Heartbeat failed: {hb_error}")

                row = merge_row()

                if row:
                    last_data_time = current_time
                    connection_warnings = 0

                    row["timestamp_iso"] = datetime.datetime.now().isoformat(timespec="milliseconds")
                    pw.writerow({k: row.get(k) for k in parsed_fields}); pf.flush()

                    # Enhanced console display
                    et   = f"{row.get('elapsed_s',0.0):6.1f}s"
                    dist = f"{row.get('distance_m',0.0):7.1f}m"
                    spm  = f"{row.get('spm','--'):>2}spm"
                    hr   = f"{row.get('hr_bpm','--'):>3}bpm"
                    pace = row.get("pace_cur_mmss","-:--.-")
                    avg_pwr = f"{row.get('avg_power_w','--'):>3}W"
                    inst_pwr = f"{row.get('instantaneous_power_w','--'):>3}W"
                    peak_pwr = f"{row.get('peak_power_w','--'):>3}W"
                    stroke = f"{row.get('stroke_count','--'):>3}"

                    print(f"{et} {dist}  {spm}  {hr}  {pace}  Avg:{avg_pwr}  Inst:{inst_pwr}  Peak:{peak_pwr}  Stroke:{stroke}")
                else:
                    # Check for stale connection
                    if current_time - last_data_time > 5.0 and connection_warnings == 0:
                        print("⚠️  No data received for 5 seconds - PM5 may have stopped transmitting")
                        print("   Try: 1) Wake PM5 by pressing buttons, 2) Check battery, 3) Reconnect BLE")
                        connection_warnings += 1
                    elif current_time - last_data_time > 15.0 and connection_warnings == 1:
                        print("⚠️  Still no data - PM5 may need restart or BLE reconnection")
                        connection_warnings += 1

                await asyncio.sleep(0.05)  # 20Hz refresh
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"❌ Connection error: {e}")
            print("   Attempting to reconnect... (Ctrl+C to exit)")

            # Try to reconnect
            try:
                await asyncio.sleep(2.0)  # Brief pause before reconnect
                print("🔄 Attempting BLE reconnection...")
                # The async with context will handle reconnection
            except Exception as reconn_error:
                print(f"❌ Reconnection failed: {reconn_error}")
                finally:
                    # Cleanup
                    for ch in (CHAR_GEN, CHAR_ADD1, CHAR_ADD2, CHAR_EXTRA1, CHAR_EXTRA2):
                        try: await cli.stop_notify(ch)
                        except Exception: pass
                    pf.close(); rf.close(); powerf.close()
                    print("Saved:")
                    print("  ", parsed_name)
                    print("  ", raw_name)
                    print("  ", power_name)
                    return  # Success - exit retry loop

        except Exception as conn_error:
            print(f"❌ Connection failed (attempt {retry_count + 1}): {conn_error}")
            if retry_count < max_retries - 1:
                print("Retrying in 5 seconds...")
                await asyncio.sleep(5.0)
                retry_count += 1
                continue
            else:
                raise SystemExit(f"Failed to connect after {max_retries} attempts: {conn_error}")

if __name__ == "__main__":
    asyncio.run(main())
