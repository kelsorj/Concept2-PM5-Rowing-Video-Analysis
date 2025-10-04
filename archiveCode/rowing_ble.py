#!/usr/bin/env python3
import asyncio, argparse, csv, datetime as dt, struct
from typing import Dict, Any, Optional
from bleak import BleakClient, BleakScanner, BleakError

FTMS = "00001826-0000-1000-8000-00805f9b34fb"
CHAR_ROWING = "00002ad1-0000-1000-8000-00805f9b34fb"   # Indoor Rowing Machine Data
CHAR_STATUS = "00002ada-0000-1000-8000-00805f9b34fb"   # Fitness Machine Status (notify)
CHAR_CP     = "00002ad9-0000-1000-8000-00805f9b34fb"   # Fitness Machine Control Point (write)

CSV_FIELDS = ["timestamp_iso","device_name","stroke_rate_spm","stroke_count",
              "pace_s_per_500m","power_watts","total_distance_m","heart_rate_bpm",
              "elapsed_time_s","raw_hex"]

def parse_rowing(payload: bytes) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    off = 0
    try:
        if len(payload) < 2: return d
        flags, = struct.unpack_from("<H", payload, off); off += 2
        if len(payload) >= off+1:
            sr_raw, = struct.unpack_from("<B", payload, off); off += 1
            d["stroke_rate_spm"] = sr_raw / 2.0
        if len(payload) >= off+2:
            sc, = struct.unpack_from("<H", payload, off); off += 2
            d["stroke_count"] = sc
        has = lambda b: (flags & (1<<b)) != 0
        if has(1) and len(payload) >= off+1: off += 1  # avg SR (ignore)
        if has(2) and len(payload) >= off+3:
            a,b,c = struct.unpack_from("<BBB", payload, off); off += 3
            d["total_distance_m"] = a | (b<<8) | (c<<16)
        if has(3) and len(payload) >= off+2:
            pace, = struct.unpack_from("<H", payload, off); off += 2
            d["pace_s_per_500m"] = pace
        if has(4) and len(payload) >= off+2: off += 2  # avg pace
        if has(5) and len(payload) >= off+2:
            power, = struct.unpack_from("<h", payload, off); off += 2
            d["power_watts"] = power
        if has(6) and len(payload) >= off+2: off += 2  # avg power
        if has(7) and len(payload) >= off+2: off += 2  # resistance
        if has(8) and len(payload) >= off+6: off += 6  # energy triple
        if has(9) and len(payload) >= off+1:
            hr, = struct.unpack_from("<B", payload, off); off += 1
            d["heart_rate_bpm"] = hr
        if has(10) and len(payload) >= off+1: off += 1  # MET
        if has(11) and len(payload) >= off+3:
            a,b,c = struct.unpack_from("<BBB", payload, off); off += 3
            d["elapsed_time_s"] = a | (b<<8) | (c<<16)
        if has(12) and len(payload) >= off+3: off += 3  # remaining time
    except Exception:
        pass
    return d

async def scan_pm5(name: Optional[str]) -> Optional[str]:
    print("Scanning for PM5…")
    found = await BleakScanner.discover(timeout=6.0)
    if name:
        for d in found:
            if (d.name or "").strip().lower() == name.strip().lower():
                print(f"Found target: {d.name} @ {d.address}")
                return d.address
    for d in found:
        n = (d.name or "").upper()
        if "PM5" in n or "CONCEPT2" in n or "CONCEPT 2" in n:
            print(f"Found PM5-like: {d.name} @ {d.address}")
            return d.address
    # fallback: any device advertising FTMS
    for d in found:
        uuids = set(u.lower() for u in (d.metadata.get("uuids") or []))
        if FTMS in uuids:
            print(f"Found FTMS device: {d.name} @ {d.address}")
            return d.address
    print("No PM5 found. Put PM5 on Bluetooth/Connect screen.")
    return None

async def run(device_name: Optional[str], csv_path: Optional[str]):
    addr = await scan_pm5(device_name)
    if not addr: return

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = csv_path or f"concept2_ble_{ts}.csv"
    f = open(out_csv, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS); writer.writeheader()

    def on_row(_, data: bytearray):
        parsed = parse_rowing(bytes(data))
        row = {
            "timestamp_iso": dt.datetime.now().isoformat(timespec="seconds"),
            "device_name": device_name or "",
            "stroke_rate_spm": parsed.get("stroke_rate_spm"),
            "stroke_count": parsed.get("stroke_count"),
            "pace_s_per_500m": parsed.get("pace_s_per_500m"),
            "power_watts": parsed.get("power_watts"),
            "total_distance_m": parsed.get("total_distance_m"),
            "heart_rate_bpm": parsed.get("heart_rate_bpm"),
            "elapsed_time_s": parsed.get("elapsed_time_s"),
            "raw_hex": data.hex(),
        }
        writer.writerow(row); f.flush()
        print(f"[{row['timestamp_iso']}] SR={row['stroke_rate_spm'] or '-'}  "
              f"Pace={row['pace_s_per_500m'] or '-'}  "
              f"Pwr={row['power_watts'] or '-'}  "
              f"Dist={row['total_distance_m'] or '-'}  "
              f"HR={row['heart_rate_bpm'] or '-'}  "
              f"t={row['elapsed_time_s'] or '-'}")

    def on_status(_, data: bytearray):
        # Just print Fitness Machine Status notifications for debugging
        print("Status notify:", data.hex())

    print(f"Connecting to {addr} …")
    try:
        async with BleakClient(addr) as client:
            if not client.is_connected:
                raise BleakError("BLE connect failed")

            # Dump services/characteristics so we can see what's there
            svcs = getattr(client, "services", None)
            if svcs is None:
                raise BleakError("No GATT services available on this bleak version")
            print("=== GATT dump ===")
            for s in svcs:
                print("Service:", s.uuid)
                for c in s.characteristics:
                    print("  Char:", c.uuid, "props=", c.properties)

            # Find FTMS chars
            rowing_uuid = None
            status_uuid = None
            cp_uuid = None

            for s in svcs:
                if str(s.uuid).lower() == FTMS:
                    for c in s.characteristics:
                        cu = str(c.uuid).lower()
                        props = set(c.properties)
                        if cu == CHAR_ROWING or ("notify" in props and cu.endswith("2ad1-0000-1000-8000-00805f9b34fb")):
                            rowing_uuid = c.uuid
                        if cu == CHAR_STATUS or ("notify" in props and cu.endswith("2ada-0000-1000-8000-00805f9b34fb")):
                            status_uuid = c.uuid
                        if cu == CHAR_CP or ("write" in props and cu.endswith("2ad9-0000-1000-8000-00805f9b34fb")):
                            cp_uuid = c.uuid

            # Subscribe to rowing data or fallback to any FTMS notifiable char
            if rowing_uuid is None:
                # fallback: any notifiable under FTMS
                for s in svcs:
                    if str(s.uuid).lower() == FTMS:
                        for c in s.characteristics:
                            if "notify" in c.properties:
                                rowing_uuid = c.uuid; break
                    if rowing_uuid: break

            if rowing_uuid is None:
                raise BleakError("Could not find notifiable FTMS characteristic.")

            # Subscribe status if available (helpful to see control point responses)
            if status_uuid:
                await client.start_notify(status_uuid, on_status)

            await client.start_notify(rowing_uuid, on_row)
            print(f"Subscribed to rowing notifications on {rowing_uuid}")

            # Send FTMS Control Point ops if available: Request Control (0x00), Start/Resume (0x07)
            if cp_uuid:
                try:
                    await client.write_gatt_char(cp_uuid, bytes([0x00]), response=True)
                    await asyncio.sleep(0.2)
                    await client.write_gatt_char(cp_uuid, bytes([0x07]), response=True)
                    print("Sent FTMS Control Point: Request Control, Start/Resume")
                except Exception as e:
                    print("Control Point write failed (continuing):", repr(e))
            else:
                print("No Control Point char; if data remains idle, start a workout manually on PM5.")

            print(f"Logging to {out_csv}. Start 'Just Row' on PM5. Ctrl+C to stop.")
            while True:
                await asyncio.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopping…")
    except Exception as e:
        print("Error:", repr(e))
    finally:
        try: f.close()
        except Exception: pass
        print("CSV saved to:", out_csv)

def main():
    ap = argparse.ArgumentParser(description="Concept2 PM5 BLE logger (FTMS)")
    ap.add_argument("--device-name", help="Exact BLE device name to match (optional).")
    ap.add_argument("--csv", help="Output CSV path (optional).")
    args = ap.parse_args()
    asyncio.run(run(args.device_name, args.csv))

if __name__ == "__main__":
    main()
