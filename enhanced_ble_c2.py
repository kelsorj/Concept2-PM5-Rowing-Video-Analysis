#!/usr/bin/env python3
"""
Enhanced BLE Concept2 PM5 Interface
-----------------------------------
Collects ForcePlotData correctly via BLE using Concept2's CSAFE commands.

Requires: bleak (pip install bleak)
"""

import asyncio
from bleak import BleakClient, BleakScanner

# Concept2 BLE UUIDs
SERVICE_UUID             = "ce060010-43e5-11e4-916c-0800200c9a66"
CONTROL_CHAR_UUID        = "ce060031-43e5-11e4-916c-0800200c9a66"  # write
STROKE_STATE_CHAR_UUID   = "ce060021-43e5-11e4-916c-0800200c9a66"  # notify
FORCEPLOT_CHAR_UUID      = "ce060022-43e5-11e4-916c-0800200c9a66"  # notify

# CSAFE command constants
CSAFE_SETUSERCFG1_CMD = 0x1A
CSAFE_PM_GET_FORCEPLOTDATA = 0x6F
FRAME_START = 0xF1
FRAME_STOP  = 0xF2

current_stroke_state = None


def build_forceplot_request() -> bytes:
    """
    Construct CSAFE frame to request ForcePlotData:
    F1 1A 03 6F 00 00 F2
    """
    return bytes([
        FRAME_START,
        CSAFE_SETUSERCFG1_CMD,
        0x03,
        CSAFE_PM_GET_FORCEPLOTDATA,
        0x00,
        0x00,
        FRAME_STOP,
    ])


async def handle_stroke_state(sender, data, client):
    """
    Stroke State characteristic notifications:
      0 Waiting for wheel min speed
      1 Waiting for acceleration
      2 Driving
      3 Dwelling after drive (end of stroke)
      4 Recovery
    """
    global current_stroke_state
    new_state = data[0]

    # detect drive -> dwell transition
    if current_stroke_state == 2 and new_state == 3:
        print("→ End of stroke detected, requesting ForcePlotData...")
        await client.write_gatt_char(CONTROL_CHAR_UUID, build_forceplot_request())

    current_stroke_state = new_state


def handle_forceplot(sender, data):
    """
    Notification handler for ForcePlot characteristic.
    byte[0] = number of samples
    byte[1:N] = samples (0–255)
    """
    if not data:
        return

    count = data[0]
    samples = list(data[1:1 + count])
    print(f"ForcePlot ({count} samples): {samples}")


async def connect_and_run():
    print("Scanning for Concept2 PM5...")
    device = await BleakScanner.find_device_by_filter(
        lambda d, ad: d.name and "PM5" in d.name
    )

    if not device:
        print("❌ No PM5 found nearby.")
        return

    print(f"Connecting to {device.name} [{device.address}]...")
    async with BleakClient(device) as client:
        # Subscribe to stroke state + forceplot notifications
        await client.start_notify(
            STROKE_STATE_CHAR_UUID,
            lambda s, d: asyncio.create_task(handle_stroke_state(s, d, client)),
        )
        await client.start_notify(FORCEPLOT_CHAR_UUID, handle_forceplot)

        print("✅ Connected. Start rowing to generate data. (Ctrl+C to stop)")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            await client.stop_notify(STROKE_STATE_CHAR_UUID)
            await client.stop_notify(FORCEPLOT_CHAR_UUID)


if __name__ == "__main__":
    asyncio.run(connect_and_run())
