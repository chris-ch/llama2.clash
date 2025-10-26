#!/usr/bin/env python3
import serial, struct, time, mmap, os, sys
from pathlib import Path

# -------------------------------------------------
# 1. Config
# -------------------------------------------------
DDR_BASE    = 0x8000_0000                # PL-visible DDR window (AXI master reads model here)
MODEL_PATH  = "/mnt/sdcard/model.bin"    # Source file on SD card

CTRL_BASE   = 0xA000_0000                # Base address of AXI-Lite control slave in PL
REG_PWR     = CTRL_BASE + 0x00           # [WO] powerOn – set to 1 to enable decoder
REG_TEMP    = CTRL_BASE + 0x04           # [WO] temperature (Q16.16 fixed-point)
REG_SEED    = CTRL_BASE + 0x08           # [WO] RNG seed
REG_IN_TOK  = CTRL_BASE + 0x0C           # [WO] input token (u32)
REG_IN_VAL  = CTRL_BASE + 0x10           # [WO] input token valid strobe (pulse 1→0)
REG_OUT_TOK = CTRL_BASE + 0x14           # [RO] output token from PL
REG_OUT_VAL = CTRL_BASE + 0x18           # [RO] output valid flag from PL

# -------------------------------------------------
# 2. Load model.bin → DDR (PL reads parameters via AXI master)
# -------------------------------------------------
if not Path(MODEL_PATH).exists():
    print(f"[!] Model not found: {MODEL_PATH}")
    sys.exit(1)

model_size = Path(MODEL_PATH).stat().st_size
mem_fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
ddr_map, offset = mmap.mmap(mem_fd, model_size, offset=DDR_BASE)

print(f"[*] Copying {model_size/1024**2:.1f} MiB model to DDR @ 0x{DDR_BASE:08X}")
with open(MODEL_PATH, "rb") as f:
    chunk = f.read(1<<20)
    pos = 0
    while chunk:
        ddr_map[offset + pos : offset + pos + len(chunk)] = chunk
        pos += len(chunk)
        chunk = f.read(1<<20)
print("[+] Model loaded.")

# -------------------------------------------------
# 3. USB CDC-ACM setup (host ↔ PS)
# -------------------------------------------------
ser = serial.Serial('/dev/ttyACM0', 2000000, timeout=5)
REQ_NEXT_TOKEN = 0x01   # Host → PS: request next token
RESP_TOKEN     = 0x81   # PS → Host: reply with generated token

# -------------------------------------------------
# 4. /dev/mem register access helpers
# -------------------------------------------------
def w32(addr, val):
    """Write 32-bit value to PL control register (PS → PL data)"""
    mmap.mmap(mem_fd, 4, offset=addr, prot=mmap.PROT_WRITE)[0:4] = struct.pack("<I", val)

def r32(addr):
    """Read 32-bit value from PL status register (PL → PS data)"""
    return struct.unpack("<I", mmap.mmap(mem_fd, 4, offset=addr)[0:4])[0]

# -------------------------------------------------
# 5. Initialize PL decoder
# -------------------------------------------------
w32(REG_PWR,  1)          # Assert powerOn – enables clock/reset in PL
w32(REG_SEED, 0x12345678) # Optional: fixed seed for reproducibility

# -------------------------------------------------
# 6. Main request/response loop (host ↔ PL via PS)
# -------------------------------------------------
print("[*] Ready. Waiting for host requests via USB...")
while True:
    # --- 1. Receive request from host (9 bytes) ---
    req = ser.read(9)
    if len(req) != 9 or req[0] != REQ_NEXT_TOKEN:
        continue  # Ignore malformed packets

    _, in_token, temp_q16 = struct.unpack("<BII", req)
    # → Data received from host: input token + temperature

    # --- 2. Send data to PL (PS → PL) ---
    w32(REG_IN_TOK, in_token)      # Input token
    w32(REG_TEMP,   temp_q16)      # Temperature (Q16.16)
    w32(REG_IN_VAL, 1)             # Assert valid
    time.sleep(1e-6)
    w32(REG_IN_VAL, 0)             # Deassert → creates rising edge

    # --- 3. Wait for PL to produce output (poll status) ---
    while r32(REG_OUT_VAL) == 0:
        pass  # PL sets OUT_VAL=1 when token is ready

    # --- 4. Read result from PL (PL → PS) ---
    out_token = r32(REG_OUT_TOK)
    # → Data received from PL: generated token

    # --- 5. Send response back to host ---
    resp = struct.pack("<BI", RESP_TOKEN, out_token)
    ser.write(resp)
    # → Data sent to host: output token

    print(f"→ {in_token} @ {temp_q16/(1<<16):.3f} → {out_token}")
