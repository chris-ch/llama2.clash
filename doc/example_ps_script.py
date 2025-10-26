#!/usr/bin/env python3
import serial, struct, time, mmap, os, sys
from pathlib import Path

# -------------------------------------------------
# 1. Config
# -------------------------------------------------
DDR_BASE    = 0x8000_0000                # PL-visible DDR window (AXI master reads model here)
MODEL_PATH  = "/mnt/sdcard/model_ie8.bin"  # **Pre-quantized** IE8 binary (generated offline)

CTRL_BASE   = 0xA000_0000                # Base address of AXI-Lite control slave in PL
REG_PWR     = CTRL_BASE + 0x00           # [WO] powerOn – set to 1 to enable decoder
REG_TEMP    = CTRL_BASE + 0x04           # [WO] temperature (Q16.16 fixed-point)
REG_SEED    = CTRL_BASE + 0x08           # [WO] RNG seed
REG_IN_TOK  = CTRL_BASE + 0x0C           # [WO] input token (u32)
REG_IN_VAL  = CTRL_BASE + 0x10           # [WO] input token valid strobe (pulse 1→0)
REG_OUT_TOK = CTRL_BASE + 0x14           # [RO] output token from PL
REG_OUT_VAL = CTRL_BASE + 0x18           # [RO] output valid flag from PL
REG_MODEL_BASE = CTRL_BASE + 0x20        # [WO] Optional: tell PL where the model starts

# -------------------------------------------------
# 2. Model constants (must match Clash elaboration)
# -------------------------------------------------
VOCAB_SIZE      = 32000
MODEL_DIM       = 4096          # hidden size
HEAD_DIM        = 128
NUM_HEADS       = 32
NUM_LAYERS      = 32
HIDDEN_DIM      = 11008         # FFN intermediate
SEQ_LEN         = 2048          # maximum sequence length used for rotary cache
ROTARY_DIM      = 128           # same as HEAD_DIM

BLOCK_SIZE      = 128           # IE8 block size
MANTISSA_BITS   = 8
EXP_BITS        = 7
BYTES_PER_MANT  = 1
BYTES_PER_EXP   = 1             # 7-bit exponent padded to 1 byte
BYTES_PER_BLOCK = BLOCK_SIZE + BYTES_PER_EXP
ALIGN_PAD       = (4 - (BYTES_PER_BLOCK % 4)) % 4
BYTES_PER_BLOCK_ALIGNED = BYTES_PER_BLOCK + ALIGN_PAD   # 132 bytes

# -------------------------------------------------
# 2. Helper: size of an IE8 matrix (rows × cols)
# -------------------------------------------------
def ie8_matrix_bytes(rows: int, cols: int) -> int:
    """Return total aligned bytes for a MatI8E rows×cols."""
    blocks_per_row = (cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    row_bytes = blocks_per_row * BYTES_PER_BLOCK_ALIGNED
    return rows * row_bytes

# -------------------------------------------------
# 2. Compute DDR layout (exact isomorphism with DecoderParameters)
# -------------------------------------------------
offset = 0

# ---- EmbeddingComponentQ -------------------------------------------------
embed_matrix_bytes = ie8_matrix_bytes(VOCAB_SIZE, MODEL_DIM)
rms_final_bytes    = MODEL_DIM * 2                     # FP16 → 2 bytes each
offset_vocab_q     = offset
offset_rms_final   = offset_vocab_q + embed_matrix_bytes
offset += embed_matrix_bytes + rms_final_bytes

# ---- Per-layer offsets ---------------------------------------------------
layer_offsets = []                     # start address of each TransformerLayerComponent
head_offsets  = []                     # inside each layer, start of headsQ vector

for layer_idx in range(NUM_LAYERS):
    layer_start = offset
    layer_offsets.append(layer_start)

    # ---- MultiHeadAttentionComponentQ ------------------------------------
    # headsQ : Vec NUM_HEADS SingleHeadComponentQ
    head_start = offset
    head_offsets.append(head_start)

    for _ in range(NUM_HEADS):
        # wqHeadQ, wkHeadQ, wvHeadQ : each MatI8E HEAD_DIM × MODEL_DIM
        for _ in range(3):
            offset += ie8_matrix_bytes(HEAD_DIM, MODEL_DIM)
        # rotaryF : RotaryEncodingComponentF
        offset += SEQ_LEN * ROTARY_DIM * 2 * 2   # cos + sin, each FP16
    # mWoQ : Vec NUM_HEADS (MatI8E MODEL_DIM × HEAD_DIM)
    for _ in range(NUM_HEADS):
        offset += ie8_matrix_bytes(MODEL_DIM, HEAD_DIM)
    # rmsAttF : Vec MODEL_DIM FixedPoint (FP16)
    offset += MODEL_DIM * 2

    # ---- FeedForwardNetworkComponentQ ------------------------------------
    # fW1Q, fW3Q : MatI8E HIDDEN_DIM × MODEL_DIM
    for _ in range(2):
        offset += ie8_matrix_bytes(HIDDEN_DIM, MODEL_DIM)
    # fW2Q : MatI8E MODEL_DIM × HIDDEN_DIM
    offset += ie8_matrix_bytes(MODEL_DIM, HIDDEN_DIM)
    # fRMSFfnF : Vec MODEL_DIM FixedPoint (FP16)
    offset += MODEL_DIM * 2

# ---- Final size ---------------------------------------------------------
total_model_size = offset
print(f"[*] Expected IE8 model size: {total_model_size / 1024**2:.2f} MiB")

# -------------------------------------------------
# 2. Verify model file
# -------------------------------------------------
if not Path(MODEL_PATH).exists():
    print(f"[!] Model not found: {MODEL_PATH}")
    sys.exit(1)

actual_size = Path(MODEL_PATH).stat().st_size
if actual_size != total_model_size:
    print(f"[!] Size mismatch! Expected {total_model_size} bytes, got {actual_size}")
    sys.exit(1)

# -------------------------------------------------
# 2. Map DDR and copy model
# -------------------------------------------------
mem_fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
ddr_map = mmap.mmap(mem_fd, total_model_size,
                    offset=DDR_BASE,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE)

print(f"[*] Copying {total_model_size/1024**2:.1f} MiB IE8 model → DDR @ 0x{DDR_BASE:08X}")
with open(MODEL_PATH, "rb") as f:
    pos = 0
    chunk_sz = 1 << 20                     # 1 MiB chunks
    while True:
        chunk = f.read(chunk_sz)
        if not chunk:
            break
        ddr_map[pos:pos + len(chunk)] = chunk
        pos += len(chunk)
print("[+] Model loaded into DDR.")

# -------------------------------------------------
# 2. (Optional) Tell PL the base address
# -------------------------------------------------
def w32(addr: int, val: int):
    mmap.mmap(mem_fd, 4, offset=addr, prot=mmap.PROT_WRITE)[0:4] = struct.pack("<I", val)

w32(REG_MODEL_BASE, DDR_BASE)   # PL can read this register if needed

# -------------------------------------------------
# 3. USB CDC-ACM setup (host ↔ PS)
# -------------------------------------------------
ser = serial.Serial('/dev/ttyACM0', 2000000, timeout=5)
REQ_NEXT_TOKEN = 0x01   # Host → PS: request next token
RESP_TOKEN     = 0x81   # PS → Host: reply with generated token

# -------------------------------------------------
# 4. /dev/mem register access helpers
# -------------------------------------------------
def r32(addr: int) -> int:
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
        continue

    _, in_token, temp_q16 = struct.unpack("<BII", req)

    # --- 2. Send data to PL (PS → PL) ---
    w32(REG_IN_TOK, in_token)
    w32(REG_TEMP,   temp_q16)
    w32(REG_IN_VAL, 1)
    time.sleep(1e-6)
    w32(REG_IN_VAL, 0)          # rising edge

    # --- 3. Wait for PL output ---
    while r32(REG_OUT_VAL) == 0:
        pass

    # --- 4. Read result ---
    out_token = r32(REG_OUT_TOK)

    # --- 5. Reply to host ---
    resp = struct.pack("<BI", RESP_TOKEN, out_token)
    ser.write(resp)

    print(f"→ {in_token} @ {temp_q16/(1<<16):.3f} → {out_token}")
