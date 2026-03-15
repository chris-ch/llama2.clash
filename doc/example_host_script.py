#!/usr/bin/env python3
"""
Host-side script (laptop/PC):
  - Sends raw text prompt to Zynq over USB.
  - Zynq performs on-device tokenization using token.bin + model.bin.
  - Receives generated tokens back and prints them.
"""

import serial, struct, sys

# -------------------------------------------------
# 1. USB setup (new protocol: text → tokens)
# -------------------------------------------------
ser = serial.Serial('/dev/ttyACM0', 2000000, timeout=10)  # Linux; use COMx on Windows

REQ_INFER_TEXT = 0x02   # Host → PS: send null-terminated UTF-8 string
RESP_TOKEN     = 0x81   # PS → Host: single generated token (u32)

# -------------------------------------------------
# 2. Send text prompt (null-terminated)
# -------------------------------------------------
def send_prompt(text: str):
    data = text.encode('utf-8') + b'\0'  # null-terminate
    # Optional: length prefix if strings can contain \0
    ser.write(data)
    print(f"Sent prompt ({len(data)} bytes): {text!r}")

# -------------------------------------------------
# 3. Receive stream of output tokens until EOS
# -------------------------------------------------
def receive_tokens(eos_tokens=frozenset([0, 1, 2])):
    print("Generated tokens:", end=' ', flush=True)
    while True:
        resp = ser.read(5)
        if len(resp) != 5 or resp[0] != RESP_TOKEN:
            raise RuntimeError("Invalid response frame")
        _, token = struct.unpack("<BI", resp)
        print(token, end=' ', flush=True)
        if token in eos_tokens:
            print("(EOS)")
            break

# -------------------------------------------------
# 4. Main
# -------------------------------------------------
if __name__ == "__main__":
    prompt = "Once upon a time"
    try:
        send_prompt(prompt)
        receive_tokens()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        ser.close()
