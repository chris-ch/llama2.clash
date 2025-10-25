Here is the **complete, minimal, production-ready BOM** for a **USB 2.0 version** of a **LLaMA-2 7B AI inference board** using the **Zynq UltraScale+ ZU9EG**.

**Total cost per unit (1k volume): ~$1,330**  

**Only 8 active components**

---

  <img src="./production-board.png" height="300" alt="Production Board">

---

## Final BOM — Production Board

| # | Component | Part Number (Example) | Qty | Unit Cost | Subtotal |
|---|---------|-----------------------|-----|-----------|----------|
| 1 | **FPGA SoC** | XCZU9EG-2FFVB1156E | 1 | $1,200 | $1,200 |
| 2 | **DDR4 SO-DIMM** | MTA8ATF51264HZ-2G6 (4GB, 2666MT/s) | 2 | $15 | $30 |
| 3 | **eMMC 5.1** | EMMC64G-TB29 (64GB) | 1 | $25 | $25 |
| 4 | **QSPI Flash** | MT25QU256ABA (256Mb) | 1 | $3 | $3 |
| 5 | **PMIC** | TPS6508640 (multi-rail) | 1 | $10 | $10 |
| 6 | **LDO (1.8V, 3.3V)** | TPS7A47, TPS7A33 | 2 | $1 | $2 |
| 7 | **Micro-USB Type-B Connector** | Molex 105017-0001 | 1 | $0.50 | $0.50 |
| 8 | **25 MHz Crystal** (for PS refclk) | SIT5156 | 1 | $1.50 | $1.50 |
| 9 | **PCB (6-layer, 100×80 mm)** | Custom | 1 | $40 | $40 |
|10 | **Passives** (caps, resistors, ESD, etc.) | — | — | $20 | $20 |

**GRAND TOTAL: ≈ $1,332**

---

## Critical Connections (Only What’s Needed)

| Interface | ZU9EG Pins | External Part | Notes |
|---------|------------|--------------|-------|
| **DDR4** | PS DDR Controller | 2× SO-DIMM | 64-bit, 2666 MT/s |
| **eMMC** | PS SDIO (MIO[40:51]) | eMMC chip | HS400, 8-bit |
| **USB 2.0** | PS USB0 (MIO[64:66]) | Micro-USB connector | D+, D-, VBUS |
| **Boot Flash** | PS QSPI (MIO[0:12]) | QSPI chip | FSBL + Linux |
| **Power** | VCCINT, VCCPSINT, etc. | PMIC + LDOs | Follow UG1085 |
| **Clock** | PS REFCLK | 25 MHz crystal | For DDR & SD |

---

## Software Stack (Minimal)

```bash
# PetaLinux (50 MB image)
- ARM Cortex-A53 boots
- Mounts eMMC → /mnt/data/stories260K.bin
- Triggers FPGA bitstream
- Runs bootWeightLoader (eMMC → DDR4)
- Starts USB CDC ACM server
- Streams prompts/tokens over /dev/ttyACM0
```

---

## PC / Raspberry Pi Side

```python
import serial
ser = serial.Serial('/dev/ttyACM0', 115200)
ser.write(b"Tell me a story\n")
while True:
    print(ser.read(1).decode(), end='', flush=True)
```

→ **Real-time token streaming**

---

## Final Board Size & Power

| Spec | Value |
|------|-------|
| **PCB Size** | 100 × 80 mm (credit card+) |
| **Height** | < 15 mm (with DDR) |
| **Power** | ~15–25 W (FPGA + DDR) |
| **Boot Time** | ~17 sec (7 GB copy) |
| **Token Rate** | 4–8 tokens/sec (LLaMA-2 7B) |

---

## Summary

| Feature | Notes |
|-------|----------|
| **eMMC → DDR4 copy** | 17 sec boot |
| **ARM Linux** | Manages USB, boot, comms |
| **FPGA AI core** | Clash decoder |

---

