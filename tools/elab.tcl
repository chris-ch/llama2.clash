# tools/elab.tcl -- batch RTL elaboration for Clash-generated llama2 decoder
#
# Can be launched from anywhere:
#   vivado -mode batch -source tools/elab.tcl
#
# Assumes this repository layout:
#   repo_root/
#     tools/elab.tcl
#     temp/clash-verilog-model-110m/full/LLaMa2.Decoder.Decoder.topEntity/*.v

# ----------------------------
# Configuration
# ----------------------------

# Keep memory pressure lower
set_param general.maxThreads 1

# Project settings
set TOP decoder
set PART xczu7ev-ffvc1156-3-e

# Relative to repo root
set RTL_REL_DIR "temp/clash-verilog-model-110m/full/LLaMa2.Decoder.Decoder.topEntity"
set OUT_REL_DIR "temp/vivado-elab-110m"

# ----------------------------
# Resolve paths
# ----------------------------

# Directory containing this script (tools/)
set SCRIPT_DIR [file normalize [file dirname [info script]]]

# Repo root = parent of tools/
set REPO_ROOT [file normalize [file join $SCRIPT_DIR ..]]

# Absolute paths
set RTL_DIR [file normalize [file join $REPO_ROOT $RTL_REL_DIR]]
set OUT_DIR [file normalize [file join $REPO_ROOT $OUT_REL_DIR]]

puts "=== Vivado RTL elaboration starting ==="
puts "SCRIPT_DIR = $SCRIPT_DIR"
puts "REPO_ROOT  = $REPO_ROOT"
puts "RTL_DIR    = $RTL_DIR"
puts "OUT_DIR    = $OUT_DIR"
puts "TOP        = $TOP"
puts "PART       = $PART"

# Check RTL directory exists
if {![file isdirectory $RTL_DIR]} {
    puts "ERROR: RTL directory not found:"
    puts "  $RTL_DIR"
    exit 1
}

# Create output directory
file mkdir $OUT_DIR

# Read all Verilog files in the RTL directory
set vfiles [lsort [glob -nocomplain -directory $RTL_DIR *.v]]

if {[llength $vfiles] == 0} {
    puts "ERROR: No .v files found in:"
    puts "  $RTL_DIR"
    exit 1
}

puts "Reading [llength $vfiles] Verilog files..."
foreach f $vfiles {
    puts "  [file tail $f]"
}

# ----------------------------
# Read + elaborate
# ----------------------------

read_verilog $vfiles

# Show compile order
report_compile_order -file [file join $OUT_DIR compile_order_pre_elab.rpt]

# Elaborate RTL only (no full synthesis)
# NOTE: If this still gets OOM-killed, next try:
# synth_design -top $TOP -part $PART -rtl -rtl_skip_mlo
synth_design -top $TOP -part $PART -rtl

# ----------------------------
# Reports
# ----------------------------

report_hierarchy     -file [file join $OUT_DIR hierarchy.rpt]
report_compile_order -file [file join $OUT_DIR compile_order_post_elab.rpt]
report_drc           -file [file join $OUT_DIR rtl_drc.rpt]

puts "=== RTL elaboration completed successfully ==="
puts "Reports written to:"
puts "  $OUT_DIR"
