# tools/elab.tcl -- batch RTL elaboration for Clash-generated LLaMa2 decoder
#
# Usage examples:
#   vivado -mode batch -source tools/elab.tcl -tclargs "temp/clash-verilog-model-110m/full/LLaMa2.Decoder.Decoder.topEntity" "temp/vivado-elab-110m"
#   vivado -mode batch -source tools/elab.tcl -tclargs rtl_dir out_dir
#
# If no arguments are provided, default values are used.
#
# ----------------------------
# Configuration & Parameters
# ----------------------------

set_param general.maxThreads 1

# Project settings
set PART "xczu7ev-ffvc1156-3-e"

# Default values (relative to repository root)
set DEFAULT_RTL_REL_DIR "temp/clash-verilog-model-110m/full/LLaMa2.Decoder.Decoder.topEntity"
set DEFAULT_OUT_REL_DIR "temp/vivado-elab-110m"

# Accept command-line parameters for directories only
if { $argc >= 2 } {
    set RTL_REL_DIR [lindex $argv 0]
    set OUT_REL_DIR [lindex $argv 1]
    puts "INFO: Using provided paths:"
    puts "      RTL_REL_DIR = $RTL_REL_DIR"
    puts "      OUT_REL_DIR = $OUT_REL_DIR"
} else {
    set RTL_REL_DIR $DEFAULT_RTL_REL_DIR
    set OUT_REL_DIR $DEFAULT_OUT_REL_DIR
    puts "INFO: No command-line arguments provided. Using default paths."
}

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
puts "PART       = $PART"

# Check RTL directory exists
if {![file isdirectory $RTL_DIR]} {
    puts "ERROR: RTL directory not found:"
    puts "  $RTL_DIR"
    exit 1
}

# Create output directory
file mkdir $OUT_DIR

# Read all Verilog files
set vfiles [lsort [glob -nocomplain -directory $RTL_DIR *.v]]

if {[llength $vfiles] == 0} {
    puts "ERROR: No .v files found in $RTL_DIR"
    exit 1
}

puts "Reading [llength $vfiles] Verilog files..."
foreach f $vfiles {
    puts "  [file tail $f]"
}

read_verilog $vfiles

report_compile_order -file [file join $OUT_DIR compile_order_pre_elab.rpt]

# ===================================================================
# Determine top module from clash-manifest.json (authoritative)
# ===================================================================

# Clash writes clash-manifest.json in every output directory.
# The top_component.name field is the exact Verilog module name for the top.

set TOP ""
set manifest_file [file join $RTL_DIR clash-manifest.json]

if {[file exists $manifest_file]} {
    set fh [open $manifest_file r]
    set manifest [read $fh]
    close $fh
    # Find the "top_component" key, then search for "name" within that substring.
    # Split into two steps to avoid \{ / \} inside {} which confuses Tcl's brace counter.
    set tc_idx [string first {"top_component"} $manifest]
    if {$tc_idx >= 0} {
        set tc_substr [string range $manifest $tc_idx end]
        if {[regexp {"name"\s*:\s*"([^"]+)"} $tc_substr -> TOP]} {
            puts "INFO: Top module from clash-manifest.json = $TOP"
        } else {
            puts "WARNING: Could not parse name from top_component in clash-manifest.json"
        }
    } else {
        puts "WARNING: top_component key not found in clash-manifest.json"
    }
} else {
    puts "WARNING: clash-manifest.json not found in $RTL_DIR"
}

# Fallback: the module that is never instantiated by any other module
if {$TOP eq ""} {
    set all_modules {}
    foreach f $vfiles {
        set fh [open $f r]
        set content [read $fh]
        close $fh
        if {[regexp {module\s+(\w+)} $content -> modname]} {
            lappend all_modules $modname
        }
    }
    set instantiated {}
    foreach f $vfiles {
        set fh [open $f r]
        set content [read $fh]
        close $fh
        foreach {_ inst_type} [regexp -all -inline {\m(\w+)\s+(?:#\s*\([^)]*\)\s*)?\w+\s*\(} $content] {
            lappend instantiated $inst_type
        }
    }
    foreach m $all_modules {
        if {[lsearch -exact $instantiated $m] < 0} {
            set TOP $m
            break
        }
    }
    if {$TOP ne ""} { puts "INFO: Top module from instantiation analysis = $TOP" }
}

# Last resort: shortest .v filename (Clash tops are always short/clean names)
if {$TOP eq ""} {
    set TOP [file rootname [lindex [lsort -command {apply { {a b} { expr {[string length $a] - [string length $b]} } }} \
                                        [glob -nocomplain -tails -directory $RTL_DIR *.v]] 0]]
    puts "INFO: Top module from shortest filename fallback = $TOP"
}

if {$TOP eq ""} {
    puts "ERROR: Could not determine top module"
    exit 1
}

# ===================================================================
# Elaborate with explicit top
# ===================================================================

puts "Elaborating design..."
synth_design -top $TOP -part $PART -rtl

# ===================================================================
# Reports
# ===================================================================

write_verilog -mode funcsim -force -file [file join $OUT_DIR post_elab_netlist.v]
write_verilog -mode synth_stub -force -file [file join $OUT_DIR post_elab_stub.v]

report_drc -file [file join $OUT_DIR rtl_drc.rpt]
report_compile_order -file [file join $OUT_DIR compile_order_post_elab.rpt]

# Simple hierarchy list (no proc needed)
set fh [open [file join $OUT_DIR hierarchy.rpt] w]
puts $fh "=== Hierarchical Cells in Elaborated Design ==="
puts $fh "Top module: $TOP"
puts $fh ""
puts $fh [join [get_cells -hierarchical -filter {IS_PRIMITIVE == 0}] "\n"]
close $fh

puts "=== RTL elaboration completed successfully ==="
puts "Top module used : $TOP"
puts "Functional netlist : $OUT_DIR/post_elab_netlist.v"
puts "Synth stub         : $OUT_DIR/post_elab_stub.v"
puts "Hierarchy report   : $OUT_DIR/hierarchy.rpt"
puts "All files written to: $OUT_DIR"
