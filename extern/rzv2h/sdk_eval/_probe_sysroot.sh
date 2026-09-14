#!/usr/bin/env bash
# Probe the RZ/V2H board rootfs (via the cross-SDK aarch64 sysroot) for the
# GStreamer + Python stack our pipeline needs. Run inside drpai-tvm-v2h.
# Target rootfs sysroot (NOT the x86_64-pokysdk-linux cross-compiler dir).
SR=$(ls -d /opt/*/*/sysroots/*-poky-linux 2>/dev/null | grep -v pokysdk | head -1)
[ -d "$SR" ] || SR=$(ls -d /opt/*/sysroots/*-poky-linux 2>/dev/null | grep -v pokysdk | head -1)
echo "sysroot = $SR"
echo "--- python3 ---"; ls -d "$SR"/usr/lib/python3* 2>/dev/null | head -1
echo "--- gstreamer core ---"; ls "$SR"/usr/lib/libgstreamer-1.0.so.* 2>/dev/null
grep -h "Version" "$SR"/usr/lib/pkgconfig/gstreamer-1.0.pc 2>/dev/null
echo "--- gst-python loader (libgstpython) ---"; find "$SR" -name 'libgstpython*' 2>/dev/null | head
echo "--- GstAnalytics (lib + typelib) ---"
find "$SR" -iname '*gstanalytics*' 2>/dev/null | head
ls "$SR"/usr/lib/girepository-1.0/ 2>/dev/null | grep -iE 'Analytics|GstApp|GstBase|^Gst-' | head
echo "--- python modules on target: gi / numpy / cairo / cv2 ---"
for m in gi numpy cairo cv2; do
  hit=$(find "$SR" -maxdepth 7 -path '*python3*' -iname "${m}" 2>/dev/null | head -1)
  echo "$m: ${hit:-MISSING}"
done
echo "--- tvm / mera python runtime on target? ---"
find "$SR" -iname '*tvm*' -o -iname '*mera*' 2>/dev/null | grep -i python | head
echo "--- gstreamer plugins present (count) ---"
ls "$SR"/usr/lib/gstreamer-1.0/*.so 2>/dev/null | wc -l
