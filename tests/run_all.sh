#!/bin/bash
# Run ALL test suites in parallel (16 cores)
cd /root/workspace/Image-Deskew
.venv/bin/python -m pytest tests/test_deskew.py -q > /tmp/t1.log 2>&1 &
P1=$!
.venv/bin/python -m pytest tests/test_inpaint.py -q > /tmp/t2.log 2>&1 &
P2=$!
.venv/bin/python -m pytest tests/test_watermark.py -q > /tmp/t3.log 2>&1 &
P3=$!
.venv/bin/python -m pytest tests/test_preprocessors.py -q > /tmp/t4.log 2>&1 &
P4=$!
.venv/bin/python -m pytest tests/test_pipeline.py -q > /tmp/t5.log 2>&1 &
P5=$!
wait $P1 $P2 $P3 $P4 $P5
echo "=== deskew ==="; tail -1 /tmp/t1.log
echo "=== inpaint ==="; tail -1 /tmp/t2.log
echo "=== watermark ==="; tail -1 /tmp/t3.log
echo "=== preprocessors ==="; tail -1 /tmp/t4.log
echo "=== pipeline ==="; tail -1 /tmp/t5.log
