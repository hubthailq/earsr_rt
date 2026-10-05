DO DO TRE TREN JETSON (T3)
==========================

Thu muc nay tu chua: chi can chep sang Jetson, khong can PyTorch, khong can ma nguon cua project.

1. Tren may huan luyen (da lam san neu thu muc onnx/ co file):
       python scripts/export_for_device.py
   Lenh nay tao onnx/*.onnx (opset 13) va onnx/index.json.

2. Chep ca thu muc deploy_jetson sang Jetson (scp hoac USB).

3. Tren Jetson, khoa che do nguon TRUOC khi do:
       sudo nvpmodel -m 0
       sudo jetson_clocks

4. Chay:
       python3 bench_trtexec.py --note "Jetson Nano 4GB, JetPack 4.6.1, nvpmodel -m 0, jetson_clocks"

5. Gui lai: latency_trt.csv va thu muc logs/.

Doc ket qua:
- status = ok            : co so do; cot median_ms va p_ms (phan vi 95).
- status = build_failed  : TensorRT khong dung duoc engine. Voi cac file spanvar_replicate va
                           spanvar_reflect, day chinh la cau tra loi cho cau hoi "thiet bi co ho tro
                           kieu dem nay khong". Cot error ghi ly do.
- Kieu dem nao cham hon spanvar_zero qua 10% thi bi loai khoi T6 (iv).

LUU Y: bench_trtexec.py chua duoc chay thu tren thiet bi that (luc viet chua co Jetson).
Neu trtexec tren may ban in dong Latency theo dang khac, cot so do se trong nhung log tho van
nam trong logs/; gui lai log do la du.

Dien thoai Android: chua co script. Se viet khi biet doi may (TFLite GPU delegate hoac NNAPI).
