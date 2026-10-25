# Khử tiếng vọng âm học bằng LMS / NLMS

Khử tiếng vọng âm học (Acoustic Echo Cancellation, AEC) bằng bộ lọc thích nghi **LMS**, viết bằng NumPy/SciPy. **NLMS** là biến thể chuẩn hoá của LMS, dùng để khắc phục việc LMS phụ thuộc vào mức tín hiệu vào.

## Bài toán

Đầu gần (người dùng laptop không đeo tai nghe) phát giọng của đầu xa `x(n)` ra loa. Micro thu lại tiếng vọng `r(n)` sau khi âm thanh đi qua phòng, cộng với giọng của chính người ở đầu gần `s(n)` và nhiễu `v(n)`:

```
r(n) = (h * x)(n)                 tiếng vọng qua đường truyền h (đáp ứng xung phòng)
d(n) = r(n) + s(n) + v(n)         tín hiệu micro
y(n) = wᵀ(n) x(n)                 ước lượng tiếng vọng của bộ lọc thích nghi
e(n) = d(n) − y(n)                đầu ra của AEC, gửi sang đầu xa
```

Bộ lọc `w` học dần để tiến tới `h`, và `e(n)` chỉ còn lại giọng đầu gần.

## Cấu trúc thư mục

```
aec-lms/
├── aec/                    # Thuật toán
│   ├── lms.py              # lms(x, d, M, mu, normalize=False, delta, adapt, hist_every)
│   │                       #   -> e, y_hat, w_hist   (normalize=True ⇒ NLMS)
│   └── dtd.py              # geigel(x, d, L, T, hangover): bộ phát hiện nói chồng Geigel -> mặt nạ bool
├── sim/                    # Mô phỏng tín hiệu
│   ├── rir.py              # make_rir(): đáp ứng xung phòng tổng hợp (nhiễu trắng × đường bao suy giảm theo RT60)
│   ├── scenario.py         # build_scenario(): tạo r, d từ x, h, s với ERL và ENR cho trước
│   ├── conversation.py     # Ghép một cuộc gọi hai người từ hai file ghi âm theo kịch bản CALL_SCRIPT
│   └── audio.py            # Đọc/ghi WAV (scipy.io.wavfile), chuyển mono, resample
├── eval/                   # Đánh giá
│   ├── metrics.py          # ERLE(t), misalignment, mức xác lập, thời gian hội tụ
│   └── plots.py            # Kiểu hình vẽ chung (LMS luôn màu xanh, NLMS luôn màu cam)
├── experiments/
│   ├── demo.py             # Demo đầu-cuối: chạy LMS và NLMS trên giọng nói thật, xuất WAV và hình (--dtd để bật DTD)
│   ├── exp1_stepsize.py    # Thí nghiệm 1: quét µ của LMS, đánh đổi tốc độ và sai số xác lập
│   ├── exp2_lms_vs_nlms.py # Thí nghiệm 2a: quét mức đầu xa; 2b: nhiễu trắng và tiếng nói
│   └── exp3_doubletalk.py  # Thí nghiệm 3: nói chồng, không DTD / Geigel / DTD lý tưởng
├── tests/                  # Kiểm thử pytest
│   ├── test_convergence.py # LMS/NLMS phải hội tụ (misalignment < −30 dB) trên trường hợp dễ
│   ├── test_rir.py         # Năng lượng, tính tái lập theo seed, RT60 đo lại khớp RT60 yêu cầu
│   ├── test_conversation.py# Cắt câu theo khoảng lặng, xếp câu lên dòng thời gian
│   └── test_dtd.py         # Geigel ít báo nhầm khi chỉ đầu xa nói, phát hiện được đầu gần, hangover đúng
├── data/                   # Dữ liệu ghi âm (đầu vào)
│   ├── far_end.wav         # Giọng đầu xa cho chế độ 1 (bản gốc: far_end.m4a)
│   ├── near_end.wav        # Giọng đầu gần cho chế độ 1 (bản gốc: near_end.m4a)
│   └── call/               # Dữ liệu cho chế độ cuộc gọi
│       ├── script.vi.md    # Kịch bản lời thoại và hướng dẫn thu âm
│       ├── far_end.wav     # Toàn bộ lời của An (đầu xa), bản gốc an_voice.m4a
│       └── near_end.wav    # Toàn bộ lời của Bình (đầu gần), bản gốc binh_voice.m4a
├── results/                # Kết quả sinh ra (WAV + PNG)
│   ├── rir.png             # Hình RIR, sinh bởi sim/rir.py
│   ├── demo/               # Kết quả chế độ 1
│   ├── call/               # Kết quả chế độ cuộc gọi
│   ├── call_geigel/        # Chế độ cuộc gọi với --dtd geigel (tương tự: call_ideal/)
│   └── exp1/ exp2/ exp3/   # Hình (và WAV của thí nghiệm 3) cho báo cáo
├── report/                 # (đang trống) báo cáo
├── requirements.txt
└── pytest.ini              # Đặt pythonpath = . để test import được aec/ và sim/
```

## Cài đặt

Cần **Python ≥ 3.9**, đã thử với Python 3.10.

```bash
git clone https://github.com/dnhd2903/aec-lms.git
cd aec-lms
python3 -m pip install -r requirements.txt     # numpy, scipy, matplotlib, pytest
```

Không bắt buộc dùng môi trường ảo. Tuy nhiên trên Ubuntu ≥ 23.04 hoặc Debian ≥ 12, lệnh `pip install` vào Python hệ thống sẽ bị từ chối với lỗi `externally-managed-environment`. Khi đó hãy tạo môi trường ảo rồi cài lại:

```bash
sudo apt install python3-venv                  # chỉ cần nếu lệnh dưới báo thiếu venv/ensurepip
python3 -m venv .venv
source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Chỉ cần cài thêm `ffmpeg` nếu muốn chuyển file ghi âm `.m4a` sang `.wav` (xem bên dưới).

> **Lưu ý:** mọi lệnh đều phải chạy từ thư mục gốc của repo (`aec-lms/`). Code được chạy dưới dạng module (`python3 -m ...`) và dùng đường dẫn tương đối `data/`, `results/`.

## Chạy code

### 1. Kiểm thử

```bash
python3 -m pytest -q
```

Kết quả mong đợi là `20 passed`.

### 2. Xem đáp ứng xung phòng (RIR)

```bash
python3 -m sim.rir                            # vẽ RIR → results/rir.png
python3 -m sim.rir --rt60 0.2 --L 512         # thay đổi RT60 và độ dài
python3 -m sim.rir --wav data/far_end.wav     # thêm: results/rir_demo_dry.wav và rir_demo_echo.wav để nghe thử tiếng vọng
```

### 3. Demo khử tiếng vọng

**Chế độ 1: hai bản ghi, một đoạn đầu gần** (mặc định dùng `data/far_end.wav` và `data/near_end.wav`):

```bash
python3 -m experiments.demo                   # đầu gần bắt đầu nói ở giây thứ 12
python3 -m experiments.demo --near-start 16   # đổi thời điểm đầu gần bắt đầu nói
python3 -m experiments.demo --no-near         # chỉ có đầu xa: đầu ra phải gần như im lặng
```

**Chế độ 2: cuộc gọi hai người theo kịch bản** (`data/call/script.vi.md`), có xen lời và nói chồng (double-talk):

```bash
python3 -m experiments.demo --call data/call
```

**Bộ phát hiện nói chồng (DTD)** dùng được với cả hai chế độ, qua tùy chọn `--dtd`:

```bash
python3 -m experiments.demo --call data/call                # mặc định --dtd none: thích nghi liên tục
python3 -m experiments.demo --call data/call --dtd geigel   # đóng băng thích nghi khi Geigel phát hiện nói chồng
python3 -m experiments.demo --call data/call --dtd ideal    # đóng băng đúng lúc đầu gần nói (chỉ làm được trong mô phỏng)
```

| `--dtd` | Bộ lọc dừng cập nhật khi | Kết quả lưu ở |
|---|---|---|
| `none` (mặc định) | không bao giờ | `results/demo/`, `results/call/` |
| `geigel` | `\|d(n)\| > 0.5·max\|x\|` trên 256 mẫu gần nhất, giữ thêm 30 ms ([aec/dtd.py](aec/dtd.py)) | `results/demo_geigel/`, `results/call_geigel/` |
| `ideal` | đầu gần đang nói (dùng thông tin thật, làm cận trên) | `results/demo_ideal/`, `results/call_ideal/` |

Mỗi lựa chọn có thư mục riêng, nên có thể nghe so sánh các bản với nhau. Ở chế độ cuộc gọi, Geigel giúp NLMS hồi phục gần như hoàn toàn sau mỗi đoạn nói chồng: ERLE của đoạn chỉ đầu xa nói ngay sau lần nói chồng đầu tiên tăng từ 13 dB lên 35 dB.

Kết quả được lưu trong thư mục ở bảng trên:

| File | Nội dung |
|---|---|
| `far_end.wav` | `x(n)`: giọng đầu xa phát ra loa |
| `far_end_hears_no_aec.wav` | `d(n)`: đầu xa nghe thấy gì khi **không có** AEC |
| `far_end_hears_lms.wav` | `e(n)`: đầu xa nghe thấy gì khi dùng AEC LMS |
| `far_end_hears_nlms.wav` | `e(n)`: đầu xa nghe thấy gì khi dùng AEC NLMS |
| `demo.png` | Dạng sóng, ERLE(t) và misalignment(t) của LMS và NLMS |

Mọi file WAV dùng chung một hệ số tỉ lệ, nên độ to tương đối giữa chúng đúng với thực tế. Giọng đầu xa còn nghe thấy trong các file `far_end_hears_*.wav` chính là tiếng vọng chưa khử hết. Chương trình cũng in ra bảng ERLE theo từng đoạn (chỉ đầu xa nói, nói chồng) và misalignment cuối cùng.

**Các tham số chính** (xem đủ bằng `python3 -m experiments.demo --help`):

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--erl` | 8 dB | Echo return loss, tức tiếng vọng nhỏ hơn `x` bao nhiêu dB |
| `--enr` | 35 dB | Tỉ số tiếng vọng trên nhiễu |
| `--ser` | 0 dB | Tỉ số giọng đầu gần trên tiếng vọng (đo khi đang nói) |
| `--rt60` | 0.1 s | Thời gian vang của phòng |
| `--lh` | 256 | Độ dài đường truyền tiếng vọng `L_h` (mẫu, 32 ms ở 8 kHz) |
| `--taps` | `= --lh` | Số hệ số `M` của bộ lọc thích nghi |
| `--mu-nlms` | 0.5 | Bước NLMS chuẩn hoá (0 < µ̃ < 2) |
| `--dtd` | `none` | Bộ phát hiện nói chồng: `none`, `geigel` hoặc `ideal` |
| `--seed` | 0 | Seed cho RIR và nhiễu |
| `--out` | `results/demo` hoặc `results/call` (thêm `_geigel` / `_ideal` khi dùng `--dtd`) | Thư mục lưu kết quả |

Bước LMS được chọn tự động là `µ = 0.1 / (M·σ²ₓ)` theo công suất đầu xa, nằm an toàn trong miền ổn định `0 < µ < 2/(M·σ²ₓ)`. Mọi tín hiệu được đưa về 8 kHz trước khi xử lý.

> Với `--dtd none`, trong lúc hai bên cùng nói, bộ lọc bị `s(n)` làm lệch: NLMS ở chế độ cuộc gọi kết thúc với misalignment +10.6 dB, tức là đã phân kỳ. Đó là hành vi dự kiến, và là lý do cần DTD. Thí nghiệm 3 phân tích kỹ trường hợp này.

### 4. Các thí nghiệm cho báo cáo

Mỗi script in ra một bảng số liệu và lưu hình vào `results/expN/`. Mỗi script chạy trong khoảng 1–12 giây.

```bash
python3 -m experiments.exp1_stepsize      # results/exp1/stepsize.png
python3 -m experiments.exp2_lms_vs_nlms   # results/exp2/gain.png, results/exp2/speech.png
python3 -m experiments.exp3_doubletalk    # results/exp3/doubletalk.png + 7 file WAV
```

| Thí nghiệm | Câu hỏi | Kết quả chính |
|---|---|---|
| 1. Quét µ (nhiễu trắng) | µ lớn hay nhỏ thì tốt hơn? | µ lớn hội tụ nhanh hơn nhưng sai số xác lập cao hơn; đường đo khớp lý thuyết trong khoảng 0.5 dB |
| 2a. Quét mức đầu xa | Cùng một µ có an toàn ở mọi âm lượng không? | LMS phân kỳ khi đầu xa to hơn khoảng 13 dB, đúng như lý thuyết; NLMS không đổi |
| 2b. Nhiễu trắng và tiếng nói | Vì sao LMS chậm trên tiếng nói? | Độ trải trị riêng χ ≈ 44 dB, LMS chỉ đạt misalignment −8 dB sau 30 s (nhiễu trắng: −48 dB) |
| 3. Nói chồng | DTD giúp được bao nhiêu? | ERLE khi nói chồng (NLMS): không DTD −1 dB, Geigel 0 dB, DTD lý tưởng 35 dB; Geigel ngăn được phân kỳ (+13 → −6.5 dB) |

## Dùng dữ liệu ghi âm của bạn

1. Ghi âm theo hướng dẫn trong [`data/call/script.vi.md`](data/call/script.vi.md): mỗi vai một file, mỗi câu cách nhau khoảng 2 giây im lặng.
2. Chuyển sang WAV mono:

   ```bash
   ffmpeg -i data/call/an_voice.m4a   -ac 1 -ar 16000 data/call/far_end.wav
   ffmpeg -i data/call/binh_voice.m4a -ac 1 -ar 16000 data/call/near_end.wav
   ```

   Với chế độ 1 thì làm tương tự cho `data/far_end.m4a` và `data/near_end.m4a`. Tần số lấy mẫu nào cũng được vì code sẽ tự resample về 8 kHz.
3. Chạy `python3 -m experiments.demo --call data/call`.

Nếu số câu cắt được không khớp kịch bản (4 câu của An, 5 câu của Bình), hoặc một câu dài quá và chồng lên câu sau của cùng người nói, chương trình sẽ báo lỗi và chỉ ra câu bị sai.

## Dùng trực tiếp trong Python

```python
from aec.lms import lms
from sim.rir import make_rir
from sim.scenario import build_scenario
import numpy as np

fs, M = 8000, 256
x = np.random.default_rng(0).standard_normal(3 * fs)            # đầu xa: nhiễu trắng
h = make_rir(M, rt60=0.1, fs=fs, seed=0)
x, r, s, v, d, h = build_scenario(x, h, erl_db=8.0, enr_db=35.0)

e, y_hat, w_hist = lms(x, d, M, mu=0.5, normalize=True)          # NLMS
```
