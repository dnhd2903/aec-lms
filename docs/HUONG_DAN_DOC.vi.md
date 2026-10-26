# Hướng dẫn đọc code — 9 giờ, 5 buổi

Mục tiêu: sau 5 buổi, bạn giải thích được từng hình trong báo cáo và trả lời được câu hỏi "dòng code nào tạo ra kết quả này?".

**Nguyên tắc:** chỉ đọc kỹ phần **thuật toán** và **thí nghiệm**. Phần vẽ hình (khối `# --- figure ---`, các lệnh `ax.plot`, `style`) chỉ cần lướt. Trước mỗi buổi, hãy chạy script tương ứng để có số liệu trước mặt khi đọc.

| Buổi | Thời lượng | Nội dung | File |
|---|---|---|---|
| 1 | 2 giờ | Lõi: LMS/NLMS và testbench | `aec/lms.py`, `sim/rir.py`, `sim/scenario.py`, `tests/test_convergence.py` |
| 2 | 1.5 giờ | Chỉ số đánh giá + Thí nghiệm 1 | `eval/metrics.py`, `experiments/exp1_stepsize.py` |
| 3 | 2 giờ | Thí nghiệm 2a, 2b | `experiments/exp2_lms_vs_nlms.py` |
| 4 | 2 giờ | DTD + Thí nghiệm 3 | `aec/dtd.py`, `experiments/exp3_doubletalk.py` |
| 5 | 1.5 giờ | Demo (có `--dtd`) + ghi chú cho báo cáo | `experiments/demo.py` (lướt), ghi lại số liệu |

Mọi đường dẫn file và lệnh trong tài liệu này đều tính từ thư mục repo `aec_lms/`.

**Có thể bỏ qua:** `sim/audio.py` (đọc/ghi WAV), phần bên trong `sim/conversation.py` (cắt câu theo khoảng lặng), `eval/plots.py`. Chúng không chứa gì về LMS.

---

## Buổi 1 — Lõi (2 giờ)

**`aec/lms.py` (45 phút) — file quan trọng nhất.** Vòng lặp `for n in range(N)` chỉ có 6 dòng, và mỗi dòng ứng với một công thức trên slide:

| Dòng code | Công thức |
|---|---|
| `x_vec = x_pad[n : n + M][::-1]` | `x(n) = [x(n), x(n−1), …, x(n−M+1)]ᵀ` — **đảo ngược thời gian** |
| `y_hat[n] = w @ x_vec` | `y(n) = wᵀ(n) x(n)` |
| `e[n] = d[n] - y_hat[n]` | `e(n) = d(n) − y(n)` |
| `w += mu * e[n] * x_vec` | LMS: `w(n+1) = w(n) + µ e(n) x(n)` |
| `w += (mu / (delta + x_vec @ x_vec)) * e[n] * x_vec` | NLMS: chia cho `δ + ‖x(n)‖²` |
| `if adapt is None or adapt[n]:` | DTD cắm vào đây: `adapt[n] = False` ⇒ vẫn lọc, nhưng không cập nhật `w` |

Bài tập: với `M = 3` và `n = 0`, hãy tự viết ra `x_pad` và `x_vec`, để thấy vì sao cần đệm `M − 1` số 0 ở đầu.

**`sim/rir.py` (15 phút).** Chỉ cần đọc hàm `make_rir`, gồm 4 dòng: nhiễu trắng × đường bao `10^(−3n/(fs·RT60))`, rồi chuẩn hoá `‖h‖ = 1`.

**`sim/scenario.py` (25 phút).** Hàm `build_scenario`: `r = h * x`, chỉnh độ lợi để đạt ERL, cộng nhiễu `v` theo ENR, rồi `d = r + s + v`. Hãy đọc chú thích về `_NOISE_STREAM`. Đó là một lỗi tinh vi: nếu `v` sinh từ cùng seed với `x` thì `v` tương quan với `x`, và bộ lọc sẽ "khử" luôn cả nhiễu.

**`tests/test_convergence.py` (20 phút).** Đọc tên từng test và câu `assert`. Đáng chú ý là `test_tap_order_identifies_a_pure_delay`: nó bắt được lỗi số 1 trong kế hoạch (thứ tự vector đầu vào).

**Chạy (15 phút):** `python3 -m pytest -q`, rồi `python3 -m experiments.demo`, và nghe 4 file trong `results/demo/`.

✅ Tự kiểm tra: vì sao `e(n)` là *đầu ra* của AEC chứ không phải phần dư cần bỏ đi? Vì sao LMS và NLMS dùng chung một hàm?

---

## Buổi 2 — Chỉ số + Thí nghiệm 1 (1.5 giờ)

**`eval/metrics.py` (20 phút).** Có 3 hàm chính:
- `sliding_erle_db`: ERLE theo cửa sổ trượt, dùng **tiếng vọng thật** `r`. Hàm `np.convolve(·, ones(win))` chính là tổng trượt.
- `misalignment_db`: `20·log10(‖h − w‖/‖h‖)` tính cho mọi ảnh chụp `w` trong `w_hist`.
- `convergence_index`: điểm đầu tiên mà đường cong vào trong khoảng 3 dB quanh mức xác lập.

**`experiments/exp1_stepsize.py` (40 phút).** Đọc kỹ docstring (phần lý thuyết) và hàm `lms_theory_curve`. Đây là công thức MSD của LMS với đầu vào trắng:

```
ρ = 1 − µσ²(2 − µσ²(M+2))          hệ số suy giảm mỗi mẫu
MSD(∞) = µMσ²ᵥ / (2 − µσ²(M+2))    sai số xác lập
```

**Kết quả** (`results/exp1/stepsize.png`):

| α = µMσ² | t hội tụ đo / lý thuyết | misalignment xác lập đo / lý thuyết |
|---|---|---|
| 0.05 | 3.90 / 3.85 s | −51.4 / −50.9 dB |
| 0.1 | 1.90 / 1.86 s | −48.1 / −47.8 dB |
| 0.5 | 0.33 / 0.40 s | −39.8 / −39.8 dB |
| 1.0 | 0.21 / 0.26 s | −35.0 / −35.0 dB |

**Cho báo cáo:**
- µ tăng 20 lần thì hội tụ nhanh hơn khoảng 19 lần, nhưng sàn sai số cao hơn 16 dB. Đó là đánh đổi tốc độ / misadjustment (kế hoạch §3 điểm 5).
- *Chỗ chưa khớp:* với α lớn, hội tụ đo được nhanh hơn lý thuyết khoảng 20%. Lý thuyết dựa trên giả thiết độc lập (independence assumption), và giả thiết này kém chính xác hơn khi µ lớn.

---

## Buổi 3 — Thí nghiệm 2 (2 giờ)

**2a: quét mức đầu xa** (hàm `exp2a`, `results/exp2/gain.png`). µ được cố định cho mức tín hiệu ×1, sau đó phát đầu xa to hơn `g` lần. Với LMS, bước hiệu dụng là `α = g²·α₀`.

- LMS: càng to thì hội tụ càng nhanh nhưng sàn càng cao. Ở **+13 dB** (α = 2.0) thì misalignment chỉ còn −1.2 dB, và từ **+14 dB** trở lên LMS **phân kỳ**. Lý thuyết dự đoán ngưỡng 13.0 dB, khớp chính xác.
- NLMS: **−39.8 dB ở mọi mức**, ba đường cong trùng nhau hoàn toàn.
- Một chi tiết hay: NLMS với µ̃ = 0.5 cho đúng kết quả của LMS với α = 0.5 (đều −39.8 dB). Trên nhiễu trắng, NLMS chính là LMS với µ được tự chỉnh theo công suất.

**2b: nhiễu trắng và tiếng nói cùng công suất** (hàm `exp2b`, `results/exp2/speech.png`). Đọc hàm `eigenvalues_db`: nó ước lượng ma trận `R_xx` (Toeplitz M×M) từ hàm tự tương quan, rồi lấy trị riêng.

| Đầu vào | χ = λmax/λmin | LMS | NLMS |
|---|---|---|---|
| Nhiễu trắng | 0.7 dB (≈ 1) | −47.8 dB | −39.8 dB |
| Tiếng nói | 43.7 dB (≈ 23 000) | −8.3 dB | −25.0 dB |

**Cho báo cáo (phần thảo luận):**
1. LMS trên tiếng nói chậm hơn rất nhiều. Đây là hệ quả của độ trải trị riêng (kế hoạch §3 điểm 3).
2. *Chỗ khác kế hoạch:* NLMS tốt hơn hẳn LMS trên tiếng nói (−25 so với −8 dB). Lý do: tiếng nói **không dừng**. µ của LMS được chọn theo công suất trung bình dài hạn, nên ở những đoạn nói nhỏ thì bước thực tế quá bé. NLMS chuẩn hoá theo công suất *tức thời*, tức là khắc phục điểm 2 *bên trong* một tín hiệu. Nhưng NLMS vẫn chịu ảnh hưởng của χ: trên tiếng nói nó đạt −25 dB, kém xa −40 dB trên nhiễu trắng.
3. Hình (c): ERLE trên tiếng nói khoảng 30 dB dù misalignment chỉ khoảng −8 dB. Bộ lọc chỉ cần khớp `h` ở những dải tần mà tiếng nói có năng lượng, còn misalignment thì tính trên mọi dải tần. Vì vậy cần cả hai chỉ số.

---

## Buổi 4 — Double-talk (2 giờ)

**`aec/dtd.py` (30 phút).** Quy tắc Geigel `|d(n)| > T·max|x(n…n−L+1)|`. Có 3 dòng cần hiểu: `sliding_window_view(...).max(axis=1)` (đỉnh trượt của `|x|`), phép so sánh, và `np.convolve(detected, ones(hangover+1))`. Dòng cuối kéo dài cờ thêm `hangover` mẫu sau mỗi lần phát hiện. Đọc thêm `tests/test_dtd.py`.

**`experiments/exp3_doubletalk.py` (60 phút).** Có 3 cấu hình DTD: không có, Geigel, lý tưởng. DTD lý tưởng đóng băng đúng đoạn 12–22 s; nó chỉ làm được trong mô phỏng, nên dùng làm cận trên. Mỗi cấu hình chạy với cả LMS và NLMS.

| | ERLE: chỉ đầu xa | **khi nói chồng** | sau đó | misalignment cuối đoạn nói chồng |
|---|---|---|---|---|
| LMS, không DTD | 19.6 | 4.2 | 27.2 | −5.9 dB |
| NLMS, không DTD | 23.5 | −1.2 | 10.4 | **+13.4 dB** (phân kỳ) |
| LMS, Geigel | 19.6 | 5.9 | 22.2 | −6.1 dB |
| NLMS, Geigel | 23.2 | 0.3 | 17.7 | −6.5 dB |
| LMS, lý tưởng | 19.6 | 29.1 | 32.7 | −6.6 dB |
| NLMS, lý tưởng | 23.5 | **35.3** | 38.4 | −18.7 dB |

Geigel (T = 0.5, L = 256, hangover 30 ms): báo nhầm 9.3% thời gian khi chỉ đầu xa nói, phát hiện được 54% thời gian đầu gần nói.

**Nghe (30 phút):** 7 file trong `results/exp3/`. So sánh `far_end_hears_nlms_no_dtd.wav`, `..._geigel.wav` và `..._ideal.wav`.

**Cho báo cáo:**
1. Không có DTD thì NLMS phân kỳ (misalignment lên +13 dB). Đây là bằng chứng rằng `s(n)` bị bộ lọc coi là "sai số".
2. Geigel ngăn được phân kỳ (+13 → −6.5 dB) và giúp hồi phục nhanh hơn sau đoạn nói chồng (10.4 → 17.7 dB). Nhưng nó **còn xa DTD lý tưởng** (0.3 so với 35.3 dB).
3. Nhìn hình (b) quanh giây 12.9: misalignment của NLMS + Geigel nhảy từ −18 lên −5 dB **ngay lúc đầu gần bắt đầu nói**, trước khi Geigel kịp phát hiện. Phát hiện chậm ở đầu câu và bỏ sót đoạn nói nhỏ là hai điểm yếu của Geigel ở SER = 0 dB.
4. LMS ít bị nói chồng phá hơn NLMS (4.2 so với −1.2 dB), vì bước của LMS nhỏ hơn. Hội tụ chậm thì cũng bị lệch chậm, nên đây là một đánh đổi khác.
5. Hướng phát triển: DTD dựa trên tương quan chéo chuẩn hoá (NCC) thay vì so sánh biên độ.

---

## Buổi 5 — Tổng kết (1.5 giờ)

- **`experiments/demo.py` (30 phút):** chỉ cần đọc `main()` và `build_two_recordings`. Phần vẽ hình giống các thí nghiệm. Phần DTD chỉ có hai dòng trong `main()`:
  ```python
  frozen = {"none": ..., "geigel": geigel(x, d, L=M), "ideal": near_mask}[args.dtd]
  lms(..., adapt=~frozen, ...)          # đúng cách cắm DTD giống Thí nghiệm 3
  ```
  Chạy cả ba lựa chọn để có âm thanh cuộc gọi cho buổi thuyết trình. Mỗi lựa chọn lưu ra một thư mục riêng:
  ```bash
  python3 -m experiments.demo --call data/call                # → results/call/
  python3 -m experiments.demo --call data/call --dtd geigel   # → results/call_geigel/
  python3 -m experiments.demo --call data/call --dtd ideal    # → results/call_ideal/
  ```
  Hãy nghe `far_end_hears_nlms.wav` trong ba thư mục, nhất là đoạn **21–24 s** ngay sau lần nói chồng đầu tiên. Không có DTD thì tiếng vọng quay lại rõ; có Geigel thì gần như biến mất.

  Kết quả với NLMS ở chế độ cuộc gọi:

  | | Không DTD | Geigel | Lý tưởng |
  |---|---|---|---|
  | ERLE khi nói chồng (19.5–20.7 s) | −2.0 dB | 8.8 dB | 36.2 dB |
  | ERLE chỉ đầu xa nói ngay sau đó (20.7–24.5 s) | 13.3 dB | **34.5 dB** | 37.0 dB |
  | Misalignment cuối cùng | +10.6 dB (phân kỳ) | −13.2 dB | −27.4 dB |

  **Cho báo cáo:** Geigel hiệu quả hơn hẳn trong cuộc gọi so với Thí nghiệm 3. Ở Thí nghiệm 3, ERLE sau đoạn nói chồng là 17.7 dB; trong cuộc gọi là 34.5 dB. Lý do: các đoạn nói chồng trong cuộc gọi ngắn (1–2 s, giống hội thoại thật), nên lần bỏ sót nào cũng chỉ gây thiệt hại ngắn và bộ lọc kịp hồi phục. Còn Thí nghiệm 3 có 10 s nói chồng liên tục, là trường hợp xấu nhất. Đây là một ý tốt cho phần thảo luận: hiệu quả của DTD phụ thuộc vào kịch bản.
- **Ghi chú cho báo cáo (60 phút):** với mỗi hình trong `results/exp*/`, viết 2–3 câu: hình cho thấy gì, khớp hoặc không khớp với lý thuyết nào. Các mục "Cho báo cáo" ở trên đã là bản nháp sẵn.

### Bảng tham số dùng chung (báo cáo §3)

| Tham số | Giá trị |
|---|---|
| fs | 8 kHz |
| M = L_h | 256 (32 ms) |
| RT60 | 0.1 s |
| ERL / ENR | 8 dB / 35 dB |
| µ LMS | `0.1/(M·σ²ₓ)` (trừ Thí nghiệm 1) |
| µ̃ NLMS, δ | 0.5, `1e-2·M·σ²ₓ` (tiếng nói) |
| Geigel | T = 0.5, L = 256, hangover = 240 mẫu |
| Seed | 0 |
