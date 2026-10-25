# Kịch bản cuộc gọi — An (đầu xa) gọi cho Bình (đầu gần)

**Tình huống:** An gọi Zoom cho Bình để bàn bài tập lớn. Bình ngồi ở nhà, dùng loa và micro của laptop, **không đeo tai nghe**. AEC chạy trên laptop của Bình.

Thứ tự và chỗ chồng lời được định nghĩa trong `sim/conversation.py` (`CALL_SCRIPT`). Khi thu, **không cần căn thời gian**: code tự xếp các câu lên dòng thời gian.

---

## File 1 — `far_end.wav`: lời của An (4 câu)

Nói như đang gọi điện cho bạn, tự nhiên, không đọc như bản tin.

| # | Lời thoại | Độ dài gợi ý |
|---|---|---|
| A1 | Alo, Bình à? An đây. Mình vừa xem lại kế hoạch bài tập lớn môn xử lý tín hiệu số, có mấy việc muốn bàn với cậu một chút. Giờ cậu có rảnh không? | 7–9 s |
| A2 | Phần mô phỏng tiếng vọng thì mình làm xong rồi, chạy cũng ổn. Còn phần thuật toán LMS thì mình định nhờ cậu viết, vì tuần sau mình phải đi thực tập nên không có nhiều thời gian. Với lại phần báo cáo thì mình nghĩ là chúng ta nên chia ra… | 10–12 s |
| A3 | À, phần đó thì hai đứa cùng làm, mỗi người một nửa. | 3–4 s |
| A4 | Ok, thế nhé. Chào cậu! | 2 s |

- **A1 phải dài** (≥ 7 s): đây là lúc bộ lọc học đường truyền tiếng vọng trước khi Bình lên tiếng.
- **A2 phải dài** (≥ 9 s): Bình sẽ "ừ ừ" ở giữa câu và ngắt lời ở 2 giây cuối. Câu cuối của A2 nói hơi kéo dài, như đang bị cắt ngang.

## File 2 — `near_end.wav`: lời của Bình (5 câu)

| # | Lời thoại | Độ dài gợi ý | Rơi vào lúc |
|---|---|---|---|
| B1 | Ừ, mình đang rảnh, cậu nói đi. | 2–3 s | sau A1 |
| B2 | Ừ… ừ. | ~1 s | **giữa** A2 (ừ hử) |
| B3 | Khoan đã, thế còn phần thí nghiệm thì ai làm? | 2–3 s | **ngắt lời** cuối A2 |
| B4 | Được, thế cuối tuần mình gặp nhau ở thư viện để làm cho xong nhé. | 3–4 s | sau A3 |
| B5 | Ừ, chào cậu, hẹn gặp lại! | ~2 s | **cùng lúc** với A4 |

- B2 phải **ngắn** (khoảng 1 s): nếu dài quá nó sẽ chồng lên B3 và code sẽ báo lỗi.

---

## Cách thu (một người đóng cả hai vai)

1. Bấm ghi âm, **chờ 2 giây** rồi mới nói câu đầu.
2. Nói hết một câu → **im lặng khoảng 2 giây** → câu tiếp theo. Không nói gì, không gõ phím trong lúc nghỉ.
3. Hết câu cuối, **chờ 2 giây** rồi mới bấm dừng.
4. Thu File 1 (An) và File 2 (Bình) riêng. Nếu được, đổi giọng một chút giữa hai vai (ví dụ An nói nhanh và cao hơn, Bình chậm và trầm hơn) để khi nghe dễ phân biệt.
5. Điện thoại cách miệng 20–30 cm, cả hai file cùng một cách thu.

Lỡ nói sai một câu: thu lại cả file đó (code đếm số câu theo khoảng lặng, nói thừa hay thiếu câu sẽ bị báo lỗi).

## Chuyển sang WAV và chạy

```bash
cd ~/Documents/Master-HUST/Advanced_DSP/aec_lms
ffmpeg -i data/call/an.m4a   -ac 1 -ar 16000 data/call/far_end.wav
ffmpeg -i data/call/binh.m4a -ac 1 -ar 16000 data/call/near_end.wav
python3 -m experiments.demo --call data/call
```

Không cần `-ss` để cắt đầu: tiếng bấm nút ngắn sẽ được code tự bỏ qua, miễn là bạn chờ 2 giây trước khi nói.
