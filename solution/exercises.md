# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Khi temperature tăng từ 0.0 đến 1.8, phản hồi từ mô hình trở nên đa dạng và sáng tạo hơn nhưng giảm dần tính chính xác. Ở mức 0.0 và 0.7, câu trả lời rất mạch lạc, chuẩn xác và tập trung vào các sự thật thực tế. Từ mức 1.2 trở đi, văn phong bắt đầu lan man, lặp từ, và đến 1.8 thì phản hồi kém mạch lạc rõ rệt với cấu trúc câu rườm rà hoặc từ ngữ thiếu tự nhiên.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Tôi sẽ đặt temperature = 0.0 (hoặc tối đa 0.2) cho trợ lý soạn thảo hợp đồng pháp lý nhằm đảm bảo tính chính xác tuyệt đối, nhất quán và không phát sinh thông tin ngẫu nhiên. Ngược lại, tôi sẽ đặt temperature = 0.7 - 0.9 cho trợ lý viết slogan quảng cáo để khuyến khích mô hình đưa ra những ý tưởng mới lạ, độc đáo và giàu tính sáng tạo ngôn từ.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Tổng lượng token đầu ra: 20.000 users × 2 lượt × 500 tokens = 20.000.000 tokens (20.000k tokens). Chi phí mỗi ngày với GPT-4o là 20.000 × $0.010 = $200/ngày ($6.000/tháng); với GPT-4o-mini là 20.000 × $0.0006 = $12/ngày ($360/tháng). Model lớn xứng đáng chi phí cho các bài toán phân tích pháp lý, y tế hoặc giải quyết các vấn đề logic phức tạp. Model nhỏ là lựa chọn đúng cho tác vụ đơn giản như phân loại văn bản, phân tích cảm xúc hoặc tóm tắt ngắn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Phản hồi từ nhà thơ sử dụng văn phong giàu hình ảnh, nhẹ nhàng, ví máy học như cây xanh tự lớn theo mùa mà không dùng thuật ngữ kỹ thuật. Phản hồi từ kỹ sư phần mềm senior có giọng văn khúc chiết, chuẩn xác, giải thích thuật toán, dữ liệu huấn luyện và đưa ví dụ đoạn mã Python cụ thể. Qua đó, system prompt giúp điều khiển hiệu quả vai trò persona, giọng văn (tone), độ sâu kỹ thuật, ngôn ngữ và định dạng trình bày của mô hình.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Số token thực tế đếm bằng tiktoken (khoảng 250 - 300 tokens) cao hơn đáng kể so với ước lượng thô 150 / 0.75 = 200 tokens (chênh lệch khoảng 25% - 40%). Dùng ước lượng thô sẽ dự toán THIẾU ngân sách vì các bộ tokenizer chuẩn (cl100k_base / o200k_base) được tối ưu cho tiếng Anh, dẫn đến từ tiếng Việt có dấu bị chia tách thành nhiều subword tokens nhỏ hơn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản (a) hưởng lợi nhiều nhất từ streaming vì phản hồi hiển thị từng token giảm thời gian chờ đợi cảm nhận (perceived latency), mang lại cảm giác tương tác phản hồi tức thì. Trợ lý giọng nói (b) cũng tận dụng streaming để bắt đầu tổng hợp tiếng nói (TTS) từ các câu đầu tiên. Ngược lại, pipeline dịch tài liệu chạy ngầm ban đêm (c) không cần streaming vì là tác vụ batch tự động, nhận response trọn gói giúp đơn giản hóa logic lưu trữ và xử lý.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Exponential backoff giúp giảm áp lực dồn dập lên server bằng cách tăng gấp đôi thời gian chờ sau mỗi lần thử lại, tạo khoảng nghỉ cho API phục hồi. Kỹ thuật "jitter" giải quyết vấn đề "thảm họa đồng bộ" (thùng tháo nước), tránh việc hàng ngàn client gọi lại chính xác cùng một thời điểm sau khi kết thúc khoảng thời gian backoff cố định.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt: "Bạn là trợ giảng thân thiện của khóa AI Practical Competency, giải thích khái niệm ngắn gọn bằng tiếng Việt."
> 1. Nếu xóa "ngắn gọn bằng tiếng Việt": Trợ lý sẽ trả lời dài dòng và có thể dùng tiếng Anh khi gặp thuật ngữ chuyên ngành.
> 2. Nếu xóa "trợ giảng thân thiện của khóa AI Practical Competency": Trợ lý sẽ mất đi giọng văn gần gũi, hỗ trợ học viên và trở thành một chatbot tư vấn chung chung.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Ở lượt 1, người dùng nói: "Tôi đang làm dự án e-commerce bằng Python". Sau 4 lượt trao đổi về các câu hỏi khác, ở lượt 6 người dùng hỏi: "Làm sao kết nối database cho dự án của tôi?". Do history chỉ giữ 4 lượt gần nhất (lượt 2-5), ngữ cảnh "dự án e-commerce bằng Python" ở lượt 1 đã bị xóa, dẫn đến trợ lý hướng dẫn chung chung hoặc gợi ý ngôn ngữ khác như Java/Node.js. Cách khắc phục: Sử dụng LLM chạy ngầm để tóm tắt các lượt hội thoại đã qua (Summarization) và duy trì bản tóm tắt này trong system prompt, hoặc lưu thông tin ngữ cảnh chính vào biến state dài hạn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
