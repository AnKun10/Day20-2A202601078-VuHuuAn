# Exit Ticket

## 1. Case nào nên dùng multi-agent? Vì sao?

Nên dùng khi task **phân rã tự nhiên thành các vai trò có kỹ năng khác nhau** và cần
**grounding + traceability**:

- Research/report cần nguồn thật và citation: Researcher gọi search tool nên câu trả lời
  bám vào nguồn; baseline một-prompt trả lời từ trí nhớ nên citation coverage = 0%.
- Pipeline dài mà một prompt làm tất cả sẽ loãng context (search → phân tích → viết):
  mỗi agent có system prompt chuyên biệt, chất lượng từng khâu cao hơn.
- Hệ cần debug/audit được: state trung gian (`research_notes`, `analysis_notes`,
  `route_history`) + trace LangSmith cho biết chính xác sai ở khâu nào — điều một
  completion đơn không cho.
- Cần guardrail độc lập từng khâu: retry riêng cho search, quality gate (Critic) trước khi
  trả kết quả.

Bằng chứng từ benchmark của lab: multi-agent đạt citation coverage cao (baseline: 0%) và
route/trace giải thích được từng bước — đổi lấy ~2-3× latency và ~4-5× token cost.

## 2. Case nào KHÔNG nên dùng multi-agent? Vì sao?

Không nên khi **một prompt tốt đã đủ**:

- Câu hỏi kiến thức tổng quát, không cần nguồn mới (giải thích khái niệm, viết đoạn văn,
  refactor code ngắn): benchmark cho thấy baseline nhanh hơn 2-3×, rẻ hơn 4-5×, quality
  LLM-judge gần như ngang nhau.
- Ứng dụng nhạy latency (chatbot tương tác): mỗi hop supervisor→worker cộng thêm một vòng
  LLM call.
- Ngân sách token chặt: overhead đến từ việc mỗi worker nhận lại gần như toàn bộ context
  (sources + notes lặp lại qua từng prompt).
- Chưa có evaluation: thêm agent = thêm điểm hỏng (search down, routing sai, vòng lặp).
  Nếu không benchmark được cải thiện cụ thể, độ phức tạp đó là chi phí thuần — đúng tinh
  thần "đừng thêm agent nếu không có lý do rõ ràng".

Quy tắc rút ra: bắt đầu bằng single-agent làm baseline đo được; chỉ tách vai trò khi metric
chỉ ra điểm nghẽn mà việc tách giải quyết được (thiếu grounding, context loãng, cần audit).
