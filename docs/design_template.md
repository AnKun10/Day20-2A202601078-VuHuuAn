# Design Template

## Problem

Hệ thống nhận một câu hỏi nghiên cứu dạng dài (ví dụ: "Research GraphRAG state-of-the-art
and write a 500-word summary"), tự tìm nguồn trên web, phân tích/đối chiếu các nguồn, và
viết câu trả lời cuối cùng có trích dẫn. Đầu ra phải kèm trace từng bước và benchmark so
với single-agent baseline.

## Why multi-agent?

Single-agent (một prompt làm tất cả) gặp 3 vấn đề với task này:

1. **Context loãng**: một prompt vừa search vừa phân tích vừa viết khiến model không tập
   trung vào từng kỹ năng; kết quả thiếu citation vì không có nguồn thật.
2. **Không debug được**: khi output sai, không biết sai ở khâu thu thập, phân tích hay viết.
3. **Không có grounding**: baseline không gọi search nên trả lời từ trí nhớ, dễ hallucinate
   và citation coverage = 0.

Tách vai trò cho phép mỗi bước có prompt chuyên biệt, state trung gian debug được, và
citation bám vào nguồn thật.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Rule-based routing: nhìn state, chọn bước tiếp theo hoặc dừng | Toàn bộ `ResearchState` | `route_history` mới + `iteration` | Lặp vô hạn nếu thiếu stop condition → chặn bằng `max_iterations` |
| Researcher | Search (Tavily/mock) + tóm tắt nguồn thành research notes, giữ index [n] | `request.query`, `max_sources` | `sources`, `research_notes` | Search API down → ghi `errors`, supervisor retry 1 lần rồi fallback bỏ qua |
| Analyst | Trích claim chính, đối chiếu nguồn, cờ evidence yếu | `research_notes`, `sources` | `analysis_notes` | Chỉ paraphrase thay vì phân tích → prompt cấm viết final answer |
| Writer | Tổng hợp thành câu trả lời có citation [n] + mục Sources | `analysis_notes`, `sources`, `audience` | `final_answer` | Bịa nguồn → prompt cấm invent; Critic đo coverage sau đó |
| Critic (bonus) | Đo citation coverage, liệt kê nguồn chưa được cite | `final_answer`, `sources` | `agent_results` (findings) | Không chặn luồng — chỉ chạy đúng 1 lần rồi done |

## Shared state

`ResearchState` (Pydantic) — single source of truth truyền qua mọi node:

- `request` — query + max_sources + audience: input gốc, mọi agent đều cần.
- `iteration`, `route_history` — guardrail chống lặp + bằng chứng routing để debug/nộp bài.
- `sources` — danh sách `SourceDocument` có index ổn định để citation [n] trỏ về.
- `research_notes` / `analysis_notes` / `final_answer` — output từng stage; field nào còn
  thiếu chính là tín hiệu routing cho Supervisor.
- `agent_results` — token/cost/duration từng bước, để benchmark tổng hợp cost.
- `trace` — span cục bộ (bổ sung cho LangSmith).
- `errors` — worker fail ghi vào đây; Supervisor đọc để quyết định retry/fallback.

## Routing policy

```text
supervisor ──(sources trống)──────────> researcher ──> supervisor
supervisor ──(chưa có analysis_notes)─> analyst ────> supervisor
supervisor ──(đủ dữ liệu)─────────────> writer ─────> supervisor
supervisor ──(có final_answer, chưa critic)─> critic ─> supervisor
supervisor ──(critic đã chạy / hết budget)──> DONE
```

Rule-based (không dùng LLM để route): deterministic, rẻ, unit-test được. Agent nào fail
2 lần thì bị bỏ qua (fallback) thay vì retry mãi.

## Guardrails

- Max iterations: `MAX_ITERATIONS=6` (Settings), check trước mọi quyết định route; kèm
  `recursion_limit` của LangGraph làm lưới an toàn thứ hai.
- Timeout: `TIMEOUT_SECONDS=60` áp vào OpenAI client và httpx (Tavily).
- Retry: tenacity trong `LLMClient.complete` — 3 lần, exponential backoff, chỉ với lỗi
  transient (rate limit, timeout, connection, 5xx).
- Fallback: worker fail ghi vào `state.errors`; fail 2 lần → Supervisor bỏ qua agent đó
  và đi tiếp với dữ liệu đang có.
- Validation: mọi input/output qua Pydantic (`ResearchQuery` min_length, `BenchmarkMetrics`
  ràng buộc 0-10 / 0-1); query sai bị chặn ngay ở CLI.

## Benchmark plan

- **Queries** (3, chạy qua cả 2 hệ): GraphRAG state-of-the-art; LoRA vs full fine-tuning;
  approaches to evaluating multi-agent LLM systems.
- **Metrics**: latency (wall-clock), cost (token usage × pricing), quality (LLM-as-judge
  0-10), citation coverage (nguồn được cite / tổng nguồn), failure rate.
- **Expected outcome**: multi-agent chậm hơn và đắt hơn (~4-5× token) nhưng có citation
  coverage cao (baseline = 0 vì không có nguồn) và trace đầy đủ; quality ngang hoặc nhỉnh
  hơn với câu hỏi cần thông tin mới. Kết quả thật xem `reports/benchmark_report.md`.
