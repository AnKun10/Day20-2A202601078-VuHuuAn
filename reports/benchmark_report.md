# Benchmark Report

## Aggregate

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline (mean) | 8.79 | 0.00046 | 8.3 |  | 0% | mean over 3 queries |
| multi-agent (mean) | 25.32 | 0.00206 | 8.0 | 100% | 0% | mean over 3 queries |

## Per-query detail

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline :: Research GraphRAG state-of-the-art and w... | 8.15 | 0.00046 | 7.0 |  | 0% | routes: single |
| baseline :: Compare LoRA and full fine-tuning for ad... | 9.40 | 0.00051 | 9.0 |  | 0% | routes: single |
| baseline :: What are the main approaches to evaluati... | 8.83 | 0.00042 | 9.0 |  | 0% | routes: single |
| multi-agent :: Research GraphRAG state-of-the-art and w... | 23.43 | 0.00194 | 8.0 | 100% | 0% | routes: researcher -> analyst -> writer -> critic -> done |
| multi-agent :: Compare LoRA and full fine-tuning for ad... | 31.69 | 0.00236 | 8.0 | 100% | 0% | routes: researcher -> analyst -> writer -> critic -> done |
| multi-agent :: What are the main approaches to evaluati... | 20.85 | 0.00188 | 8.0 | 100% | 0% | routes: researcher -> analyst -> writer -> critic -> done |

## Analysis

**Trade-off chính.** Multi-agent chậm hơn ~2.9× (25.3s vs 8.8s) và đắt hơn ~4.5×
($0.00206 vs $0.00046) vì mỗi bước là một LLM call riêng và context (sources + notes)
được lặp lại qua từng prompt. Đổi lại, multi-agent đạt **citation coverage 100%** trên cả
3 query (baseline: 0 nguồn — trả lời hoàn toàn từ trí nhớ của model), và mọi câu trả lời
đều bám vào nguồn Tavily thật, xem được từng bước trên LangSmith.

**Quality score cần đọc kỹ.** LLM judge chấm baseline nhỉnh hơn (8.3 vs 8.0). Điều này
minh họa đúng bài học của lab: với câu hỏi kiến thức tổng quát, một prompt tốt đã đủ và
judge ưu ái văn phong trôi chảy hơn là tính grounded. Multi-agent chỉ thực sự thắng khi
cần thông tin mới/có thể kiểm chứng — thứ mà citation coverage (100% vs 0%) phản ánh còn
quality score thì không.

**Failure mode gặp phải và cách fix.**

1. *Vòng lặp Supervisor ↔ Researcher khi search fail*: nếu Tavily lỗi, Supervisor thấy
   `sources` vẫn trống và cứ route lại Researcher đến khi hết iteration. Fix: worker fail
   được ghi vào `state.errors`; Supervisor bỏ qua agent đã fail 2 lần (fallback) và
   `max_iterations` + `recursion_limit` của LangGraph là hai lưới an toàn chặn vòng lặp
   (có unit test `test_workflow_stops_on_iteration_budget_even_if_workers_fail`).
2. *Đếm nhầm iteration budget*: `record_route("done")` cũng tăng `iteration`, nên số
   iteration cuối = số route + 1 — lúc đầu test assert `iteration <= max_iterations` và
   fail. Fix: hiểu rõ ngữ nghĩa counter và assert `<= max_iterations + 1`; đây là lý do
   `route_history` cần nằm trong state để debug được.
3. *Judge trả về text thay vì số*: LLM judge đôi khi trả "Score: 8/10". Fix: parse bằng
   regex số đầu tiên + clamp về [0, 10], trả `None` (thay vì crash) nếu không parse được.

## How metrics are computed

- **Latency**: wall-clock seconds per query (`time.perf_counter`).
- **Cost**: estimated from token usage recorded by `LLMClient` per agent step.
- **Quality**: LLM-as-judge rubric 0-10 (relevance, grounding, clarity).
- **Citation coverage**: sources referenced in the final answer / total sources.
- **Failure rate**: failed runs / total runs (exceptions caught by the benchmark).
