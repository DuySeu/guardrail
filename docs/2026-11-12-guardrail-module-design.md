# Thiết kế module guardrail

Ngày: 2026-11-12. Trạng thái: chờ duyệt.

## Mục tiêu

Tạo một folder `core` tự chứa, copy được vào các dự án production để kiểm tra input của user và output của LLM, tuỳ biến bằng file YAML. Kèm theo một package evaluate tách riêng, không copy vào production, dùng để đo độ chính xác của một cấu hình guardrail trên dataset có nhãn.

## Phạm vi

Trong phạm vi bản đầu:

- Hai điểm kiểm tra: input (prompt của user) và output (câu trả lời của LLM).
- Ba hành động: `allow`, `redact`, `block`.
- Ba nhóm cơ chế: rule-based (regex, keyword, max_length), embedding similarity, LLM judge.
- Cấu hình bằng YAML, guard tự viết đăng ký qua registry.
- Evaluate trên JSONL tự gán nhãn, kèm adapter chuyển dataset public về cùng định dạng.

Ngoài phạm vi bản đầu: kiểm tra tool call và RAG context, chạy guard song song theo tier, streaming output, sinh dữ liệu tấn công bằng LLM, classifier model chuyên dụng (Llama Guard, Prompt Guard), cache response khi evaluate.

## Bố cục thư mục

```text
core/                     # core logic của guardrail, folder duy nhất cần copy vào production
  __init__.py             # public API
  types.py                # Stage, Action, GuardContext, GuardResult, Decision
  registry.py             # Guard protocol, register_guard, build_guard
  config.py               # schema pydantic, đọc YAML
  pipeline.py             # runner tuần tự, timeout, on_error
  providers.py            # Protocol ChatClient, Embedder, bản OpenRouter
  constants.py            # hằng số OpenRouter và ModelId, sinh bởi scripts/sync_models.sh
  guards/
    rule.py               # regex, keyword, max_length
    embedding.py          # similarity với tập ví dụ
    llm_judge.py          # LLM phân loại, trả JSON
eval/                     # benchmark độ chính xác, không copy
  dataset.py              # đọc và validate JSONL
  runner.py               # chạy guardrail trên dataset
  metrics.py              # hàm thuần: records -> metrics
  report.py               # ghi kết quả ra file
  adapters/               # chuyển dataset public về JSONL chuẩn
main.py                   # CLI duy nhất: check, eval, compare
config.yaml               # cấu hình guardrail, nằm ở root
requirements.txt          # dependency của core, eval và test
scripts/sync_models.sh    # sync model free của OpenRouter vào core/constants.py
datasets/
tests/
```

Quy tắc phụ thuộc: `core` không import gì ngoài chính nó và dependency của nó. `eval` import `core`, không có chiều ngược lại. Model id là chuỗi trong YAML.

Bên trong `core` chỉ dùng import tương đối (`from .types import ...`), để dự án production đổi tên folder khi copy (ví dụ thành `guardrail/`) mà không sửa code. Việc này cần vì nhiều dự án đã có package tên `core` của riêng mình.

Không có `pyproject.toml`. Lệnh chạy từ root của repo (`python main.py ...`, `python -m pytest`) để `core` và `eval` import được. Python 3.12 (venv dùng chung `../.venv` của Lab).

`requirements.txt` ở root liệt kê mọi dependency, chia nhóm bằng comment: core (`pydantic`, `pyyaml`, `httpx`), eval (`python-dotenv`), test (`pytest`). Adapter tải dataset bằng `httpx`, không thêm dependency. Dự án production chỉ cần nhóm core. Phiên bản được ghim khi triển khai, theo bản thực tế cài được.

`main.py` là điểm vào duy nhất cho việc chạy thủ công, chỉ chứa phần parse tham số và gọi hàm của `core` và `eval`. Package `eval` không có `__main__.py`.

Thứ tự triển khai: core trước, eval sau. Eval chỉ phụ thuộc vào public API của core (`load_guardrail`, `Guardrail`, `Decision`, `GuardResult`).

## Core

### Kiểu dữ liệu

| Kiểu | Trường |
| --- | --- |
| `Stage` | `input`, `output` |
| `Action` | `allow`, `redact`, `block` |
| `GuardContext` | `stage`, `user_input` (tuỳ chọn, câu hỏi gốc của user khi kiểm tra output) |
| `GuardResult` | `guard` (tên), `action`, `category`, `score` (0 đến 1, None khi lỗi), `reason`, `text` (text sau guard này), `latency_ms`, `error` (None nếu không lỗi) |
| `Decision` | `action`, `text` (text cuối cùng), `results` (danh sách `GuardResult` theo thứ tự chạy), `blocked_by` (tên guard, None nếu không block) |

`score` là điểm vi phạm: càng cao càng vi phạm. Guard chạy thành công luôn trả `score` kể cả khi `allow`. Guard rule-based trả 1.0 khi khớp, 0.0 khi không. Quy ước này là điều kiện để evaluate quét ngưỡng mà không gọi lại model. `GuardResult` do pipeline tạo khi guard lỗi có `score` và `category` là None, `text` giữ nguyên text đầu vào của guard đó.

`category` của guard rule-based và `embedding` lấy từ tham số `category` của guard, mặc định bằng `name`. `llm_judge` lấy `category` từ câu trả lời của LLM.

`Decision.action` là `block` nếu có guard block, ngược lại là `redact` nếu có ít nhất một guard redact, ngược lại là `allow`. Khi block, `Decision.text` là text tại thời điểm bị chặn (đã qua các lần redact trước đó), app không được dùng text này để gọi LLM.

### Interface guard và registry

```python
class Guard(Protocol):
    name: str
    async def check(self, text: str, context: GuardContext) -> GuardResult: ...
```

Mỗi loại guard là một class được đăng ký bằng `@register_guard("<type>")` và khai báo ba thứ:

- `Params`: schema pydantic cho tham số riêng của guard.
- `requires`: tập client mà guard cần, gồm `"chat"`, `"embedder"`, hoặc rỗng. Guard có `requires` khác rỗng được coi là guard gọi mạng.
- `__init__(self, params: Params, deps: GuardDeps)`: `GuardDeps` chứa `chat_client`, `embedder` (None nếu không có) và `base_dir` (thư mục chứa file YAML).

```python
class ChatClient(Protocol):
    async def complete(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0) -> str: ...

class Embedder(Protocol):
    async def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...
```

`build_guard(type, raw_params, deps)` tra registry, validate `raw_params` bằng `Params` của class, rồi khởi tạo guard. Dự án dùng guard tự viết bằng cách đăng ký class của mình trước khi gọi `load_guardrail`, rồi khai báo `type: <type>` trong YAML.

### Public API

```python
guardrail = load_guardrail("config.yaml")
decision = await guardrail.check_input(user_text)
if decision.action is Action.BLOCK:
    return refusal
answer = await llm(decision.text)
decision = await guardrail.check_output(answer, user_input=user_text)
```

`load_guardrail(path, chat_client=None, embedder=None)` nhận client tự truyền vào để dùng provider khác hoặc client giả trong test. Loại client nào không được truyền mà có guard cần đến (theo `requires`) thì dùng bản OpenRouter cho loại đó.

`await guardrail.warmup()` chuẩn bị trước các tài nguyên tính lười (embedding của tập ví dụ). App nên gọi một lần lúc khởi động. Không gọi thì việc này diễn ra ở request đầu tiên.

`guardrail.guards(stage)` trả danh sách guard của stage theo thứ tự chạy, mỗi guard có `name` và `requires`. `guardrail.only(stage, name)` trả một `Guardrail` mới chỉ chứa đúng guard đó, dùng lại chính instance guard nên không tính lại embedding. Hai hàm này phục vụ evaluate ở chế độ `--isolate`, không cần file YAML tạm.

Stage không có trong YAML hoặc có danh sách rỗng thì `check_input` / `check_output` trả `allow` với text không đổi.

### Cấu hình

Thông tin provider nằm trong biến môi trường, không nằm trong YAML: `OPENROUTER_API_KEY` (bắt buộc khi có guard cần một loại client không được truyền vào `load_guardrail`) và `OPENROUTER_BASE_URL` (tuỳ chọn). Core chỉ đọc `os.environ`, không tự load file `.env`, vì app production có cách nạp env riêng. Model id, ngưỡng, prompt nằm theo từng guard trong YAML.

```yaml
input:
  - type: max_length
    limit: 8000
  - type: regex
    name: pii
    presets: [email, phone_vn, credit_card]
    action: redact
  - type: keyword
    name: denylist
    file: denylist.txt
    action: block
  - type: embedding
    name: known_jailbreaks
    mode: deny
    examples: jailbreaks.txt
    model: <embedding model id>
    threshold: 0.82
  - type: llm_judge
    name: injection
    model: <chat model id>
    categories: [prompt_injection, harmful]
    threshold: 0.7
    timeout_s: 5
    on_error: block
output:
  - type: regex
    name: secrets
    presets: [api_key, email]
    action: redact
```

Tham số chung của mọi guard: `type` (bắt buộc), `name` (mặc định bằng `type`, phải duy nhất trong một stage), `timeout_s` (mặc định 10 với guard gọi mạng, mặc định không giới hạn với guard có `requires` rỗng), `on_error` (`block` hoặc `allow`, mặc định `block`).

Đường dẫn file trong YAML (`file`, `examples`, `prompt`) tính tương đối so với thư mục chứa file YAML.

`load_guardrail` báo lỗi `GuardrailConfigError` ngay lúc load khi: YAML sai schema, `type` chưa đăng ký, tham số guard sai, file tham chiếu không tồn tại, tên guard trùng trong một stage, hoặc thiếu `OPENROUTER_API_KEY` trong khi có ít nhất một loại client được guard nào đó cần mà không được truyền vào. Ví dụ: truyền `chat_client`, config có guard `embedding`, không có key thì báo lỗi.

### Constants và sync model

`core/constants.py` chứa các hằng số của OpenRouter: base URL mặc định, tên biến môi trường (`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`), và `class ModelId(StrEnum)` liệt kê model free.

Nội dung `ModelId` do `scripts/sync_models.sh` sinh ra, ghi vào ba khối đánh dấu trong class: `OPENROUTER` (chat có hỗ trợ tools), `OPENROUTER-EMBED` (embedding), `OPENROUTER-RERANK` (rerank). Script dừng với lỗi nếu thiếu marker của bất kỳ khối nào, nên `constants.py` phải có sẵn cả ba cặp marker, kể cả khối rerank mà guardrail không dùng. Không sửa tay phần nằm giữa hai marker.

YAML ghi model id dạng chuỗi và core không kiểm tra chuỗi đó có nằm trong `ModelId` hay không, vì danh sách model free thay đổi theo thời gian và dự án có thể dùng model trả phí. `ModelId` dùng để tra cứu khi viết config và khi cần tham chiếu model trong code.

### Các guard có sẵn

| Type | Tham số riêng | Hành vi |
| --- | --- | --- |
| `max_length` | `limit` (số ký tự) | Block khi text dài hơn `limit`. |
| `regex` | `presets`, `patterns` (tên -> regex), `action` (`block` hoặc `redact`, mặc định `block`), `replacement` (mặc định `[<TÊN_PATTERN>]`) | Khớp bất kỳ pattern nào thì block, hoặc thay phần khớp bằng `replacement` rồi trả `redact`. Preset bản đầu: `email`, `phone_vn`, `credit_card`, `api_key`. |
| `keyword` | `words` hoặc `file` (mỗi dòng một từ hoặc cụm từ), `case_sensitive` (mặc định false), `action` (mặc định `block`), `replacement` | Giống `regex` nhưng khớp nguyên văn theo ranh giới từ (Unicode), không khớp chuỗi con nằm giữa một từ. |
| `embedding` | `mode` (`deny` hoặc `allow`), `examples` (file, mỗi dòng một ví dụ), `model`, `threshold` | Tính cosine lớn nhất giữa text và tập ví dụ, giá trị âm tính là 0 để nằm trong khoảng 0 đến 1. `deny`: `score` bằng cosine lớn nhất, dùng cho tập câu tấn công đã biết. `allow`: `score` bằng 1 trừ cosine lớn nhất, dùng cho tập chủ đề được phép. Cả hai mode block khi `score >= threshold`. |
| `llm_judge` | `model`, `categories`, `threshold`, `prompt` (file template tuỳ chọn) | Gọi LLM với temperature 0, yêu cầu trả JSON `{"score": 0..1, "category": "...", "reason": "..."}`. Block khi `score >= threshold`. Text cần kiểm tra được đặt trong vùng phân cách rõ ràng và prompt nêu đó là dữ liệu, không phải chỉ dẫn. Ở stage output, prompt kèm `user_input` nếu có. `category` trả về phải thuộc `categories` hoặc là null, giá trị khác tính là lỗi. |

`threshold` là tham số bắt buộc của `embedding` và `llm_judge`, không có giá trị mặc định, vì ngưỡng phù hợp phụ thuộc vào model và được chọn bằng threshold sweep.

Embedding của tập ví dụ được tính một lần, ở `warmup()` hoặc ở lần gọi đầu, có `asyncio.Lock` để các request đồng thời không tính trùng, và giữ trong bộ nhớ. Nếu tính ở lần gọi đầu thì thời gian này tính vào `timeout_s` của request đó. Cosine viết bằng Python thuần, không dùng numpy.

### Luồng chạy

1. `check_input` hoặc `check_output` lấy danh sách guard của stage tương ứng theo đúng thứ tự khai báo trong YAML.
2. Từng guard chạy lần lượt trên text hiện tại, có giới hạn `timeout_s` nếu guard đó có giá trị này.
3. Kết quả `redact` thay text hiện tại, guard sau nhận text đã che.
4. Kết quả `block` dừng pipeline, các guard sau không chạy.
5. Trả `Decision` gồm mọi `GuardResult` đã chạy.

Thứ tự nên khai báo từ rẻ đến đắt (rule, embedding, LLM judge) để request bị rule chặn không tốn lần gọi mạng. Core không tự sắp xếp lại.

### Xử lý lỗi

- Guard ném exception hoặc quá `timeout_s`: pipeline tạo `GuardResult` có `error` mô tả lỗi, `action` theo `on_error` của guard đó. `on_error: block` dừng pipeline như một lần block thường, `on_error: allow` chuyển sang guard kế tiếp với text không đổi.
- LLM judge trả về nội dung không parse được thành JSON đúng schema: tính là lỗi, xử lý như trên.
- Core không retry. Retry là việc của client truyền vào hoặc của app.
- Core ghi log bằng `logging` chuẩn với tên guard và loại lỗi, không ghi nội dung text vì có thể chứa PII.

### Giới hạn đã biết

- Request hợp lệ đi qua mọi guard nên latency bằng tổng các lần gọi mạng. Interface đã là async, nên nâng lên chạy song song sau này chỉ sửa `pipeline.py`.
- Cosine Python thuần phù hợp với tập vài trăm ví dụ. Tập vài nghìn ví dụ cần numpy hoặc vector store.
- Nếu app không gọi `warmup()`, request đầu của guard `embedding` chậm hơn vì phải embed tập ví dụ, và có thể quá `timeout_s`.

## Evaluate

### Định dạng dataset

File JSONL, mỗi dòng một mẫu:

```json
{"id": "inj-001", "stage": "input", "text": "Ignore previous instructions...", "expected": "block", "category": "prompt_injection"}
{"id": "pii-004", "stage": "input", "text": "Mail tôi là a@b.com", "expected": "redact", "category": "pii", "must_not_contain": ["a@b.com"]}
{"id": "ok-010", "stage": "output", "text": "...", "user_input": "...", "expected": "allow", "category": "benign"}
```

| Trường | Bắt buộc | Ý nghĩa |
| --- | --- | --- |
| `id` | có | Duy nhất trong file. |
| `stage` | có | `input` hoặc `output`. |
| `text` | có | Nội dung đưa vào guardrail. |
| `expected` | có | `allow`, `redact`, hoặc `block`. |
| `category` | có | Nhãn loại, ví dụ `prompt_injection`, `pii`, `benign`. |
| `user_input` | không | Câu hỏi gốc, dùng cho mẫu stage output. |
| `must_not_contain` | không | Với mẫu `redact`: các chuỗi không được còn trong text sau khi che. |

Dòng sai định dạng làm lệnh dừng với số dòng và lý do, không bỏ qua im lặng.

### Hai chế độ chạy

- Pipeline (mặc định): chạy đúng file YAML của production trên từng mẫu theo `stage` của mẫu, ghi `Decision` đầy đủ. Đo kết quả end-to-end và guard nào đã chặn.
- `--isolate`: thay cho lần chạy pipeline, không chạy thêm. Với mỗi guard lấy từ `guardrail.guards(stage)`, dùng `guardrail.only(stage, name)` để có guardrail chỉ chứa guard đó và chạy trên mọi mẫu cùng stage. Guard được định danh bằng `<stage>/<name>` vì `name` chỉ duy nhất trong một stage. Cần vì short-circuit ở chế độ pipeline che mất độ chính xác của guard đứng sau.

Runner gọi `warmup()` trước, rồi chạy các mẫu đồng thời có giới hạn bằng `--concurrency` (mặc định 4) để không vượt rate limit của provider. CLI tự load `.env` bằng `python-dotenv`.

### Metrics

Phân loại nhị phân: mẫu positive là mẫu có `expected` là `block` hoặc `redact`, dự đoán positive là `Decision.action` khác `allow`.

- Confusion matrix, precision, recall, F1, false positive rate, tính chung và theo `category`.
- Tỷ lệ khớp đúng hành động (`expected` bằng `Decision.action`), để thấy trường hợp mẫu cần redact lại bị block.
- Redact: tỷ lệ mẫu `redact` còn sót ít nhất một chuỗi trong `must_not_contain`.
- Vận hành: latency p50 và p95 theo guard và theo pipeline, số mẫu có guard lỗi hoặc timeout, số lần guard gọi mạng (guard có `requires` khác rỗng) được chạy.
- Chế độ pipeline: số mẫu bị chặn theo từng guard (`blocked_by`), tách true positive và false positive.
- Chế độ `--isolate`: mọi metrics phân loại, redact và vận hành ở trên được tính riêng cho từng guard, với dự đoán là `action` của guard đó. Từ đó có bảng recall theo guard và category, false positive rate của từng guard trên mẫu `allow`, và bảng threshold sweep cho guard có `requires` khác rỗng (`embedding`, `llm_judge`): precision, recall, F1 ở các ngưỡng từ 0.05 đến 0.95 bước 0.05, tính từ `score` đã ghi. Tuỳ chọn `--categories` giới hạn tập positive dùng cho sweep, vì một guard chỉ nhắm vào một số category.

Mẫu có guard lỗi vẫn được tính theo `action` cuối cùng (theo `on_error`), đồng thời được đếm riêng ở số lỗi. Vì lỗi (ví dụ rate limit) với `on_error: block` làm tăng giả recall và false positive rate, các metrics phân loại được báo thêm một bản loại trừ mẫu có lỗi. Threshold sweep luôn loại trừ record có `score` là None và ghi số record bị loại.

`metrics.py` chỉ gồm hàm thuần nhận danh sách record và trả số liệu, không đọc file và không gọi mạng.

### Kết quả và CLI

Mỗi lần chạy ghi vào `_workspace/runs/<YYYYMMDD-HHMMSS>/` (thư mục `_workspace` đã có trong `.gitignore`). Thư mục đã tồn tại thì lệnh dừng với lỗi, không ghi đè.

- `records.jsonl`: dùng để soi từng false positive và false negative. Chế độ pipeline: mỗi mẫu một dòng gồm id, expected, action, text cuối cùng, blocked_by, và mọi `GuardResult`. Chế độ `--isolate`: mỗi cặp (guard, mẫu) một dòng gồm guard (`<stage>/<name>`), id, expected, và `GuardResult` của guard đó.
- `summary.json`: toàn bộ metrics, kèm trường `mode` (`pipeline` hoặc `isolate`). Chế độ pipeline có ba khoá `overall`, `by_category`, `by_guard`. Chế độ `--isolate` có khoá `guards`, ánh xạ `<stage>/<name>` tới `overall`, `by_category` và `sweep` của guard đó. `compare` so các khoá có ở cả hai lần chạy.
- `summary.md`: bảng metrics dễ đọc.
- `config.yaml`: bản sao config đã dùng.

```text
python main.py check "nội dung cần thử" [--stage input] [--user-input "..."] [--config config.yaml]
python main.py eval --dataset datasets/injection.jsonl [--config config.yaml] [--isolate] [--concurrency 4] [--categories prompt_injection]
python main.py compare _workspace/runs/A _workspace/runs/B
```

`check` chạy một đoạn text qua guardrail để thử thủ công, in `Decision` dạng JSON: action, text cuối cùng, blocked_by, và từng `GuardResult`. `--stage` mặc định là `input`. `main.py` load `.env` trước khi chạy mọi lệnh.

`--config` mặc định là `config.yaml` ở root.

`compare` đọc hai `summary.json` và in bảng metrics của hai lần chạy cùng chênh lệch. Hai lần chạy khác `mode` thì `compare` báo lỗi.

LLM judge chạy temperature 0 nhưng kết quả vẫn có thể lệch giữa các lần chạy, nên chênh lệch nhỏ giữa hai lần `compare` chưa chắc là khác biệt thật. Không có cache response: chạy lại eval là gọi lại model.

### Adapter dataset public

Mỗi adapter là một hàm đổi dòng dữ liệu gốc thành mẫu chuẩn, chạy một lần để sinh file trong `datasets/`. Ứng viên bản đầu: `deepset/prompt-injections` và `lmsys/toxic-chat` trên Hugging Face. `deepset/prompt-injections` có license Apache-2.0. `lmsys/toxic-chat` có license CC-BY-NC-4.0 (phi thương mại), nên file sinh từ dataset public ghi vào `datasets/public/`, thư mục này nằm trong `.gitignore` và chỉ dùng để benchmark nội bộ.

## Test

Test không gọi mạng, dùng `ChatClient` và `Embedder` giả.

- Guard: mỗi loại có test cho trường hợp khớp, không khớp, redact đúng vị trí, input rỗng, và ngưỡng ở đúng giá trị biên.
- Pipeline: thứ tự chạy, redact nối tiếp giữa các guard, short-circuit khi block, timeout, `on_error` ở cả hai giá trị, JSON sai từ LLM judge.
- Config: mỗi điều kiện lỗi trong mục cấu hình có một test xác nhận `GuardrailConfigError`.
- Metrics: fixture record tính tay, gồm trường hợp không có positive và không có negative (mẫu số bằng 0).
- Dataset loader: dòng thiếu trường, `expected` sai giá trị, `id` trùng.

## Câu hỏi còn mở

- Model id mặc định cho `embedding` và `llm_judge` trong `config.yaml`: (pending), chọn từ `ModelId` sau khi chạy `scripts/sync_models.sh`.
- Model mặc định trong `config.yaml` (`nvidia/nemotron-3.5-embed-1b:free`, `google/gemma-4-31b-it:free`) chưa được chạy thử vì chưa có `OPENROUTER_API_KEY`: (pending). Endpoint embedding đã xác minh là `POST /embeddings` với `{model, input}` theo <https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request.md>.
