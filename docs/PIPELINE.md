# Pipeline xử lý video Douyin (post_process)

Tài liệu mô tả flow hiện tại trên n8n: từ dữ liệu Douyin (Apify) đến các video ngắn đã dựng lại,
đã che caption tiếng Trung và có lời thoại tiếng Việt (chưa có giọng đọc).

Cập nhật: 2026-11-20.

## Tổng quan

```
Apify (JSON Douyin)
  │
  ▼
1. make_dir          tạo thư mục job
2. download_media    tải video + nhạc, HEVC -> H.264
3. split_video       cắt video thành các cảnh (clip) + scenelog
4. compress_clips    bản nhỏ của các clip để gửi Gemini
5. gemini_reorder    Gemini lên kế hoạch nhiều video ngắn (thứ tự cảnh)
6. remove_captions   dò vị trí + thời gian caption tiếng Trung (captions.json), không sửa clip
7. render_videos     ghép bản cuối + bản proxy (gửi Gemini), cả hai từ clip gốc
8. gemini_voiceover  Gemini viết lời thoại tiếng Việt cho từng video
9. gemini_tts        Gemini TTS đọc lời thoại (giọng Puck), có timestamp từng câu
10. compose_videos   ghép giọng đọc + phụ đề tiếng Việt; tấm kính che luôn caption tiếng Trung
  │
  ▼
video hoàn chỉnh: <job>/output/final/
```

Bước 6 không phụ thuộc bước 4-5: có thể chạy ngay sau bước 3 (song song), miễn là xong trước bước 7.

## Quy ước chung

- Mọi script nằm trong `post_process/`, chạy trong container n8n bằng
  `python -m post_process.<tên_script> ...` (đã có `PYTHONPATH=/opt`, Python trong `/opt/venv`).
  Trong container **không dùng `uv run`**.
- Đường dẫn trong container: thư mục `/data/media` = `C:\Code\n8n-media\media` trên máy.
  Mọi file phải đọc/ghi trong `/data/media` (n8n giới hạn bằng `N8N_RESTRICT_FILE_ACCESS_TO`).
- **stdout**: dòng đầu là kết quả. Có thêm dòng sau ở `render_videos` (`proxy:`), `split_video` (`long_scenes:`) và
  `gemini_reorder` (`overlap:`); với các bước này n8n lấy kết quả bằng `.match(/^dir:(.*)$/m)[1]` (hoặc `json:`, `proxy:`)
  thay vì `.trim().replace(...)`. **stderr** là nhật ký tiến trình
  và lý do lỗi. Exit code `0` = thành công, `1` = thất bại; khi thất bại stdout luôn là `fail`.
- Tiền tố stdout:

  | Tiền tố | Ý nghĩa |
  |---|---|
  | `done` | thành công, không có đường dẫn trả về |
  | `mp4:<đường dẫn>` | file video |
  | `dir:<đường dẫn>` | thư mục |
  | `proxy:<đường dẫn>` | thư mục proxy (dòng thứ hai của `render_videos`) |
  | `json:<đường dẫn>` | file JSON |
  | `fail` | thất bại, xem stderr |

- **Mã lỗi** (các bước gọi Gemini: 5, 8, 9): mỗi lỗi là một dòng stderr dạng
  `[<script>] <MÃ>: <nội dung>`, định nghĩa trong `post_process/gemini_errors.py`. Lấy mã trong n8n:
  `{{ ($json.stderr.match(/\] (E_[A-Z_]+):/) || [])[1] }}`

  | Mã | Ý nghĩa | Nên làm |
  |---|---|---|
  | `E_QUOTA_DAILY` | hết hạn mức request/ngày của model; báo **ngay**, không thử lại. Cuối dòng có `retry_after=<giây>s` | chờ theo `retry_after` hoặc nâng tier, rồi chạy lại |
  | `E_RATE_LIMIT` | vẫn bị giới hạn theo phút sau khi đã thử lại | chạy lại sau ít phút |
  | `E_AUTH` | thiếu/sai API key, không có quyền (HTTP 401/403) | kiểm tra `GEMINI_API_KEY` |
  | `E_BAD_REQUEST` | request bị từ chối (HTTP 400/404: sai tên model, dữ liệu sai) | xem nội dung lỗi |
  | `E_SERVER` | lỗi phía Google (HTTP 5xx) sau khi thử lại | chạy lại sau |
  | `E_NETWORK` | không kết nối được Gemini sau khi thử lại | kiểm tra mạng |
  | `E_NO_OUTPUT` | Gemini trả lời nhưng thiếu nội dung (không có âm thanh, câu trả lời rỗng) | chạy lại |
  | `E_INVALID_ANSWER` | câu trả lời vẫn sai quy tắc sau mọi lần sửa | chạy lại / xem `gem_*response.json` |
  | `E_TOO_LONG` | giọng đọc dài hơn video kể cả khi tăng tốc tối đa (bước 9) | chạy lại bước 8 cho video đó |
  | `E_INPUT` | thiếu/sai file hoặc tham số đầu vào | sửa lệnh |
  | `E_UNEXPECTED` | lỗi khác (bug) | báo lại |

- Trong n8n, đường dẫn của bước sau được tính từ stdout của node ngay trước (`$json.stdout`).
  Khi cần stdout của một node xa hơn, dùng `$('Tên node').item.json.stdout`.
- Sửa file trong `post_process/` thì phải build lại image: `docker compose up -d --build n8n`.
  Đổi biến trong `.env` thì phải tạo lại container: `docker compose up -d --force-recreate n8n`.

## Cấu trúc thư mục của một job

```
/data/media/jobs/<jobId>/
├── input/
│   ├── <awemeId>.mp4                 video tải về (H.264)                     [2]
│   └── <awemeId>.mp3                 nhạc                                     [2]
├── output/
│   ├── <awemeId>_video_<N>.mp4       video ghép, 4K, còn caption tiếng Trung  [7]
│   └── final/<awemeId>_video_<N>.mp4 video hoàn chỉnh: giọng đọc + phụ đề    [10]
└── temp/
    └── <awemeId>/
        ├── sources/<N>.mp4           các cảnh, chất lượng gốc, còn caption    [3]
        ├── scenelog.csv / .json      danh sách cảnh + thời gian               [3]
        ├── stats.json                thống kê: long_scenes [3], overlaps [5]
        ├── runlog.txt                log của scenedetect                      [3]
        ├── compressed/<N>.mp4        bản nhỏ 480p có nhãn SCENE N             [4]
        ├── request.json              request gửi Gemini (lên kế hoạch)        [5]
        ├── gem_response.json         response gốc của Gemini                  [5]
        ├── reorder.json              kế hoạch các video + timestamp           [5]
        ├── clean/<N>.mp4             các cảnh đã che caption (chỉ --mode panel|fill) [6]
        ├── clean/captions.json       báo cáo vùng caption của từng clip       [6]
        ├── proxy/proxy_<awemeId>_video_<N>.mp4   bản nhẹ 480x854, còn caption [7]
        ├── voiceover.json            lời thoại tiếng Việt                     [8]
        ├── gem_voiceover_response.json  response gốc của Gemini               [8]
        ├── voice/voice_video_<N>.wav giọng đọc, 24 kHz mono                   [9]
        └── tts.json                  timestamp từng câu theo thời gian video  [9]
```

Số trong ngoặc vuông là bước tạo ra file.

## Các bước

### 1. make_dir

Tạo thư mục job cùng `input/`, `output/`, `temp/`, và mọi thư mục cha còn thiếu (`/data/media`, `/data/media/jobs`).
Nếu thư mục `./media` trên máy bị xoá khi container đang chạy, mount `/data/media` bị hỏng: script báo `fail` kèm
hướng dẫn (tạo lại `./media` hoặc khởi động lại container).

| | |
|---|---|
| Lệnh | `python -m post_process.make_dir <dir_path>` |
| Input | `dir_path`: thư mục job, ví dụ `/data/media/jobs/<jobId>` |
| Output | `<dir_path>/input`, `<dir_path>/output`, `<dir_path>/temp` (đã tồn tại thì bỏ qua) |
| stdout | `done` / `fail` |

### 2. download_media

Tải video không watermark và nhạc; nếu video là HEVC (H.265) thì chuyển sang H.264 + AAC và xoá bản HEVC.

| | |
|---|---|
| Lệnh | `python -m post_process.download_media <aweme_id> <video_url> <music_url> <output_dir>` |
| Input | `aweme_id`: `awemeId` (chỉ chữ, số, `_`, `-`); `video_url`: `videoUrlNoWatermark`; `music_url`: `musicUrl`; `output_dir`: `<job>/input` |
| Output | `<output_dir>/<awemeId>.mp4`, `<output_dir>/<awemeId>.mp3` |
| stdout | `mp4:<đường dẫn video>` / `fail` |

Lệnh n8n (dữ liệu từ node Apify):
```
python -m post_process.download_media "{{ $json.awemeId }}" "{{ $json.videoUrlNoWatermark }}" "{{ $json.musicUrl }}" /data/media/jobs/<jobId>/input
```
Link Douyin có chữ ký và hết hạn sau một thời gian: chạy ngay sau khi lấy JSON. Bắt buộc đặt URL trong ngoặc kép
(URL có ký tự `&`).

### 3. split_video

Cắt video thành các cảnh bằng PySceneDetect, đặt tên `1.mp4`, `2.mp4`… và xuất danh sách cảnh.

| | |
|---|---|
| Lệnh | `python -m post_process.split_video <input_video> <output_dir>` |
| Input | `input_video`: video từ bước 2; `output_dir`: `<job>/temp/<awemeId>/sources` |
| Output | `<output_dir>/<N>.mp4`; ở thư mục cha: `scenelog.csv`, `scenelog.json`, `runlog.txt` |
| stdout | `dir:<output_dir>` rồi một dòng `long_scenes:` / `fail` |

Dòng `long_scenes:` (cũng ghi vào `stats.json`) thống kê các cảnh dài **từ 3 giây trở lên** (`LONG_SCENE_S`), dạng
`long_scenes:<số cảnh ≥ 3s>_<tổng số cảnh>-<số thứ tự các cảnh đó, tăng dần, nối bằng _>`;
không có cảnh nào thì bỏ cả phần `-...`:

```
long_scenes:4_47-4_6_7_18     4 trên 47 cảnh dài ≥ 3s: cảnh 4, 6, 7, 18
long_scenes:0_47              không có cảnh nào
```

Parse trong n8n (Code node hoặc biểu thức):
`const [, long, total, list] = $json.stdout.match(/^long_scenes:(\d+)_(\d+)(?:-([\d_]+))?$/m);`
`const scenes = list ? list.split('_').map(Number) : [];`

Lệnh n8n (sau `download_media`, stdout `mp4:.../input/<awemeId>.mp4`):
```
python -m post_process.split_video "{{ $json.stdout.trim().replace(/^mp4:/, '') }}" "{{ (p => p.split('/').slice(0, -2).join('/') + '/temp/' + p.split('/').pop().replace(/\.mp4$/, '') + '/sources')($json.stdout.trim().replace(/^mp4:/, '')) }}"
```

`scenelog.json`: `{"timecode_list": [...], "scenes": [{"Scene Number", "Start Time (seconds)", "End Time (seconds)", "Length (seconds)", ...}]}`.

### 4. compress_clips

Tạo bản nhỏ của từng clip để gửi Gemini: cạnh ngắn 480px, tối đa 30 fps, H.264 CRF 30, âm thanh mono 64k,
in nhãn `SCENE N` ở góc trên trái. Clip gốc không bị sửa.

| | |
|---|---|
| Lệnh | `python -m post_process.compress_clips <clips_dir> <output_dir> [--short-side 480] [--fps 30] [--no-label]` |
| Input | `clips_dir`: `.../sources`; `output_dir`: `.../compressed` (phải khác `clips_dir`) |
| Output | `<output_dir>/<N>.mp4` |
| stdout | `dir:<output_dir>` / `fail` |

Lệnh n8n (sau `split_video`, stdout `dir:.../sources`):
```
python -m post_process.compress_clips "{{ $json.stdout.match(/^dir:(.*)$/m)[1].trim() }}" "{{ $json.stdout.match(/^dir:(.*)$/m)[1].trim().replace(/\/sources$/, '/compressed') }}"
```

### 5. gemini_reorder

Gửi cả 40 clip nén (base64, một request) cho Gemini để lên kế hoạch **nhiều video ngắn**, mỗi video một góc
tiếp cận riêng. Gemini chỉ trả về thứ tự cảnh; script kiểm tra quy tắc rồi tự tính timestamp từ `scenelog.json`.

| | |
|---|---|
| Lệnh | `python -m post_process.gemini_reorder <scenelog_json> <clips_dir> <output_json> [--dry-run]` |
| Input | `scenelog_json`: `.../scenelog.json`; `clips_dir`: `.../compressed`; `output_json`: `.../reorder.json` |
| Môi trường | `GEMINI_API_KEY` (bắt buộc), `GEMINI_MODEL` (mặc định `gemini-3.8-flash`) |
| Output | `request.json` (request gửi đi), `gem_response.json` (response mọi lần gọi), `reorder.json` (chỉ khi thành công) — cùng thư mục |
| stdout | `json:<reorder.json>` rồi các dòng `overlap:`, `overlap_video:` / `fail` (`--dry-run`: chỉ ghi `request.json`, stdout `json:<request.json>`) |

**Kiểm tra đoạn trùng:** sau khi có kế hoạch hợp lệ, script tìm các đoạn **từ 2 cảnh liên tiếp trở lên** (cùng cảnh,
cùng thứ tự) xuất hiện ở từ hai video trở lên, ví dụ video 1 `1,2,3,4,9` và video 2 `7,2,3,4,19` trùng đoạn `2-3-4`.
Một đoạn nằm ở nhiều video chỉ in **một dòng** kèm mọi video có nó. Thời lượng một đoạn = tổng thời lượng các cảnh
của nó. Mỗi đoạn trùng một dòng `overlap:`, rồi mỗi video một dòng `overlap_video:` (số giây: 3 chữ số thập phân):
```
overlap:<các video, nối bằng _>-<các cảnh, nối bằng _>-<số giây>
overlap_video:<video>-<số giây bị trùng với video khác>_<thời lượng video>

overlap:1_2_3-2_3_4-2.700       video 1, 2 và 3 cùng có đoạn cảnh 2, 3, 4, dài 2.7s
overlap:1_4-10_11-1.733
overlap_video:1-4.433_6.700     video 1: 4.433s trên 6.7s bị trùng (mỗi cảnh tính một lần)
overlap_video:2-2.700_5.500
overlap_video:3-2.700_4.333
overlap_video:4-1.733_4.366     (video không trùng gì: 0.000_...)
```
Không có đoạn trùng nào thì không có dòng `overlap:`, các dòng `overlap_video:` đều là `0.000`.
Đây chỉ là thống kê, không làm job thất bại.

**`stats.json`** (`temp/<awemeId>/stats.json`, module `job_stats.py`): bản đầy đủ của các thống kê trên stdout, để xem
lại sau. `split_video` tạo mới file với khóa `long_scenes` (thêm `durations`, `long_seconds`, `total_seconds`);
`gemini_reorder` thêm khóa `overlaps` (`runs`, `per_video` có `shared_ratio`), xoá khóa này khi bắt đầu chạy lại.

Gom stdout của cả hai bước thành một JSON trong n8n: Code node [flows/code/job_stats.js](../flows/code/job_stats.js)
(cấu trúc giống `stats.json`). Đặt ở nhánh riêng, không chèn giữa chuỗi node.

**Gửi thống kê lên Discord** (công cụ cho người phát triển, không phải tính năng): nhánh riêng từ node "Nấu các cảnh"
→ Code node [flows/code/discord_stats.js](../flows/code/discord_stats.js) (mode "Run Once for Each Item") đọc
`stats.json` + `reorder.json`, trả về `content` (tin nhắn ≤ 2000 ký tự: cảnh dài, % trùng từng video, các đoạn trùng)
và `color` (xanh < 20% trùng, vàng < 50%, đỏ ≥ 50%) → node Discord gửi `{{ $json.content }}`.

**Số video đầu ra:** `REORDER_TARGET_VIDEOS` trong `.env` (hiện là **4**) = tạo đúng N video; để trống = Gemini tự quyết
(tối thiểu đủ để dùng hết các cảnh). N được hạ xuống khi video gốc quá ngắn (cần ít nhất thời lượng tối thiểu mỗi video,
mỗi cảnh dùng tối đa 2 lần). Khi N video × thời lượng tối đa không chứa hết video gốc (video gốc dài), vẫn giữ N video và
**bỏ quy tắc dùng hết các cảnh** (Gemini chọn cảnh hay nhất). Các điều chỉnh này ghi trong `reorder.json` → `rules.notes`.

**Thời lượng mỗi video:** `TARGET_VIDEO_S` trong `.env` (mặc định **15**). Khoảng cho phép đi theo mục tiêu:
tối đa = mục tiêu + 5 giây, tối thiểu = min(8, mục tiêu − 5) giây. Ví dụ 15 → 8-20s, 10 → 5-15s, 30 → 8-35s.
Mục tiêu phải từ 6 giây trở lên, hoặc **0 = mục tiêu là độ dài video gốc**: khoảng cho phép theo cùng công thức (gốc 49s →
8-54s; thực tế không quá độ dài gốc vì một video không lặp cảnh), không bắt mỗi video dùng đủ mọi cảnh; khi đó bỏ hai
quy tắc "mỗi cảnh tối đa 2 video" và "dùng chung tối đa 50%" (chỉ cấm hai video giống hệt nhau), số video
`REORDER_TARGET_VIDEOS` được giữ nguyên, không bị hạ. Giá trị dùng thực tế ghi trong `reorder.json` → `rules`
(`full_source`, `target_video_s`, `min_video_s`, `max_video_s`).

**Ưu tiên khi chỉnh: giảm trùng cảnh giữa các video, nhưng vẫn giữ đủ nội dung.** Mỗi video phải tự đứng được
(mở đầu, các ý chính của góc tiếp cận, kêu gọi mua), không cắt bớt ý chỉ để né trùng.

Quy tắc kiểm tra (sai thì gửi lỗi lại cho Gemini trong cùng cuộc trò chuyện, tối đa 3 lần):

- mỗi video trong khoảng cho phép ở trên (mặc định 8-20 giây, mục tiêu 15 giây);
- trong một video mỗi cảnh xuất hiện tối đa một lần;
- đúng `REORDER_TARGET_VIDEOS` video (nếu có đặt);
- mọi cảnh được dùng ít nhất một lần (trừ khi bị bỏ như trên);
- mỗi cảnh dùng trong tối đa 2 video;
- hai video dùng chung tối đa 50% số cảnh của video ngắn hơn.

`reorder.json`:
```
{
  "model", "product", "aspects": [...], "rules": {...},
  "source_scene_count", "source_total_duration", "video_count",
  "videos": [{
    "video_index", "angle", "title", "summary", "key_points": [...],
    "scene_count", "total_duration",
    "scenes": [{"position", "scene_number", "source_file", "src_start", "src_end",
                "duration", "out_start", "out_end", "description", "reason"}]
  }],
  "scene_usage": {"<scene_number>": [<video_index>, ...]}
}
```
Trường lấy từ Gemini: `product`, `aspects`, `angle`, `title`, `summary`, `key_points`, thứ tự cảnh, `description`, `reason`.
Mọi thời gian (`src_*`, `out_*`, `duration`, `total_duration`) do script tính từ `scenelog.json`.
Khi chạy thật, `reorder.json` và `gem_response.json` của lần trước bị xoá trước khi gọi Gemini.

Lệnh n8n (sau `compress_clips`, stdout `dir:.../compressed`):
```
python -m post_process.gemini_reorder "{{ $json.stdout.trim().replace(/^dir:/, '').replace(/\/compressed$/, '/scenelog.json') }}" "{{ $json.stdout.trim().replace(/^dir:/, '') }}" "{{ $json.stdout.trim().replace(/^dir:/, '').replace(/\/compressed$/, '/reorder.json') }}"
```

### 6. remove_captions

Tìm caption nhỏ (dòng chữ cố định của video) bằng OpenCV, không dùng AI. Mặc định (`--mode detect`) **chỉ dò**:
ghi vị trí và thời gian caption của từng clip vào `captions.json`, không encode clip nào. Việc che làm một lần duy
nhất ở bước 10 (tấm kính phụ đề), nên video chỉ encode một lần ít hơn, không bị mờ hai lớp, và tấm kính chỉ cần ôm
khung chữ thật (`text_box`, không đệm). Vùng chữ căn giữa theo khung hình (caption Douyin luôn căn giữa; nếu một
bên bị dính vật sáng thì lấy theo bên gần tâm hơn).

`--mode panel|fill` (cách cũ) che ngay trên clip: tấm kính mờ bo góc (`box`, có đệm, cách mép khung hình ít nhất 3%),
chữ bên dưới được xoá trước khi làm mờ, và ghi clip vào `output_dir`; khi đó bước 7 phải lấy clip từ `clean/`. Caption dài gần hết chiều ngang vẫn được nhận. Khi không nhận ra được theo dòng chữ (chữ dính vào
vật sáng phía sau, hoặc chữ to hơn caption nằm trên dòng caption), bước dự phòng đo chiều cao thật của chữ theo mật độ
pixel từng hàng quanh dòng caption (chấp nhận chữ cao tới 2,4 lần caption, câu ngắn từ 2 chữ).
Mọi chữ nằm trên dòng caption đều bị che (kể cả chữ to hơn caption); chữ overlay ở vị trí khác được giữ nguyên.

| | |
|---|---|
| Lệnh | `python -m post_process.remove_captions <sources_dir> <output_dir> [--mode detect\|panel\|fill] [--preview DIR]` |
| Input | `sources_dir`: `.../sources`; `output_dir`: `.../clean` (phải khác `sources_dir`) |
| Output | `<output_dir>/captions.json`; với `panel\|fill` thêm `<output_dir>/<N>.mp4` (clip có caption thì encode lại, không có thì copy nguyên) |
| stdout | `dir:<output_dir>` / `fail` |
| Thời gian | khoảng 3-4 phút cho 40 clip 4K (đặt timeout node từ 10 phút) |

`--mode fill`: chỉ xoá chữ, không có tấm kính. `--preview DIR`: lưu ảnh trước/sau của từng clip (`panel|fill`), hoặc
một khung hình có vẽ khung chữ dò được màu xanh (`detect`).

`captions.json`: `{"mode", "caption_line": {"center_y", "height"}, "frame": {"width", "height"}, "clips": {"<N>.mp4": {"caption", "method", "glyphs_removed", "detection", "box": {"x","y","w","h"}, "text_box": {"x","y","w","h"}, "start", "end", "hits", "mask_share"}}}`.

Lệnh n8n (sau `gemini_reorder`, stdout `json:.../reorder.json`):
```
python -m post_process.remove_captions "{{ $json.stdout.match(/^json:(.*)$/m)[1].trim().replace(/\/reorder\.json$/, '/sources') }}" "{{ $json.stdout.match(/^json:(.*)$/m)[1].trim().replace(/\/reorder\.json$/, '/clean') }}"
```

### 7. render_videos

Ghép các cảnh theo `reorder.json`. Một lần chạy xuất cả **bản cuối** (chất lượng gốc) và **bản proxy** (480x854)
để gửi Gemini, cả hai từ `sources/` nên còn caption tiếng Trung (bước 10 che). Hai bản có cùng cảnh, cùng thứ tự, cùng số khung hình.
Mỗi video ghép đủ mọi cảnh, nguyên độ dài (không phải bản tóm tắt).

| | |
|---|---|
| Lệnh | `python -m post_process.render_videos <reorder_json> <clips_dir> <output_dir> [--proxy-dir DIR] [--proxy-clips DIR] [--proxy-height 854] [--prefix TEXT] [--no-audio] [--height PX]` |
| Input | `reorder_json`: `.../reorder.json`; `clips_dir`: `.../sources` (`.../clean` nếu bước 6 chạy `panel\|fill`); `output_dir`: `<job>/output`; `--proxy-dir`: `.../proxy`; `--proxy-clips`: `.../sources`; `--prefix`: `<awemeId>_` |
| Output | `<output_dir>/<prefix>video_<N>.mp4`, `<proxy_dir>/proxy_<prefix>video_<N>.mp4` |
| stdout | dòng 1 `dir:<output_dir>`, dòng 2 `proxy:<proxy_dir>` (khi có `--proxy-dir`) / `fail` |

Kiểm tra: thiếu clip nào (ở cả hai thư mục) thì báo ngay trước khi render; một clip lệch quá 0,25 giây so với
`duration` của cảnh trong kế hoạch (clip sai/hỏng) thì `fail`; video render lệch quá 0,5 giây so với tổng thời lượng
đo thật của các clip (lỗi khi ghép) thì `fail` (không so với `total_duration`: con số này cộng từ CSV, còn mỗi file
clip dài hơn vài ms do đệm AAC, cộng dồn qua nhiều cảnh); tham số đầu không phải file `reorder.json`, hoặc `--proxy-dir` trùng thư mục khác, thì `fail`.
Video được encode lại (H.264 CRF 20 + AAC 192k) để tránh lỗi âm thanh ở chỗ nối. `--no-audio` bỏ âm thanh gốc.

Lệnh n8n (sau `remove_captions`, stdout `dir:.../clean`):
```
python -m post_process.render_videos "{{ $json.stdout.trim().replace(/^dir:/, '').replace(/\/clean$/, '') }}/reorder.json" "{{ $json.stdout.trim().replace(/^dir:/, '').replace(/\/clean$/, '/sources') }}" "{{ $json.stdout.trim().replace(/^dir:/, '').split('/').slice(0, -3).join('/') }}/output" --proxy-dir "{{ $json.stdout.trim().replace(/^dir:/, '').replace(/\/clean$/, '') }}/proxy" --proxy-clips "{{ $json.stdout.trim().replace(/^dir:/, '').replace(/\/clean$/, '') }}/sources" --prefix "{{ $json.stdout.trim().replace(/^dir:/, '').split('/').slice(-2, -1)[0] }}_"
```

Lấy từng dòng stdout ở node sau: `{{ $json.stdout.match(/^dir:(.*)$/m)[1] }}`, `{{ $json.stdout.match(/^proxy:(.*)$/m)[1] }}`.

### 8. gemini_voiceover

Gửi từng proxy (base64, mỗi video một request) cùng thông tin trong `reorder.json` để Gemini viết lời thoại
tiếng Việt: năng động, bán hàng, xưng "mình" - gọi người xem "bạn". Chưa có timestamp.

| | |
|---|---|
| Lệnh | `python -m post_process.gemini_voiceover <reorder_json> <proxy_dir> [--output FILE] [--video N ...] [--rate 4.4] [--dry-run]` |
| Input | `reorder_json`: `.../reorder.json`; `proxy_dir`: `.../proxy` (tìm file `*video_<N>.mp4`) |
| Môi trường | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| Output | `voiceover.json`, `gem_voiceover_response.json` (cạnh `reorder.json`, hoặc cạnh `--output`) |
| stdout | `json:<voiceover.json>` / `fail` (`--dry-run`: ghi `voiceover_request.json`, không gọi Gemini) |

Kiểm tra (sai thì gửi lại cho Gemini, tối đa 3 lần): lời thoại phải **phủ kín video**: tổng âm tiết của phần đọc nằm
trong 85-108% của `thời lượng × --rate` (mục tiêu 95%; `--rate` mặc định 4,4 âm tiết/giây, tốc độ đọc 1 request); mỗi câu phụ đề tối đa 36 ký tự; có `tts_style`.
`--video 2 3`: chỉ viết lại các video đó, giữ nguyên các video khác trong `voiceover.json`.

`voiceover.json`:
```
{
  "model", "product", "style": {"tone", "speaker", "viewer"}, "rate",
  "videos": [{
    "video_index", "proxy_file", "title", "angle", "duration",
    "tts_style",                 câu tiếng Anh hướng dẫn giọng đọc cho Gemini TTS
    "script",                    toàn bộ lời đọc (dạng đọc)
    "subtitle_text",             toàn bộ lời đọc (dạng hiển thị)
    "syllables", "rate", "estimated_speech_s",
    "segments": [{"text": "chỉ 4,7g", "spoken": "chỉ bốn phẩy bảy gam"}]
  }]
}
```
`text` dùng làm phụ đề, `spoken` dùng cho TTS (số, ký hiệu, viết tắt đã viết thành chữ).
Script kiểm tra từng đoạn: `spoken` phải đọc đúng từng chữ tiếng Việt của `text` (chỉ số, ký hiệu, viết tắt, từ
nước ngoài được viết khác). Thừa chữ, bỏ hoặc đổi chữ có dấu thì gửi lại Gemini sửa (tối đa 3 lần), nếu không thì `fail`.

Lệnh n8n (sau `render_videos`):
```
python -m post_process.gemini_voiceover "{{ $json.stdout.match(/^proxy:(.*)$/m)[1].replace(/\/proxy$/, '') }}/reorder.json" "{{ $json.stdout.match(/^proxy:(.*)$/m)[1] }}"
```

### 9. gemini_tts

**Xoay vòng giọng:** mỗi video một giọng khác nhau, bốc ngẫu nhiên từ `GEMINI_TTS_VOICES` (mặc định Puck, Laomedeia,
Sadachbia, Fenrir; Zephyr bị bỏ vì đọc chậm nhất); giọng đã dùng trong vòng thì bỏ ra, hết danh sách mới sang vòng mới. Giọng được gán trước khi
gọi API nên video phải chuyển sang 3.1 vẫn giữ đúng giọng (hai model cùng tên giọng); chạy lại một phần (`--video`) thì
giọng của các video giữ lại được tính là đã dùng. `--voice X` = một giọng cho mọi video.

Đọc lời thoại bằng Gemini TTS, **giọng miền Nam** (câu cố định thêm vào phong cách đọc,
đổi bằng `--accent`). **Mỗi video chỉ 1 request**: mọi câu được đọc liền một lần (ngữ điệu liền mạch).
Gemini TTS không trả timestamp, nên script tìm ranh giới câu: chọn **cùng lúc** bộ khoảng lặng cho khoảng dừng
dài nhất và tốc độ đọc đều nhất giữa các câu (quy hoạch động; chọn từng chỗ gần vị trí dự kiến hay lấy nhầm chỗ ngắt
hơi giữa câu, làm phụ đề hiện sớm hơn giọng). Xét cả khoảng lặng rất ngắn (≥10ms): TTS hay đọc liền
hai câu, thiếu một chỗ ngắt là các câu ở giữa lệch dây chuyền. Không bộ nào hợp lý (câu quá ngắn, tốc độ bất thường) thì chia theo số âm tiết;
cách đã dùng ghi ở `split` (`silence` / `proportional` / `single`). Sau đó cắt khoảng lặng hai đầu, giọng bắt đầu sau 0,15s.

**Model và dự phòng:** `gemini-3.8-flash-tts` trước; khi nó báo **hết hạn mức ngày** thì chuyển **ngay** sang
`gemini-3.1-flash-tts-preview` cho video đó và các video sau. Cả hai đều hết thì dừng: `E_QUOTA_DAILY` (kèm
`retry_after` ngắn nhất), giữ các video đã xong, dòng cuối stderr ghi các video cần chạy lại (`--video ...`).
3.8 nhận phong cách qua trường riêng và thẻ `<short pause>` giữa các câu; 3.1 nhận phong cách viết trước văn bản,
các câu xuống dòng, và trả âm thanh PCM thô (script tự xử lý). `tts.json` ghi model đã dùng cho từng video.

| | |
|---|---|
| Lệnh | `python -m post_process.gemini_tts <voiceover_json> <output_dir> [--voice Puck] [--model M] [--fallback-model M] [--accent TEXT] [--video N ...] [--lead 0.15] [--max-speedup 1.15]` |
| Input | `voiceover_json`: `.../voiceover.json`; `output_dir`: `.../voice` |
| Môi trường | `GEMINI_API_KEY`; `GEMINI_TTS_MODEL` (mặc định `gemini-3.8-flash-tts`); `GEMINI_TTS_FALLBACK_MODEL` (mặc định `gemini-3.1-flash-tts-preview`, để trống = không dự phòng); `GEMINI_TTS_RPM` (mỗi model, mặc định 10); `GEMINI_TTS_VOICES` (danh sách giọng xoay vòng); `GEMINI_TTS_ACCENT` (câu yêu cầu giọng vùng miền, để trống = không) — tất cả trong `.env`, tham số dòng lệnh được ưu tiên hơn |
| Output | `<output_dir>/voice_video_<N>.wav` (24 kHz mono, bắt đầu từ giây 0 của video); `tts.json` cạnh `voiceover.json` |
| stdout | `json:<tts.json>` / `fail` (video nào lỗi thì bỏ, các video còn lại vẫn được ghi) |
| Thời gian | khoảng 7-10 giây mỗi video |

**Không để khoảng lặng ở cuối:** giọng đọc ngắn hơn video thì nới các khoảng nghỉ giữa các câu (tối đa +0,35s mỗi chỗ,
chỉ khi tách câu bằng khoảng lặng), rồi đọc chậm lại một chút (không chậm hơn 0,92x), để giọng kết thúc khoảng 0,25s
trước khi hết video (`added_pause`, `speed` trong `tts.json`).
Khớp độ dài: giọng đọc dài hơn video (trừ 0,1s cuối) thì tăng tốc giữ nguyên cao độ (`atempo`), tối đa `--max-speedup`
(1,15x). Nếu tăng tốc tối đa vẫn không đủ (giọng đọc chậm), video được **đọc lại 1 lần** bằng cùng giọng với yêu cầu
đọc nhanh hơn (tốn thêm 1 request), giữ bản ngắn hơn (`faster_retry` trong `tts.json`); vẫn dài quá thì `E_TOO_LONG`,
stderr gợi ý chạy lại `gemini_voiceover --video N --rate <tốc độ đo được>`.
Tốc độ toàn bài bất thường (đọc sót/lặp chữ) thì hỏi lại 1 lần, vẫn sai thì `E_NO_OUTPUT`.
Request giãn cách theo `GEMINI_TTS_RPM`; 429 theo phút và lỗi 400 "audio stream could not be completed ... retry"
thì chờ rồi thử lại. Hạn mức Tier 1: 100 request/ngày mỗi model, tức khoảng 100 video/ngày với 3.8 (+ dự phòng 3.1);
đặt lại lúc 0 giờ giờ Thái Bình Dương (15:00 giờ Việt Nam).

`tts.json`:
```
{
  "model", "fallback_model", "voices", "accent", "sample_rate", "audio_dir",
  "videos": [{
    "video_index",
    "voice",            giọng đã gán cho video này (xoay vòng)
    "model",            model đã thật sự dùng cho video này
    "split",            silence | proportional | single: cách tìm ranh giới câu
    "audio_file", "video_duration", "speech_duration",
    "speed",            hệ số tăng tốc đã áp dụng (1.0 = không)
    "measured_rate",    âm tiết/giây đo được
    "segments": [{"index", "text", "spoken", "start", "end"}]    giây, theo thời gian video
  }]
}
```

Lệnh n8n (sau `gemini_voiceover`, stdout `json:.../voiceover.json`):
```
python -m post_process.gemini_tts "{{ $json.stdout.trim().replace(/^json:/, '') }}" "{{ $json.stdout.trim().replace(/^json:/, '').replace(/\/voiceover\.json$/, '/voice') }}"
```

### 10. compose_videos

Ghép giọng đọc và phụ đề tiếng Việt vào video cuối (bước 7). Âm thanh gốc bị bỏ (mặc định).
Video nhỏ hơn 4K được **phóng lên 2160x3840** (lanczos, `--height 3840` mặc định; không bao giờ thu nhỏ;
`--height 0` giữ nguyên). Tấm kính và chữ được vẽ sau khi phóng nên vẫn nét; vùng che chữ Trung trong
`captions.json` (toạ độ theo clip gốc, `frame`) được nhân theo tỉ lệ. Phóng to không thêm chi tiết, file nặng hơn khoảng 3 lần.
Phụ đề: chữ trắng đậm (Noto Sans Bold) có bóng, hiện đúng `start`-`end` của từng câu trong `tts.json`,
đặt giữa một tấm kính mờ (hình chữ nhật bo góc, căn giữa khung hình, trên dòng caption).
- **Kích thước cố định cho cả video** (không phóng to thu nhỏ): chiều dài lớn nhất và chiều cao lớn nhất trong số
  nền của các câu tiếng Việt và vùng chữ Trung trong các cảnh của video (lấy từ `captions.json` + `reorder.json`).
- **Che luôn chữ Trung** (`captions.json` mode `detect`): vùng chữ Trung là `text_box` + lề an toàn 0,12 x chiều cao
  dòng caption (viền/bóng chữ), không đệm thêm, luôn cách mép khung hình ít nhất 3%. Kính được làm mờ mạnh (0,85 x chiều cao dòng caption, như tấm che cũ)
  để chữ không ánh qua; hiện cả khi `--no-subtitles`. Thiếu `reorder.json` thì `fail`.
- Với `captions.json` mode `panel|fill` (cách cũ), tấm kính phủ lên tấm che cũ (`box` + 0,3% chiều cao) để không lộ mép.
- **Hiện suốt video**, từ khung đầu đến khung cuối; giữa hai câu chỉ ẩn chữ, tấm kính vẫn còn.
- Kính được giảm độ bão hoà màu (40%) để màu của caption (ví dụ chữ vàng) không ánh qua.
- **Chữ nằm giữa tấm kính theo chiều dọc**, căn theo thân chữ (chiều cao chữ hoa), không tính dấu phía trên và đuôi
  chữ phía dưới, nên nhìn cân.
- Độ rộng chữ được đo bằng chính font; câu quá dài (hơn 90% chiều ngang) thì thu nhỏ chữ, không dưới 1,8% chiều cao.

| | |
|---|---|
| Lệnh | `python -m post_process.compose_videos <tts_json> <videos_dir> <output_dir> [--reorder FILE] [--captions FILE] [--music FILE] [--music-volume 0.12] [--original-volume 0] [--tint dark\|light] [--no-subtitles] [--video N ...] [--height 3840]` |
| Input | `tts_json`: `.../tts.json`; `videos_dir`: `<job>/output` (tìm `*video_<N>.mp4`); `output_dir`: `<job>/output/final` (phải khác `videos_dir`). Mặc định đọc `reorder.json` cạnh `tts.json` và `clean/captions.json` |
| Output | `<output_dir>/<tên file như video vào>`: H.264 CRF 20 + AAC 192k, đúng độ dài video vào |
| stdout | `dir:<output_dir>` / `fail` |
| Thời gian | khoảng 1 phút 10 giây cho 4 video 4K |

Tuỳ chọn âm thanh: `--music <awemeId>.mp3` thêm nhạc nền (lặp, mờ dần 1 giây cuối) ở `--music-volume`;
`--original-volume 0.1` giữ tiếng gốc nhỏ. `--tint light` dùng kính sáng (mặc định `dark` để chữ trắng dễ đọc).

Lệnh n8n (sau `gemini_tts`, stdout `json:.../temp/<awemeId>/tts.json`):
```
python -m post_process.compose_videos "{{ $json.stdout.trim().replace(/^json:/, '') }}" "{{ $json.stdout.trim().replace(/^json:/, '').split('/').slice(0, -3).join('/') }}/output" "{{ $json.stdout.trim().replace(/^json:/, '').split('/').slice(0, -3).join('/') }}/output/final"
```

## Cấu hình liên quan

| Ở đâu | Gì |
|---|---|
| `.env` | `GEMINI_API_KEY`, `GEMINI_MODEL` (bước 5, 8); `REORDER_TARGET_VIDEOS`, `TARGET_VIDEO_S` (bước 5); `GEMINI_TTS_MODEL`, `GEMINI_TTS_FALLBACK_MODEL`, `GEMINI_TTS_RPM`, `GEMINI_TTS_VOICES`, `GEMINI_TTS_ACCENT` (bước 9). Đổi `.env` thì phải tạo lại container |
| `Dockerfile` | Alpine 3.24 (`apk`), ffmpeg, font Noto, Python 3.14, OpenCV/numpy từ `apk`, `uv`, `post_process` copy vào `/opt/post_process` |
| `docker-compose.yml` | `./media` -> `/data/media`, `./n8n_data` -> `/home/node/.n8n` |
| Hằng số đầu mỗi script | ngưỡng quy tắc (thời lượng video, số lần dùng cảnh, kích thước caption, tốc độ đọc…) |

## Chi phí Gemini tham khảo (job 19: video gốc 51 giây, 40 cảnh)

| Bước | Số request | Token mỗi request |
|---|---|---|
| 5. gemini_reorder | 1 (+ tối đa 2 lần sửa) | khoảng 39.000 (video 22.000, suy nghĩ 12.000) |
| 8. gemini_voiceover | 1 mỗi video (3-4 video) | khoảng 8.000-9.000 |
| 9. gemini_tts | 1 mỗi video | khoảng 300-350 (chữ ~80 + âm thanh ~250); giới hạn 100 request/ngày mỗi model (Tier 1) |

## Hạn chế đã biết

- `remove_captions`: caption chữ trắng trên nền rất sáng, ít tương phản có thể bị sót; chữ overlay ở vị trí khác
  dòng caption không bị che (cố ý).
- `gemini_voiceover`: tên thương hiệu có thể được đọc không nhất quán (ví dụ "iKF"). Tốc độ đọc mặc định 3,9 âm tiết/giây
  đo với giọng Puck; thực tế dao động 3,4-4,7 tuỳ câu, nên `gemini_tts` có bước tăng tốc nhẹ để khớp.
- `gemini_tts`: ranh giới câu được đoán từ khoảng lặng (hoặc chia theo âm tiết khi không đoán được), nên phụ đề có thể
  lệch vài phần mười giây; bản dự phòng 3.1 là model preview, giọng có thể khác 3.8 dù cùng tên giọng.
- `render_videos`: bản ở `output/` vẫn còn caption và âm thanh gốc tiếng Trung; bản hoàn chỉnh ở `output/final/` thì không.
- `compose_videos`: tấm kính cố định theo câu/vùng che lớn nhất của video nên câu ngắn có thể có nhiều khoảng trống;
  chữ overlay lớn của video gốc (không nằm trên dòng caption) vẫn còn và có thể nằm sát tấm kính.
