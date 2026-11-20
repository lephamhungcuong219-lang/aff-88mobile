# Pipeline xử lý video Douyin (post_process)

Tài liệu mô tả flow hiện tại trên n8n: từ dữ liệu Douyin (Apify) đến các video ngắn đã dựng lại,
đã che caption tiếng Trung và có lời thoại tiếng Việt (chưa có giọng đọc).

Cập nhật: 2026-11-19.

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
6. remove_captions   che caption tiếng Trung trên các clip gốc
7. render_videos     ghép bản cuối (đã che caption) + bản proxy (gửi Gemini)
8. gemini_voiceover  Gemini viết lời thoại tiếng Việt cho từng video
9. gemini_tts        Gemini TTS đọc lời thoại (giọng Puck), có timestamp từng câu
10. compose_videos   ghép giọng đọc + phụ đề tiếng Việt trên tấm kính vào bản cuối
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
- **stdout** chỉ có kết quả (một dòng, riêng `render_videos` có thể hai dòng); **stderr** là nhật ký tiến trình
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
│   ├── <awemeId>_video_<N>.mp4       video cuối, đã che caption, 4K           [7]
│   └── final/<awemeId>_video_<N>.mp4 video hoàn chỉnh: giọng đọc + phụ đề    [10]
└── temp/
    └── <awemeId>/
        ├── sources/<N>.mp4           các cảnh, chất lượng gốc, còn caption    [3]
        ├── scenelog.csv / .json      danh sách cảnh + thời gian               [3]
        ├── runlog.txt                log của scenedetect                      [3]
        ├── compressed/<N>.mp4        bản nhỏ 480p có nhãn SCENE N             [4]
        ├── request.json              request gửi Gemini (lên kế hoạch)        [5]
        ├── gem_response.json         response gốc của Gemini                  [5]
        ├── reorder.json              kế hoạch các video + timestamp           [5]
        ├── clean/<N>.mp4             các cảnh đã che caption                  [6]
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

Tạo thư mục job cùng `input/`, `output/`, `temp/`.

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
| stdout | `dir:<output_dir>` / `fail` |

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
python -m post_process.compress_clips "{{ $json.stdout.trim().replace(/^dir:/, '') }}" "{{ $json.stdout.trim().replace(/^dir:/, '').replace(/\/sources$/, '/compressed') }}"
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
| stdout | `json:<reorder.json>` / `fail` (`--dry-run`: chỉ ghi `request.json`, stdout `json:<request.json>`) |

Quy tắc kiểm tra (sai thì gửi lỗi lại cho Gemini trong cùng cuộc trò chuyện, tối đa 3 lần):

- mỗi video 8-20 giây (mục tiêu 15 giây);
- trong một video mỗi cảnh xuất hiện tối đa một lần;
- mọi cảnh được dùng ít nhất một lần;
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

Tìm caption nhỏ (dòng chữ cố định của video) bằng OpenCV, không dùng AI, rồi che bằng tấm kính mờ:
hình chữ nhật bo góc, căn giữa khung hình, chữ bên dưới được xoá trước khi làm mờ.
Mọi chữ nằm trên dòng caption đều bị che (kể cả chữ to hơn caption); chữ overlay ở vị trí khác được giữ nguyên.

| | |
|---|---|
| Lệnh | `python -m post_process.remove_captions <sources_dir> <output_dir> [--mode panel\|fill] [--preview DIR]` |
| Input | `sources_dir`: `.../sources`; `output_dir`: `.../clean` (phải khác `sources_dir`) |
| Output | `<output_dir>/<N>.mp4` (clip có caption thì encode lại, không có thì copy nguyên), `<output_dir>/captions.json` |
| stdout | `dir:<output_dir>` / `fail` |
| Thời gian | khoảng 3-4 phút cho 40 clip 4K (đặt timeout node từ 10 phút) |

`--mode fill`: chỉ xoá chữ, không có tấm kính. `--preview DIR`: lưu ảnh trước/sau của từng clip.

`captions.json`: `{"mode", "caption_line": {"center_y", "height"}, "clips": {"<N>.mp4": {"caption", "method", "glyphs_removed", "detection", "box": {"x","y","w","h"}, "start", "end", "hits", "mask_share"}}}`.

Lệnh n8n (sau `gemini_reorder`, stdout `json:.../reorder.json`):
```
python -m post_process.remove_captions "{{ $json.stdout.trim().replace(/^json:/, '').replace(/\/reorder\.json$/, '/sources') }}" "{{ $json.stdout.trim().replace(/^json:/, '').replace(/\/reorder\.json$/, '/clean') }}"
```

### 7. render_videos

Ghép các cảnh theo `reorder.json`. Một lần chạy xuất cả **bản cuối** (từ `clean/`, chất lượng gốc) và **bản proxy**
(từ `sources/`, còn caption, 480x854) để gửi Gemini. Hai bản có cùng cảnh, cùng thứ tự, cùng số khung hình.
Mỗi video ghép đủ mọi cảnh, nguyên độ dài (không phải bản tóm tắt).

| | |
|---|---|
| Lệnh | `python -m post_process.render_videos <reorder_json> <clips_dir> <output_dir> [--proxy-dir DIR] [--proxy-clips DIR] [--proxy-height 854] [--prefix TEXT] [--no-audio] [--height PX]` |
| Input | `reorder_json`: `.../reorder.json`; `clips_dir`: `.../clean`; `output_dir`: `<job>/output`; `--proxy-dir`: `.../proxy`; `--proxy-clips`: `.../sources`; `--prefix`: `<awemeId>_` |
| Output | `<output_dir>/<prefix>video_<N>.mp4`, `<proxy_dir>/proxy_<prefix>video_<N>.mp4` |
| stdout | dòng 1 `dir:<output_dir>`, dòng 2 `proxy:<proxy_dir>` (khi có `--proxy-dir`) / `fail` |

Kiểm tra: thiếu clip nào (ở cả hai thư mục) thì báo ngay trước khi render; thời lượng video lệch quá 0,5 giây so với
`total_duration` thì `fail`; tham số đầu không phải file `reorder.json`, hoặc `--proxy-dir` trùng thư mục khác, thì `fail`.
Video được encode lại (H.264 CRF 20 + AAC 192k) để tránh lỗi âm thanh ở chỗ nối. `--no-audio` bỏ âm thanh gốc.

Lệnh n8n (sau `remove_captions`, stdout `dir:.../clean`):
```
python -m post_process.render_videos "{{ $json.stdout.trim().replace(/^dir:/, '').replace(/\/clean$/, '') }}/reorder.json" "{{ $json.stdout.trim().replace(/^dir:/, '') }}" "{{ $json.stdout.trim().replace(/^dir:/, '').split('/').slice(0, -3).join('/') }}/output" --proxy-dir "{{ $json.stdout.trim().replace(/^dir:/, '').replace(/\/clean$/, '') }}/proxy" --proxy-clips "{{ $json.stdout.trim().replace(/^dir:/, '').replace(/\/clean$/, '') }}/sources" --prefix "{{ $json.stdout.trim().replace(/^dir:/, '').split('/').slice(-2, -1)[0] }}_"
```

Lấy từng dòng stdout ở node sau: `{{ $json.stdout.match(/^dir:(.*)$/m)[1] }}`, `{{ $json.stdout.match(/^proxy:(.*)$/m)[1] }}`.

### 8. gemini_voiceover

Gửi từng proxy (base64, mỗi video một request) cùng thông tin trong `reorder.json` để Gemini viết lời thoại
tiếng Việt: năng động, bán hàng, xưng "mình" - gọi người xem "bạn". Chưa có timestamp.

| | |
|---|---|
| Lệnh | `python -m post_process.gemini_voiceover <reorder_json> <proxy_dir> [--output FILE] [--video N ...] [--rate 3.9] [--dry-run]` |
| Input | `reorder_json`: `.../reorder.json`; `proxy_dir`: `.../proxy` (tìm file `*video_<N>.mp4`) |
| Môi trường | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| Output | `voiceover.json`, `gem_voiceover_response.json` (cạnh `reorder.json`, hoặc cạnh `--output`) |
| stdout | `json:<voiceover.json>` / `fail` (`--dry-run`: ghi `voiceover_request.json`, không gọi Gemini) |

Kiểm tra (sai thì gửi lại cho Gemini, tối đa 3 lần): tổng âm tiết của phần đọc nằm trong 70-100% của
`thời lượng × --rate` (mục tiêu 88%); mỗi câu phụ đề tối đa 36 ký tự; có `tts_style`.
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

Lệnh n8n (sau `render_videos`):
```
python -m post_process.gemini_voiceover "{{ $json.stdout.match(/^proxy:(.*)$/m)[1].replace(/\/proxy$/, '') }}/reorder.json" "{{ $json.stdout.match(/^proxy:(.*)$/m)[1] }}"
```

### 9. gemini_tts

Đọc lời thoại bằng Gemini TTS (`gemini-3.8-flash-tts`, giọng mặc định **Puck**). Gemini TTS không trả timestamp,
nên **mỗi câu được đọc riêng** (cùng giọng, cùng `tts_style` của video): độ dài file âm thanh của câu chính là thời lượng
của câu. Script cắt khoảng lặng hai đầu mỗi câu, ghép lại (0,15s đầu + 0,12s giữa các câu) và tính `start`/`end`.

| | |
|---|---|
| Lệnh | `python -m post_process.gemini_tts <voiceover_json> <output_dir> [--voice Puck] [--model M] [--video N ...] [--gap 0.12] [--lead 0.15] [--max-speedup 1.15]` |
| Input | `voiceover_json`: `.../voiceover.json`; `output_dir`: `.../voice` |
| Môi trường | `GEMINI_API_KEY`; `GEMINI_TTS_MODEL` (mặc định `gemini-3.8-flash-tts`); `GEMINI_TTS_RPM` (mặc định 10) |
| Output | `<output_dir>/voice_video_<N>.wav` (24 kHz mono, bắt đầu từ giây 0 của video); `tts.json` cạnh `voiceover.json` |
| stdout | `json:<tts.json>` / `fail` (video nào lỗi thì bỏ, các video còn lại vẫn được ghi) |
| Thời gian | khoảng 2,5 phút cho 4 video (bị giới hạn 10 request/phút của gói Tier 1) |

Khớp độ dài: giọng đọc dài hơn video (trừ 0,1s cuối) thì tăng tốc giữ nguyên cao độ (`atempo`), tối đa `--max-speedup`
(1,15x); vẫn dài quá thì video đó `fail`, stderr gợi ý chạy lại `gemini_voiceover --video N --rate <tốc độ đo được>`.
Request được giãn cách theo `GEMINI_TTS_RPM`; gặp 429 thì chờ đúng thời gian server yêu cầu.

`tts.json`:
```
{
  "model", "voice", "sample_rate", "audio_dir",
  "videos": [{
    "video_index", "audio_file", "video_duration", "speech_duration",
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
Phụ đề: chữ trắng đậm (Noto Sans Bold) có bóng, hiện đúng `start`-`end` của từng câu trong `tts.json`,
đặt giữa **một tấm kính mờ cố định cho cả video**: hình chữ nhật bo góc, căn giữa khung hình, trên dòng caption.
Tấm kính đủ rộng cho câu dài nhất và phủ hết các tấm kính cũ của `remove_captions` trong các cảnh của video
(lấy từ `captions.json` + `reorder.json`), nên không đổi kích thước ở chỗ nối cảnh và không lộ mép tấm kính cũ.
Độ rộng chữ được đo bằng chính font; câu quá dài thì thu nhỏ chữ (không dưới 1,8% chiều cao), tấm kính rộng thì
phóng chữ lên (tối đa 3,2%).

| | |
|---|---|
| Lệnh | `python -m post_process.compose_videos <tts_json> <videos_dir> <output_dir> [--reorder FILE] [--captions FILE] [--music FILE] [--music-volume 0.12] [--original-volume 0] [--tint dark\|light] [--no-subtitles] [--video N ...]` |
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
| `.env` | `GEMINI_API_KEY`, `GEMINI_MODEL` (đọc bởi bước 5, 8) |
| `Dockerfile` | Alpine 3.24 (`apk`), ffmpeg, font Noto, Python 3.14, OpenCV/numpy từ `apk`, `uv`, `post_process` copy vào `/opt/post_process` |
| `docker-compose.yml` | `./media` -> `/data/media`, `./n8n_data` -> `/home/node/.n8n` |
| Hằng số đầu mỗi script | ngưỡng quy tắc (thời lượng video, số lần dùng cảnh, kích thước caption, tốc độ đọc…) |

## Chi phí Gemini tham khảo (job 19: video gốc 51 giây, 40 cảnh)

| Bước | Số request | Token mỗi request |
|---|---|---|
| 5. gemini_reorder | 1 (+ tối đa 2 lần sửa) | khoảng 39.000 (video 22.000, suy nghĩ 12.000) |
| 8. gemini_voiceover | 1 mỗi video (3-4 video) | khoảng 8.000-9.000 |
| 9. gemini_tts | 1 mỗi câu (5-7 câu mỗi video) | không đo; giới hạn 10 request/phút (Tier 1) |

## Hạn chế đã biết

- `remove_captions`: caption chữ trắng trên nền rất sáng, ít tương phản có thể bị sót; chữ overlay ở vị trí khác
  dòng caption không bị che (cố ý).
- `gemini_voiceover`: tên thương hiệu có thể được đọc không nhất quán (ví dụ "iKF"). Tốc độ đọc mặc định 3,9 âm tiết/giây
  đo với giọng Puck; thực tế dao động 3,4-4,7 tuỳ câu, nên `gemini_tts` có bước tăng tốc nhẹ để khớp.
- `gemini_tts`: mỗi câu đọc riêng nên ngữ điệu giữa các câu có thể không liền mạch hoàn toàn.
- `render_videos`: bản ở `output/` vẫn còn âm thanh gốc (lời tiếng Trung); bản hoàn chỉnh ở `output/final/` thì không.
- `compose_videos`: tấm kính phải phủ các tấm kính cũ nên có thể rộng hơn nhiều so với câu ngắn;
  chữ overlay lớn của video gốc (không nằm trên dòng caption) vẫn còn và có thể nằm sát tấm kính.
