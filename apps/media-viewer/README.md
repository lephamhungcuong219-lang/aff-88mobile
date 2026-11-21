# media-viewer

Trang xem, tải và xoá video final của một job n8n: `media/jobs/<workflowId>-<runId>/output/final/`.

```
https://preview.<N8N_HOST>/?id=DFWxtX6yNHTW20wS-55     (VPS, qua Caddy)
http://localhost:3000/?id=DFWxtX6yNHTW20wS-55          (local)
```

### Domain

- **Có tên miền (VPS + Caddy):** `Caddyfile` có sẵn site `preview.{$N8N_HOST}` → `media-viewer:3000`.
  Chỉ cần DNS cho `preview.<N8N_HOST>` trỏ về VPS. Ví dụ `N8N_HOST=example.com` → `https://preview.example.com/?id=...`.
- **Cloudflare Tunnel có tên (named tunnel, domain của bạn trên Cloudflare):** thêm một public hostname
  `preview.<domain>` → `http://localhost:3000` cạnh hostname của n8n.
- **Quick tunnel (`*.trycloudflare.com`) không làm được `preview.<host>`**: Cloudflare chỉ cấp đúng một tên ngẫu nhiên,
  không có subdomain con (DNS không tồn tại). Tạm thời chạy thêm một quick tunnel riêng cho trang này:
  `cloudflared tunnel --url http://localhost:3000` (sẽ ra một tên ngẫu nhiên khác, dùng `<tên đó>/?id=...`).

- Xem trước, tải về từng video hoặc các video đã chọn.
- Xoá một hoặc nhiều video final đã chọn.
- Xoá toàn bộ `media/jobs/<jobId>`; chỉ được phép khi job đã có `output/final` (tránh xoá job đang chạy dở).

## Chạy

```bash
pnpm install
pnpm dev          # API :3000 + Vite :5173 (mở http://localhost:5173/?id=...)
pnpm build && pnpm start   # một server :3000 phục vụ cả API và trang
```

Docker: service `media-viewer` trong `docker-compose.yml` ở thư mục gốc (`docker compose up -d --build media-viewer`).

Biến môi trường: `PORT` (3000), `HOST` (0.0.0.0), `MEDIA_ROOT` (mặc định `../../media`, trong Docker là `/data/media`),
`CLIENT_DIR` (mặc định `dist/client`). Phía client: `VITE_API_BASE` (mặc định `/api`) khi API nằm ở chỗ khác.

App có workspace pnpm riêng (`pnpm-workspace.yaml` trong thư mục này), không dính vào workspace n8n ở thư mục gốc.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/jobs/:id` | thông tin job + danh sách video final |
| GET | `/api/jobs/:id/videos/:name` | stream video (có Range); `?download=1` để tải về |
| DELETE | `/api/jobs/:id/videos` | body `{ "names": [...] }`, trả `{ deleted, missing }` |
| DELETE | `/api/jobs/:id` | xoá cả thư mục job (409 nếu chưa có `output/final`) |

Lỗi: `{ "error": { "code": "BAD_REQUEST" | "NOT_FOUND" | "CONFLICT" | "INTERNAL", "message" } }`.

## Cấu trúc và hướng mở rộng

```
src/
├── shared/jobs.ts                 kiểu DTO + kiểm tra job id / tên file, dùng chung server & client
├── server/
│   ├── modules/jobs/jobs.service.ts   nghiệp vụ trên filesystem, không phụ thuộc HTTP
│   ├── modules/jobs/jobs.routes.ts    lớp HTTP mỏng (Hono)
│   ├── http/                          file-response (Range, Web Response), error-handler
│   ├── common/errors.ts               AppError(code) -> status
│   └── app.ts / main.ts / config.ts
└── client/
    ├── api/                       client gọi API (http.ts, jobs.ts)
    ├── pages/job/job-page.ts      trang duy nhất hiện tại
    └── main.ts                    bảng route theo pathname
```

- **NestJS**: `JobsService` thành provider (`new JobsService(jobsRoot)` trong factory), `jobs.routes.ts` thành controller,
  `AppError` map trong một exception filter. `fileResponse` trả `Response` chuẩn Web; với Nest dùng `StreamableFile`.
- **Next.js**: route handler `app/api/jobs/[id]/...` gọi thẳng `JobsService` và `fileResponse`;
  `shared/` và `client/api/` dùng lại nguyên; trang job thành một page component.
- **Đăng nhập / phân quyền**: thêm middleware trước `app.route("/api", ...)` trong `app.ts`; module mới đặt trong `server/modules/`.
- **Chưa có auth**: trang public qua `preview.<N8N_HOST>`, ai có link (đoán được job id) đều xem và xoá được.
  Cổng 3000 trực tiếp chỉ mở ở `127.0.0.1`.
