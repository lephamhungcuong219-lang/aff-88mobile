# n8n-media

n8n (kèm ffmpeg, bật Execute Command) chạy bằng Docker, dùng Supabase PostgreSQL làm database.
Trên VPS có Caddy làm reverse proxy và tự cấp HTTPS (Let's Encrypt).

```
Internet ──443──> Caddy ──> n8n:5678 ──SSL──> Supabase Postgres (schema n8n)
                              │
                              ├── ./n8n_data  -> /home/node/.n8n
                              └── ./media     -> /data/media   (file cho ffmpeg)
```

## Chạy local

```bash
cp .env.example .env   # điền Supabase + encryption key, dùng khối "Local"
docker compose up -d --build
docker compose logs -f n8n
```

Mở http://localhost:5678

## Deploy lên VPS (Ubuntu/Debian)

### 1. Chuẩn bị
- VPS tối thiểu 2 vCPU / 2 GB RAM (ffmpeg cần thêm).
- Trỏ **A record** của domain (ví dụ `n8n.example.com`) về IP của VPS.
- Nên chọn VPS ở **cùng region với Supabase** (hiện tại là `eu-west-1`, Ireland) để giảm độ trễ DB.

### 2. Cài Docker và mở firewall
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # đăng xuất / đăng nhập lại
sudo ufw allow OpenSSH && sudo ufw allow 80/tcp && sudo ufw allow 443/tcp && sudo ufw allow 443/udp && sudo ufw enable
sudo timedatectl set-ntp true
```

### 3. Đưa code lên VPS
```bash
git clone <repo-url> ~/n8n-media    # hoặc scp/rsync thư mục này (không cần node_modules)
cd ~/n8n-media
```
Không cần tự tạo `n8n_data/`, `media/`: mỗi lần `docker compose up`, service `init-dirs` tạo `media/jobs` và
`n8n_data` nếu chưa có và giao cho user `node` (uid 1000) của container, trước khi n8n khởi động.
Nếu `n8n_data/` được copy từ máy khác (có sẵn file bên trong), chown cả thư mục một lần:
`sudo chown -R 1000:1000 n8n_data`.

### 4. Tạo `.env`
```bash
cp .env.example .env && nano .env
```
- `COMPOSE_PROFILES=proxy`
- Điền thông tin Supabase.
- `N8N_ENCRYPTION_KEY` **phải giống** key đang dùng ở local (DB đã mã hoá credentials bằng key đó).
- Dùng khối "VPS": `N8N_HOST`, `N8N_WEBHOOK_URL`, `N8N_EDITOR_BASE_URL` = domain của bạn.

### 5. Chạy
```bash
docker compose up -d --build
docker compose ps
docker compose logs -f n8n caddy
```
Mở `https://<domain>`. Lần đầu Caddy cần ~30 giây để lấy chứng chỉ.

### Cập nhật n8n
Đổi `N8N_VERSION` trong `.env`, rồi:
```bash
docker compose up -d --build
```

## Lưu ý quan trọng

- **Không chạy local và VPS cùng lúc** trên cùng một Supabase DB: cả hai đều là "leader",
  workflow có Schedule/Trigger sẽ chạy hai lần. Tắt local (`docker compose down`) khi VPS đã chạy.
- Local và VPS phải dùng **cùng `N8N_VERSION`** vì n8n tự migrate schema DB khi khởi động.
- Lỗi `timeout exceeded when trying to connect` / `Cannot use a pool after calling end on the pool`:
  do kết nối tới Supabase chậm (~1,7s/kết nối từ VN tới eu-west-1). Đã tăng pool qua các biến
  `DB_POSTGRESDB_*` trong `docker-compose.yml`; có thể chỉnh thêm trong `.env`. Supabase Session
  pooler giới hạn số kết nối, đừng đặt `DB_POSTGRESDB_POOL_SIZE` quá ~10.
- Backup: `n8n_data/` (chứa `config` có encryption key) và `media/`. Workflow/credentials nằm trong Supabase.
- Dọn job cũ: chỉ xoá bên trong `media/jobs/` (`rm -rf media/jobs/*`), **đừng xoá `media/`** khi container đang
  chạy: mount `/data/media` bị hỏng và mọi job báo lỗi cho tới khi tạo lại thư mục hoặc `docker compose up -d`.
