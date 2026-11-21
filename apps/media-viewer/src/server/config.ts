import path from "node:path";

export interface AppConfig {
  port: number;
  host: string;
  /** Thư mục chứa các job: `<jobsRoot>/<jobId>`. */
  jobsRoot: string;
  /** Bản build của client (vite build). */
  clientDir: string;
}

/** Đường dẫn tương đối tính từ thư mục chạy lệnh (apps/media-viewer). */
export function loadConfig(env: NodeJS.ProcessEnv = process.env): AppConfig {
  const mediaRoot = path.resolve(env.MEDIA_ROOT ?? "../../media");
  return {
    port: Number(env.PORT ?? 3000),
    host: env.HOST ?? "0.0.0.0",
    jobsRoot: path.join(mediaRoot, "jobs"),
    clientDir: path.resolve(env.CLIENT_DIR ?? "dist/client"),
  };
}
