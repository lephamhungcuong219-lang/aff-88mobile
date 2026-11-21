/**
 * Kiểu dữ liệu và quy tắc dùng chung cho cả server và client.
 * Không import gì từ Node hay DOM ở đây để file này dùng được ở mọi nơi
 * (Hono hiện tại, NestJS hoặc Next.js sau này).
 */

/** Job id = `<workflowId>-<runId>`, ví dụ `DFWxtX6yNHTW20wS-55`. */
export interface JobRef {
  id: string;
  workflowId: string;
  runId: string;
}

const JOB_ID_PATTERN = /^([A-Za-z0-9]{1,64})-(\d{1,20})$/;

export function parseJobId(raw: string | null | undefined): JobRef | null {
  if (!raw) return null;
  const match = JOB_ID_PATTERN.exec(raw.trim());
  if (!match) return null;
  const [id, workflowId, runId] = match as unknown as [string, string, string];
  return { id, workflowId, runId };
}

export const VIDEO_EXTENSIONS = [".mp4", ".webm", ".mov", ".m4v"] as const;

/** Tên file video hợp lệ: không có thư mục, không bắt đầu bằng dấu chấm, đuôi video. */
export function isVideoFileName(name: string): boolean {
  if (!name || name.length > 255) return false;
  if (name.startsWith(".") || /[\/\0]/.test(name)) return false;
  const lower = name.toLowerCase();
  return VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

export interface FinalVideoDto {
  name: string;
  /** byte */
  size: number;
  /** ISO 8601 */
  modifiedAt: string;
}

export interface JobDto extends JobRef {
  /** Đã có thư mục output/final, tức pipeline đã tới bước dựng video hoàn chỉnh. */
  hasFinal: boolean;
  videos: FinalVideoDto[];
}

export interface DeleteVideosRequest {
  names: string[];
}

export interface DeleteVideosResult {
  deleted: string[];
  /** Tên không còn tồn tại (đã bị xoá trước đó). */
  missing: string[];
}

export interface ApiErrorBody {
  error: {
    code: ApiErrorCode;
    message: string;
  };
}

export type ApiErrorCode = "BAD_REQUEST" | "NOT_FOUND" | "CONFLICT" | "INTERNAL";
