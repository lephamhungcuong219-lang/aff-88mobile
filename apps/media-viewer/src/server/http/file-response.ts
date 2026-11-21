import { createReadStream } from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";

const MIME_TYPES: Record<string, string> = {
  ".mp4": "video/mp4",
  ".m4v": "video/mp4",
  ".webm": "video/webm",
  ".mov": "video/quicktime",
};

export interface FileResponseOptions {
  path: string;
  size: number;
  modifiedAt: Date;
  method: string;
  /** Header `Range` của request, để tua video. */
  range?: string | undefined;
  /** Có giá trị thì trả về dạng tải xuống với tên file này. */
  downloadName?: string | undefined;
}

/** Response chuẩn Web (dùng được cho Hono, Next.js route handler...), hỗ trợ một byte range. */
export function fileResponse(opts: FileResponseOptions): Response {
  const headers = new Headers({
    "Content-Type": MIME_TYPES[path.extname(opts.path).toLowerCase()] ?? "application/octet-stream",
    "Accept-Ranges": "bytes",
    "Last-Modified": opts.modifiedAt.toUTCString(),
    "Cache-Control": "no-cache",
  });
  if (opts.downloadName) headers.set("Content-Disposition", contentDisposition(opts.downloadName));

  if (opts.size === 0) {
    headers.set("Content-Length", "0");
    return new Response(null, { status: 200, headers });
  }

  let start = 0;
  let end = opts.size - 1;
  let status = 200;
  if (opts.range) {
    const parsed = parseRange(opts.range, opts.size);
    if (!parsed) {
      headers.set("Content-Range", `bytes */${opts.size}`);
      return new Response(null, { status: 416, headers });
    }
    [start, end] = parsed;
    status = 206;
    headers.set("Content-Range", `bytes ${start}-${end}/${opts.size}`);
  }
  headers.set("Content-Length", String(end - start + 1));

  if (opts.method === "HEAD") return new Response(null, { status, headers });
  const stream = Readable.toWeb(createReadStream(opts.path, { start, end })) as ReadableStream;
  return new Response(stream, { status, headers });
}

/** Chỉ hỗ trợ một khoảng: `bytes=a-b`, `bytes=a-`, `bytes=-n`. */
function parseRange(header: string, size: number): [number, number] | null {
  const match = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!match) return null;
  const rawStart = match[1] ?? "";
  const rawEnd = match[2] ?? "";

  if (rawStart === "") {
    const suffix = Number(rawEnd);
    if (rawEnd === "" || suffix === 0) return null;
    return [Math.max(size - suffix, 0), size - 1];
  }
  const start = Number(rawStart);
  const end = rawEnd === "" ? size - 1 : Math.min(Number(rawEnd), size - 1);
  return start <= end ? [start, end] : null;
}

function contentDisposition(name: string): string {
  const ascii = name.replace(/[^\x20-\x7e]|["\\]/g, "_");
  return `attachment; filename="${ascii}"; filename*=UTF-8''${encodeURIComponent(name)}`;
}
