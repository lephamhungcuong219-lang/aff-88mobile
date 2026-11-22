import type { Stats } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import {
  isVideoFileName,
  parseJobId,
  REPORT_FILE_NAME,
  type DeleteVideosResult,
  type FinalVideoDto,
  type JobDto,
  type JobRef,
} from "../../../shared/jobs.js";
import { badRequest, conflict, notFound } from "../../common/errors.js";

/** Báo cáo chỉ vài KB; file lớn hơn thì không phải báo cáo, không đọc vào bộ nhớ. */
const MAX_REPORT_BYTES = 256 * 1024;

export interface VideoFile {
  name: string;
  path: string;
  size: number;
  modifiedAt: Date;
}

/**
 * Nghiệp vụ trên thư mục job: `<jobsRoot>/<jobId>/output/final/*.mp4`.
 * Không biết gì về HTTP, để sau này bọc lại thành provider NestJS
 * hoặc gọi thẳng từ route handler của Next.js.
 */
export class JobsService {
  constructor(private readonly jobsRoot: string) {}

  async getJob(rawId: string): Promise<JobDto> {
    const ref = this.ref(rawId);
    await this.assertJobExists(ref);
    const finalDir = this.finalDir(ref);
    const hasFinal = await isDirectory(finalDir);
    const videos = hasFinal ? await this.listVideos(finalDir) : [];
    const report = hasFinal ? await this.readReport(finalDir) : null;
    return { ...ref, hasFinal, videos, report };
  }

  async getVideo(rawId: string, name: string): Promise<VideoFile> {
    const filePath = this.videoPath(this.ref(rawId), name);
    const stat = await lstatOrNull(filePath);
    if (!stat?.isFile()) throw notFound(`Không tìm thấy video "${name}"`);
    return { name, path: filePath, size: stat.size, modifiedAt: stat.mtime };
  }

  async getReport(rawId: string): Promise<VideoFile> {
    const filePath = path.join(this.finalDir(this.ref(rawId)), REPORT_FILE_NAME);
    const stat = await lstatOrNull(filePath);
    if (!stat?.isFile()) throw notFound("Job chưa có báo cáo stats.txt");
    return { name: REPORT_FILE_NAME, path: filePath, size: stat.size, modifiedAt: stat.mtime };
  }

  async deleteVideos(rawId: string, names: unknown): Promise<DeleteVideosResult> {
    const ref = this.ref(rawId);
    if (!Array.isArray(names) || names.length === 0 || !names.every((n) => typeof n === "string")) {
      throw badRequest("names phải là mảng tên file, không rỗng");
    }
    await this.assertJobExists(ref);

    // Kiểm tra hết tên trước khi xoá file nào, để một tên sai không làm xoá dở dang.
    const targets = [...new Set(names as string[])].map((name) => ({ name, path: this.videoPath(ref, name) }));
    const result: DeleteVideosResult = { deleted: [], missing: [] };
    for (const target of targets) {
      const stat = await lstatOrNull(target.path);
      if (!stat?.isFile()) {
        result.missing.push(target.name);
        continue;
      }
      await fs.rm(target.path);
      result.deleted.push(target.name);
    }
    return result;
  }

  /** Xoá toàn bộ `<jobsRoot>/<jobId>`. Chỉ cho phép khi job đã có output/final. */
  async deleteJob(rawId: string): Promise<void> {
    const ref = this.ref(rawId);
    await this.assertJobExists(ref);
    if (!(await isDirectory(this.finalDir(ref)))) {
      throw conflict("Job chưa có output/final (có thể vẫn đang chạy), không xoá");
    }
    await fs.rm(this.jobDir(ref), { recursive: true, force: true });
  }

  private ref(rawId: string): JobRef {
    const ref = parseJobId(rawId);
    if (!ref) throw badRequest("Job id không hợp lệ, cần dạng <workflowId>-<runId>");
    return ref;
  }

  private jobDir(ref: JobRef): string {
    // parseJobId chỉ cho chữ, số và một dấu "-", nên không thể thoát khỏi jobsRoot.
    return path.join(this.jobsRoot, ref.id);
  }

  private finalDir(ref: JobRef): string {
    return path.join(this.jobDir(ref), "output", "final");
  }

  private videoPath(ref: JobRef, name: string): string {
    const finalDir = this.finalDir(ref);
    const filePath = path.join(finalDir, name);
    if (!isVideoFileName(name) || path.dirname(filePath) !== finalDir) {
      throw badRequest(`Tên video không hợp lệ: "${name}"`);
    }
    return filePath;
  }

  private async assertJobExists(ref: JobRef): Promise<void> {
    if (!(await isDirectory(this.jobDir(ref)))) throw notFound(`Không tìm thấy job ${ref.id}`);
  }

  private async readReport(finalDir: string): Promise<string | null> {
    const filePath = path.join(finalDir, REPORT_FILE_NAME);
    const stat = await lstatOrNull(filePath);
    if (!stat?.isFile() || stat.size > MAX_REPORT_BYTES) return null;
    try {
      const text = await fs.readFile(filePath, "utf8");
      return text.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n").trim() || null;
    } catch {
      return null; // bị xoá giữa chừng; thiếu báo cáo không được làm hỏng trang
    }
  }

  private async listVideos(finalDir: string): Promise<FinalVideoDto[]> {
    const entries = await fs.readdir(finalDir, { withFileTypes: true });
    const videos: FinalVideoDto[] = [];
    for (const entry of entries) {
      if (!entry.isFile() || !isVideoFileName(entry.name)) continue;
      const stat = await lstatOrNull(path.join(finalDir, entry.name));
      if (!stat) continue; // bị xoá giữa chừng
      videos.push({ name: entry.name, size: stat.size, modifiedAt: stat.mtime.toISOString() });
    }
    return videos.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  }
}

async function lstatOrNull(p: string): Promise<Stats | null> {
  try {
    return await fs.lstat(p);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw err;
  }
}

async function isDirectory(p: string): Promise<boolean> {
  return (await lstatOrNull(p))?.isDirectory() ?? false;
}
