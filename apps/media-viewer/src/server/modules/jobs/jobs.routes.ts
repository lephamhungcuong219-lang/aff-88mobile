import { Hono } from "hono";
import type { DeleteVideosRequest } from "../../../shared/jobs.js";
import { badRequest } from "../../common/errors.js";
import { fileResponse } from "../../http/file-response.js";
import type { JobsService } from "./jobs.service.js";

/**
 * REST API của job, mount tại /api/jobs.
 *   GET    /:id                 thông tin job + danh sách video final
 *   GET    /:id/videos/:name    stream video (hỗ trợ Range); ?download=1 để tải về
 *   DELETE /:id/videos          body { names: string[] }: xoá các video final đã chọn
 *   DELETE /:id                 xoá toàn bộ thư mục job
 */
export function jobsRoutes(jobs: JobsService) {
  return new Hono()
    .get("/:id", async (c) => c.json(await jobs.getJob(c.req.param("id"))))

    .get("/:id/videos/:name", async (c) => {
      const video = await jobs.getVideo(c.req.param("id"), c.req.param("name"));
      return fileResponse({
        ...video,
        method: c.req.method,
        range: c.req.header("range"),
        downloadName: c.req.query("download") ? video.name : undefined,
      });
    })

    .delete("/:id/videos", async (c) => {
      const body = await c.req.json<Partial<DeleteVideosRequest>>().catch(() => {
        throw badRequest("Body phải là JSON");
      });
      return c.json(await jobs.deleteVideos(c.req.param("id"), body.names));
    })

    .delete("/:id", async (c) => {
      await jobs.deleteJob(c.req.param("id"));
      return c.body(null, 204);
    });
}
