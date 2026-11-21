import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";
import type { AppConfig } from "./config.js";
import { errorHandler } from "./http/error-handler.js";
import { jobsRoutes } from "./modules/jobs/jobs.routes.js";
import { JobsService } from "./modules/jobs/jobs.service.js";

/** Ghép các module. Module mới (auth, users...) thì route thêm vào `api`. */
export function createApp(config: AppConfig) {
  const jobsService = new JobsService(config.jobsRoot);

  const api = new Hono().route("/jobs", jobsRoutes(jobsService));
  api.notFound((c) => c.json({ error: { code: "NOT_FOUND", message: "Không có API này" } }, 404));

  const app = new Hono();
  app.onError(errorHandler);
  app.route("/api", api);

  // Trang do Vite build ra. Hiện chỉ có /?id=<jobId>; trang mới thì thêm path vào PAGES.
  const PAGES = ["/"];
  app.use("/assets/*", serveStatic({ root: config.clientDir }));
  for (const page of PAGES) app.get(page, serveStatic({ root: config.clientDir, path: "index.html" }));

  return app;
}
