import type { ErrorHandler } from "hono";
import type { ContentfulStatusCode } from "hono/utils/http-status";
import type { ApiErrorBody, ApiErrorCode } from "../../shared/jobs.js";
import { AppError } from "../common/errors.js";

const STATUS: Record<ApiErrorCode, ContentfulStatusCode> = {
  BAD_REQUEST: 400,
  NOT_FOUND: 404,
  CONFLICT: 409,
  INTERNAL: 500,
};

export const errorHandler: ErrorHandler = (err, c) => {
  if (err instanceof AppError) {
    return c.json<ApiErrorBody>({ error: { code: err.code, message: err.message } }, STATUS[err.code]);
  }
  console.error(err);
  return c.json<ApiErrorBody>({ error: { code: "INTERNAL", message: "Lỗi máy chủ" } }, 500);
};
