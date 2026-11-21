import type { ApiErrorCode } from "../../shared/jobs.js";

/**
 * Lỗi nghiệp vụ, không phụ thuộc framework HTTP.
 * Lớp HTTP (Hono hôm nay, NestJS exception filter sau này) map `code` sang status.
 */
export class AppError extends Error {
  constructor(
    readonly code: ApiErrorCode,
    message: string,
  ) {
    super(message);
    this.name = "AppError";
  }
}

export const badRequest = (message: string) => new AppError("BAD_REQUEST", message);
export const notFound = (message: string) => new AppError("NOT_FOUND", message);
export const conflict = (message: string) => new AppError("CONFLICT", message);
