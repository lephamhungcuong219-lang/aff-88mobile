import type { ApiErrorBody, ApiErrorCode } from "../../shared/jobs";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: ApiErrorCode,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Gốc của API. Sau này có backend riêng (NestJS...) thì đặt VITE_API_BASE. */
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "/api";

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body !== undefined) headers.set("Content-Type", "application/json");

  const res = await fetch(API_BASE + path, { ...init, headers });
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as ApiErrorBody | null;
    throw new ApiError(res.status, body?.error.code ?? "INTERNAL", body?.error.message ?? `HTTP ${res.status}`);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}
