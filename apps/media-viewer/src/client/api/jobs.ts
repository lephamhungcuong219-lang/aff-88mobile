import type { DeleteVideosRequest, DeleteVideosResult, JobDto } from "../../shared/jobs";
import { API_BASE, request } from "./http";

const jobPath = (jobId: string) => `/jobs/${encodeURIComponent(jobId)}`;

export const jobsApi = {
  get: (jobId: string) => request<JobDto>(jobPath(jobId)),

  videoUrl: (jobId: string, name: string) => `${API_BASE}${jobPath(jobId)}/videos/${encodeURIComponent(name)}`,

  downloadUrl: (jobId: string, name: string) => `${jobsApi.videoUrl(jobId, name)}?download=1`,

  deleteVideos: (jobId: string, names: string[]) =>
    request<DeleteVideosResult>(`${jobPath(jobId)}/videos`, {
      method: "DELETE",
      body: JSON.stringify({ names } satisfies DeleteVideosRequest),
    }),

  deleteJob: (jobId: string) => request<void>(jobPath(jobId), { method: "DELETE" }),
};
