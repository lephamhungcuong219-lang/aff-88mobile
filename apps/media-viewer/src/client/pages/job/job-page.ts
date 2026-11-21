import { parseJobId, type FinalVideoDto, type JobDto } from "../../../shared/jobs";
import { ApiError } from "../../api/http";
import { jobsApi } from "../../api/jobs";
import { formatBytes, formatDate, h } from "../../lib/dom";

/** Trang /?id=<workflowId>-<runId>: xem, tải và xoá video final của một job. */
export async function mountJobPage(root: HTMLElement, params: URLSearchParams): Promise<void> {
  const ref = parseJobId(params.get("id"));
  if (!ref) {
    root.replaceChildren(
      notice("Thiếu hoặc sai job id. URL cần có dạng /?id=<workflowId>-<runId>, ví dụ ?id=DFWxtX6yNHTW20wS-55."),
    );
    return;
  }
  document.title = `Job ${ref.id}`;
  root.replaceChildren(notice("Đang tải…"));

  try {
    new JobPage(root, await jobsApi.get(ref.id)).render();
  } catch (err) {
    root.replaceChildren(notice(errorMessage(err)));
  }
}

class JobPage {
  private videos: FinalVideoDto[];
  private readonly selected = new Set<string>();
  private busy = false;

  private readonly summary = h("p", { class: "meta" });
  private readonly status = h("p", { class: "status", role: "status", "aria-live": "polite" });
  private readonly grid = h("ul", { class: "grid" });
  private readonly selectAll = h("input", { type: "checkbox", onchange: () => this.toggleAll() });
  private readonly downloadBtn = h("button", { type: "button", onclick: () => this.downloadSelected() });
  private readonly deleteBtn = h("button", { type: "button", class: "danger", onclick: () => this.deleteSelected() });
  private readonly deleteJobBtn = h(
    "button",
    { type: "button", class: "danger", onclick: () => this.deleteJob() },
    "Xoá toàn bộ dữ liệu job",
  );

  constructor(
    private readonly root: HTMLElement,
    private readonly job: JobDto,
  ) {
    this.videos = job.videos;
  }

  render(): void {
    const { job } = this;
    this.root.replaceChildren(
      h(
        "header",
        {},
        h("h1", {}, `Job ${job.id}`),
        h("p", { class: "meta" }, `Workflow ${job.workflowId} · Run ${job.runId}`),
        this.summary,
      ),
      h(
        "div",
        { class: "toolbar" },
        h("label", {}, this.selectAll, " Chọn tất cả"),
        this.downloadBtn,
        this.deleteBtn,
      ),
      this.status,
      this.grid,
      h(
        "section",
        { class: "danger-zone" },
        h("h2", {}, "Xoá job"),
        h(
          "p",
          {},
          `Xoá vĩnh viễn thư mục media/jobs/${job.id} (input, output, temp và các video final). Không khôi phục được.`,
        ),
        job.hasFinal ? null : h("p", { class: "meta" }, "Job chưa có output/final (có thể vẫn đang chạy) nên chưa xoá được."),
        this.deleteJobBtn,
      ),
    );
    this.grid.replaceChildren(...this.videos.map((v) => this.card(v)));
    this.refresh();
  }

  private card(video: FinalVideoDto): HTMLLIElement {
    const id = `v-${video.name}`;
    return h(
      "li",
      { class: "card", "data-name": video.name },
      h("video", { src: jobsApi.videoUrl(this.job.id, video.name), controls: true, preload: "metadata", playsInline: true }),
      h(
        "div",
        { class: "card-body" },
        h(
          "label",
          { class: "name", for: id },
          h("input", {
            id,
            type: "checkbox",
            onchange: (e: Event) => this.toggle(video.name, (e.target as HTMLInputElement).checked),
          }),
          video.name,
        ),
        h("p", { class: "meta" }, `${formatBytes(video.size)} · ${formatDate(video.modifiedAt)}`),
        h("a", { href: jobsApi.downloadUrl(this.job.id, video.name), download: video.name }, "Tải về"),
      ),
    );
  }

  /** Cập nhật các phần phụ thuộc vào danh sách chọn, không vẽ lại video đang phát. */
  private refresh(): void {
    const count = this.selected.size;
    const total = this.videos.reduce((sum, v) => sum + v.size, 0);
    this.summary.textContent = this.videos.length
      ? `${this.videos.length} video final · ${formatBytes(total)}`
      : this.job.hasFinal
        ? "Thư mục final không có video nào."
        : "Job chưa có thư mục output/final.";

    this.selectAll.checked = count > 0 && count === this.videos.length;
    this.selectAll.indeterminate = count > 0 && count < this.videos.length;
    this.selectAll.disabled = this.busy || this.videos.length === 0;
    this.downloadBtn.textContent = `Tải đã chọn (${count})`;
    this.deleteBtn.textContent = `Xoá đã chọn (${count})`;
    this.downloadBtn.disabled = this.busy || count === 0;
    this.deleteBtn.disabled = this.busy || count === 0;
    this.deleteJobBtn.disabled = this.busy || !this.job.hasFinal;

    for (const box of this.grid.querySelectorAll<HTMLInputElement>("input[type=checkbox]")) {
      const name = box.closest<HTMLElement>(".card")?.dataset.name ?? "";
      box.checked = this.selected.has(name);
      box.disabled = this.busy;
    }
  }

  private toggle(name: string, on: boolean): void {
    if (on) this.selected.add(name);
    else this.selected.delete(name);
    this.refresh();
  }

  private toggleAll(): void {
    const selectAll = this.selected.size !== this.videos.length;
    this.selected.clear();
    if (selectAll) for (const v of this.videos) this.selected.add(v.name);
    this.refresh();
  }

  private async downloadSelected(): Promise<void> {
    // Mỗi file một lượt tải; trình duyệt có thể hỏi cho phép tải nhiều file.
    for (const name of this.selected) {
      h("a", { href: jobsApi.downloadUrl(this.job.id, name), download: name }).click();
      await new Promise((r) => setTimeout(r, 400));
    }
  }

  private async deleteSelected(): Promise<void> {
    const names = [...this.selected];
    if (!confirm(`Xoá vĩnh viễn ${names.length} video final?\n\n${names.join("\n")}`)) return;

    await this.run(async () => {
      const result = await jobsApi.deleteVideos(this.job.id, names);
      const gone = new Set([...result.deleted, ...result.missing]);
      this.videos = this.videos.filter((v) => !gone.has(v.name));
      for (const name of gone) {
        this.selected.delete(name);
        this.grid.querySelector(`[data-name="${CSS.escape(name)}"]`)?.remove();
      }
      this.setStatus(`Đã xoá ${result.deleted.length} video.`);
    });
  }

  private async deleteJob(): Promise<void> {
    const { id } = this.job;
    if (!confirm(`Xoá vĩnh viễn toàn bộ dữ liệu của job ${id}?\nThư mục media/jobs/${id} sẽ bị xoá hết.`)) return;

    await this.run(async () => {
      await jobsApi.deleteJob(id);
      this.root.replaceChildren(notice(`Đã xoá toàn bộ dữ liệu của job ${id}.`));
    });
  }

  private async run(action: () => Promise<void>): Promise<void> {
    this.busy = true;
    this.setStatus("Đang xử lý…");
    this.refresh();
    try {
      await action();
    } catch (err) {
      this.setStatus(errorMessage(err), true);
    } finally {
      this.busy = false;
      this.refresh();
    }
  }

  private setStatus(text: string, isError = false): void {
    this.status.textContent = text;
    this.status.classList.toggle("error", isError);
  }
}

function notice(text: string): HTMLElement {
  return h("p", { class: "notice" }, text);
}

function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return "Không kết nối được máy chủ.";
}
