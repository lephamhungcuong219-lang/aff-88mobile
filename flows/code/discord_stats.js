// n8n Code node — mode "Run Once for Each Item", language JavaScript.
// Đọc stats.json (+ reorder.json) của job và dựng nội dung tin nhắn cho node Discord.
// Không tự gửi: node Discord phía sau dùng {{ $json.content }} (hoặc $json.embed).
// Nối nhánh riêng từ node "Nấu các cảnh" (gemini_reorder). Cần NODE_FUNCTION_ALLOW_BUILTIN=fs,path.

const fs = require('fs');
const path = require('path');

const REORDER_NODE = 'Nấu các cảnh'; // node chạy gemini_reorder
const MAX_CONTENT = 2000;            // giới hạn ký tự một tin nhắn Discord

const readJson = (file) => {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
};
const sec = (s) => `${Number(s).toFixed(2)}s`;
const pct = (r) => `${Math.round(r * 100)}%`;

/** [21..47, 1, 2, 3, 9, 8] -> "21–47, 1–3, 9, 8": dãy tăng liên tiếp từ 3 cảnh được rút gọn, giữ thứ tự. */
function sceneList(scenes) {
  const parts = [];
  for (let i = 0; i < scenes.length;) {
    let j = i;
    while (j + 1 < scenes.length && scenes[j + 1] === scenes[j] + 1) j++;
    if (j - i >= 2) parts.push(`${scenes[i]}–${scenes[j]}`);
    else parts.push(...scenes.slice(i, j + 1).map(String));
    i = j + 1;
  }
  return parts.join(', ');
}

// --- Đọc dữ liệu -------------------------------------------------------------
const stdout = $(REORDER_NODE).item.json.stdout ?? '';
const reorderJson = (stdout.match(/^json:(.*)$/m) || [])[1]?.trim();
if (!reorderJson) throw new Error(`Không thấy dòng json: trong stdout của "${REORDER_NODE}"`);

const dir = path.dirname(reorderJson);
const statsJson = path.join(dir, 'stats.json');
const jobId = (reorderJson.match(/\/jobs\/([^/]+)\//) || [])[1] ?? '?';
const awemeId = path.basename(dir);
const stats = readJson(statsJson);
if (!stats) throw new Error(`Không đọc được ${statsJson}`);
const plan = readJson(reorderJson);

// --- Cấu trúc ------------------------------------------------------------------
const lines = [`**Job ${jobId}** · awemeId ${awemeId}`];

const rules = plan?.rules;
if (plan) {
  const target = rules?.full_source ? `dài bằng video gốc (${sec(rules.target_video_s)})`
    : `mục tiêu ${rules?.target_video_s}s (${rules?.min_video_s}-${rules?.max_video_s}s)`;
  lines.push(`${plan.video_count} video · ${target}`);
}

const ls = stats.long_scenes;
lines.push('', ls ? `**Cảnh dài ≥ ${ls.threshold_s}s: ${ls.long_count}/${ls.scene_count}**` : '**Cảnh dài:** không có dữ liệu');
if (ls?.long_count) {
  lines.push(ls.scenes.map((n) => `${n} (${sec(ls.durations?.[String(n)] ?? 0)})`).join(', '));
  lines.push(`Tổng ${sec(ls.long_seconds)} / ${sec(ls.total_seconds)} video gốc`);
}

const ov = stats.overlaps;
const maxRatio = ov ? Math.max(0, ...ov.per_video.map((v) => v.shared_ratio)) : null;
if (!ov) {
  lines.push('', '**Đoạn trùng:** không có dữ liệu');
} else {
  lines.push('', '**Trùng theo video**');
  for (const v of ov.per_video) {
    lines.push(`Video ${v.video_index}: ${sec(v.shared_seconds)} / ${sec(v.total_duration)} · ${pct(v.shared_ratio)}`);
  }
  lines.push('', `**Đoạn cảnh liên tiếp bị trùng: ${ov.runs.length}**`);
  for (const r of ov.runs) {
    lines.push(`Video ${r.videos.join(', ')} · cảnh ${sceneList(r.scenes)} (${r.scenes.length} cảnh) · ${sec(r.seconds)}`);
  }
}

// So các video output với video gốc: đoạn giống hệt gốc = các cảnh số liền nhau đặt cạnh nhau.
const so = stats.source_overlaps;
if (so) {
  const limit = so.max_s > 0 ? ` (mục tiêu ${so.target_s}s, tối đa ${so.max_s}s)` : '';
  lines.push('', `**Trùng với video gốc**${limit}`);
  for (const v of so.per_video) {
    lines.push(`Video ${v.video_index}: ${sec(v.shared_seconds)} / ${sec(v.total_duration)} · ${pct(v.shared_ratio)}`
      + ` · dài nhất ${sec(v.longest_seconds)} (cảnh ${sceneList(v.longest_scenes)})`);
  }
  lines.push('', `**Đoạn giống hệt video gốc: ${so.runs.length}**`);
  const mark = { above_target: ' · trên mục tiêu', too_long: ' · QUÁ DÀI' };
  for (const r of so.runs) {
    lines.push(`Video ${r.video} · cảnh ${sceneList(r.scenes)} (${r.scenes.length} cảnh) · ${sec(r.seconds)}${mark[r.level] ?? ''}`);
  }
}

let content = lines.join('\n');
if (content.length > MAX_CONTENT) content = `${content.slice(0, MAX_CONTENT - 30)}\n… (bị cắt bớt)`;

// Màu cho embed nếu dùng: xanh < 20% trùng, vàng < 50%, đỏ từ 50%.
const color = maxRatio === null ? '#95a5a6' : maxRatio < 0.2 ? '#2ecc71' : maxRatio < 0.5 ? '#f1c40f' : '#e74c3c';

return {
  json: {
    content,
    color,
    job_id: jobId,
    aweme_id: awemeId,
    max_shared_ratio: maxRatio,
    max_source_run_s: so ? Math.max(0, ...so.per_video.map((v) => v.longest_seconds)) : null,
    stats_json: statsJson,
  },
};
