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

/** Đoạn dài nhất giống nhau giữa hai danh sách cảnh (cùng cảnh, cùng thứ tự; một cảnh lẻ cũng là một đoạn), tính bằng giây. */
function longestShared(a, b) {
  const numbers = b.map((s) => s.scene_number);
  let best = 0;
  for (let i = 0; i < a.length; i++) {
    const j = numbers.indexOf(a[i].scene_number);
    if (j < 0) continue;
    let seconds = 0;
    for (let k = 0; i + k < a.length && j + k < b.length && a[i + k].scene_number === numbers[j + k]; k++) {
      seconds += a[i + k].duration;
    }
    best = Math.max(best, seconds);
  }
  return best;
}

/** Đoạn dài nhất giống video gốc: các cảnh số liền nhau đặt cạnh nhau (một cảnh lẻ cũng là một đoạn), tính bằng giây. */
function longestSource(scenes) {
  let best = 0;
  let seconds = 0;
  scenes.forEach((s, i) => {
    const next = i > 0 && s.scene_number === scenes[i - 1].scene_number + 1;
    seconds = next ? seconds + s.duration : s.duration;
    best = Math.max(best, seconds);
  });
  return best;
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
  lines.push(`- ${plan.video_count} video`, `- ${target[0].toUpperCase()}${target.slice(1)}`);
}

const ls = stats.long_scenes;
lines.push('', ls ? `**Cảnh dài ≥ ${ls.threshold_s}s: ${ls.long_count}/${ls.scene_count}**` : '**Cảnh dài:** không có dữ liệu');
if (ls?.long_count) {
  for (const n of ls.scenes) lines.push(`- Cảnh ${n}: ${sec(ls.durations?.[String(n)] ?? 0)}`);
  lines.push(`- Tổng: ${sec(ls.long_seconds)} / ${sec(ls.total_seconds)} video gốc`);
}

// Cảnh quay liền quá dài đã được chia (số "Cảnh dài" ở trên tính SAU khi chia).
const shots = stats.scene_detection?.long_shots_split ?? [];
if (shots.length) {
  lines.push('', `**Cảnh dài đã chia nhỏ: ${shots.length}** (dài hơn ${stats.scene_detection.max_scene_s}s)`);
  for (const x of shots) {
    const into = x.scenes?.length ? `cảnh ${x.scenes.join(', ')}` : `${x.parts} phần`;
    const cut = x.trimmed > 0 ? ` (bỏ ${sec(x.trimmed)})` : '';
    lines.push(`- ${sec(x.end - x.start)} → ${into}${cut}`);
  }
}

const videos = plan?.videos ?? [];
let maxSource = null;
let maxShared = null;
if (!videos.length) {
  lines.push('', '**Đoạn trùng:** không có dữ liệu');
} else {
  lines.push('', '**Đoạn trùng dài nhất với video gốc**');
  for (const v of videos) {
    const s = longestSource(v.scenes);
    maxSource = Math.max(maxSource ?? 0, s);
    lines.push(`- Video ${v.video_index}: ${sec(s)}`);
  }
  if (videos.length > 1) {
    lines.push('', '**Đoạn trùng dài nhất giữa các video**');
    for (const v of videos) {
      lines.push(`- Video ${v.video_index}`);
      for (const o of videos) {
        if (o === v) continue;
        const s = longestShared(v.scenes, o.scenes);
        maxShared = Math.max(maxShared ?? 0, s);
        lines.push(`  - Với video ${o.video_index}: ${sec(s)}`); // thụt 2 dấu cách: bullet cấp 2 của Discord
      }
    }
  }
}

let content = lines.join('\n');
if (content.length > MAX_CONTENT) content = `${content.slice(0, MAX_CONTENT - 30)}\n… (bị cắt bớt)`;

// Màu cho embed nếu dùng: đỏ khi có đoạn vượt mức tối đa trong .env, còn lại xanh; xám khi không có dữ liệu.
const over = (value, max) => value !== null && max > 0 && value > max + 1e-6;
const color = maxSource === null ? '#95a5a6'
  : over(maxSource, rules?.source_run_max_s) || over(maxShared, rules?.max_shared_run_s) ? '#e74c3c' : '#2ecc71';

return {
  json: {
    content,
    color,
    job_id: jobId,
    aweme_id: awemeId,
    max_source_run_s: maxSource,
    max_shared_run_s: maxShared,
    stats_json: statsJson,
  },
};
