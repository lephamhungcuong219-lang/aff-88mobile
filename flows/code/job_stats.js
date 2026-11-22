// n8n Code node — mode "Run Once for Each Item", language JavaScript.
// Gom thống kê của split_video (long_scenes) và gemini_reorder (overlap, source_overlap) thành một JSON.
// Cấu trúc giống <job>/temp/<awemeId>/stats.json (file này là bản đầy đủ, luôn có trên đĩa).
//
// Đặt node ở NHÁNH RIÊNG sau node "Nấu các cảnh" (hoặc cuối workflow): Code node thay $json,
// nếu chèn giữa chuỗi thì node "Làm mờ caption của clips" sẽ mất stdout của gemini_reorder.

const SPLIT_NODE = 'Cắt cảnh trong video'; // node chạy split_video
const REORDER_NODE = 'Nấu các cảnh';       // node chạy gemini_reorder

const splitOut = $(SPLIT_NODE).item.json.stdout ?? '';
const reorderOut = $(REORDER_NODE).item.json.stdout ?? '';

// long_scenes:<số cảnh >= 3s>_<tổng số cảnh>[-<các cảnh, nối bằng _>]
const ls = splitOut.match(/^long_scenes:(\d+)_(\d+)(?:-([\d_]+))?\r?$/m);
const longScenes = ls && {
  long_count: Number(ls[1]),
  scene_count: Number(ls[2]),
  scenes: ls[3] ? ls[3].split('_').map(Number) : [],
};

// overlap:<các video>-<các cảnh>-<số giây>   (một dòng cho mỗi đoạn cảnh liên tiếp bị trùng)
const runs = [...reorderOut.matchAll(/^overlap:([\d_]+)-([\d_]+)-([\d.]+)\r?$/gm)].map((m) => ({
  videos: m[1].split('_').map(Number),
  scenes: m[2].split('_').map(Number),
  seconds: Number(m[3]),
}));

// overlap_video:<video>-<số giây bị trùng>_<thời lượng video>   (mỗi video một dòng)
const perVideo = [...reorderOut.matchAll(/^overlap_video:(\d+)-([\d.]+)_([\d.]+)\r?$/gm)].map((m) => {
  const shared = Number(m[2]);
  const total = Number(m[3]);
  return {
    video_index: Number(m[1]),
    shared_seconds: shared,
    total_duration: total,
    shared_ratio: total ? Math.round((shared / total) * 1000) / 1000 : 0,
  };
});

// So với video gốc:
// source_overlap:<video>-<các cảnh>-<số giây>   (một dòng cho mỗi đoạn giống hệt video gốc)
const sourceRuns = [...reorderOut.matchAll(/^source_overlap:(\d+)-([\d_]+)-([\d.]+)\r?$/gm)].map((m) => ({
  video: Number(m[1]),
  scenes: m[2].split('_').map(Number),
  seconds: Number(m[3]),
}));

// source_overlap_video:<video>-<số giây trong các đoạn>_<thời lượng video>_<đoạn dài nhất>   (mỗi video một dòng)
const sourcePerVideo = [...reorderOut.matchAll(/^source_overlap_video:(\d+)-([\d.]+)_([\d.]+)_([\d.]+)\r?$/gm)].map((m) => {
  const shared = Number(m[2]);
  const total = Number(m[3]);
  return {
    video_index: Number(m[1]),
    shared_seconds: shared,
    total_duration: total,
    shared_ratio: total ? Math.round((shared / total) * 1000) / 1000 : 0,
    longest_seconds: Number(m[4]),
  };
});

const reorderJson = (reorderOut.match(/^json:(.*)$/m) || [])[1]?.trim() ?? null;
const statsJson = reorderJson ? reorderJson.replace(/\/[^/]+$/, '/stats.json') : null;
const jobId = reorderJson ? (reorderJson.match(/\/jobs\/([^/]+)\//) || [])[1] ?? null : null;

return {
  json: {
    job_id: jobId,
    stats_json: statsJson,
    reorder_json: reorderJson,
    long_scenes: longScenes,
    overlaps: perVideo.length ? { runs, per_video: perVideo } : null,
    source_overlaps: sourcePerVideo.length ? { runs: sourceRuns, per_video: sourcePerVideo } : null,
    summary: {
      long_scene_count: longScenes?.long_count ?? null,
      scene_count: longScenes?.scene_count ?? null,
      overlap_run_count: runs.length,
      videos_with_overlap: perVideo.filter((v) => v.shared_seconds > 0).map((v) => v.video_index),
      max_shared_ratio: perVideo.length ? Math.max(...perVideo.map((v) => v.shared_ratio)) : null,
      source_run_count: sourceRuns.length,
      max_source_run_s: sourcePerVideo.length ? Math.max(...sourcePerVideo.map((v) => v.longest_seconds)) : null,
    },
  },
};
