export function detectionCategory(d: any): string {
  const name = String(d.class_name || d.label || "").toLowerCase();
  return name.includes("logo") ? "logo" : /cachet|stamp/.test(name) ? "stamp" : name.includes("signature") ? "signature" : "";
}

export function detectionScore(d: any): number {
  const score = Number(d.confidence ?? d.detection_confidence ?? 0);
  return Number.isFinite(score) ? Math.max(0, Math.min(1, score > 1 ? score / 100 : score)) : 0;
}

function box(d: any): { values: number[]; space: string } | null {
  const raw = d.normalized_bbox || d.bbox;
  const values = (Array.isArray(raw) ? raw : raw && [raw.x1, raw.y1, raw.x2, raw.y2])?.map(Number);
  if (!values || values.length !== 4 || !values.every(Number.isFinite) || values[2] <= values[0] || values[3] <= values[1]) return null;
  return { values, space: d.normalized_bbox ? "normalized" : "pixels" };
}

export function visibleDetections(detections: any[]): any[] {
  const kept: any[] = [];
  for (const candidate of [...detections].filter(d => detectionCategory(d)).sort((a, b) => detectionScore(b) - detectionScore(a))) {
    const a = box(candidate);
    const duplicate = a && kept.some(previous => {
      if (detectionCategory(previous) !== detectionCategory(candidate)
          || (previous.page_number ?? previous.page ?? 1) !== (candidate.page_number ?? candidate.page ?? 1)) return false;
      const b = box(previous);
      if (!b || a.space !== b.space) return false;
      const [x1, y1, x2, y2] = a.values;
      const [u1, v1, u2, v2] = b.values;
      const intersection = Math.max(0, Math.min(x2, u2) - Math.max(x1, u1)) * Math.max(0, Math.min(y2, v2) - Math.max(y1, v1));
      const areaA = (x2 - x1) * (y2 - y1), areaB = (u2 - u1) * (v2 - v1);
      return intersection / (areaA + areaB - intersection) >= 0.5;
    });
    if (!duplicate) kept.push(candidate);
  }
  return kept;
}
