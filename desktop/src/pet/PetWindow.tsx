import type { CSSProperties } from "react";

import { DanLingState } from "../api/client";
import { animationFor } from "./moodState";

type PetWindowProps = {
  state: DanLingState | null;
  loading: boolean;
  error: string | null;
};

export function PetWindow({ state, loading, error }: PetWindowProps) {
  const animation = animationFor(state?.desktop.animation);
  const metric = state?.metric;
  const progress = state?.realm.progress;

  return (
    <section className="pet-stage" data-tauri-drag-region>
      <div className="speech">
        <strong>{state?.pet_name ?? "DanLing"}</strong>
        <span>{error ?? state?.desktop.status_label ?? "连接桌面 sidecar 中"}</span>
      </div>
      <div
        className={`pet-sprite ${animation.key}`}
        style={
          {
            backgroundImage: `url(${animation.src})`,
            "--frames": animation.frames,
            "--duration": animation.duration,
            "--sprite-width": `${animation.frames * 100}%`
          } as CSSProperties
        }
        aria-label={`DanLing ${animation.key}`}
      />
      <div className="state-strip">
        <span>{state?.pet_mood ?? (loading ? "loading" : "offline")}</span>
        <span>{state?.realm.name ?? "未入境"}</span>
        <span>
          {typeof progress === "number" ? `${Math.round(progress * 100)}%` : "等待指标"}
        </span>
      </div>
      <div className="metric-grid">
        <div>
          <span>Epoch</span>
          <b>{metric?.epoch ?? "-"}</b>
        </div>
        <div>
          <span>Score</span>
          <b>{formatNumber(metric?.score)}</b>
        </div>
        <div>
          <span>Loss</span>
          <b>{formatNumber(metric?.train_loss)}</b>
        </div>
      </div>
    </section>
  );
}

function formatNumber(value: number | null | undefined): string {
  return typeof value === "number" ? value.toFixed(4) : "-";
}
