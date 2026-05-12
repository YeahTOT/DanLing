export type DanLingConfig = {
  pet_name: string;
  primary_score: string | null;
  primary_loss: string | null;
  loss_window: number;
  no_improve_patience: number;
  stale_seconds: number;
  high_memory_ratio: number;
  hot_util_percent: number;
  watch_interval: number;
  baseline_score: number | null;
  sota_score: number | null;
  realm_thresholds: Record<string, number> | number[] | null;
  source: string;
  data_path: string | null;
};

export type DanLingState = {
  pet_name: string;
  pet_mood: string;
  furnace_state: string;
  realm: {
    name: string;
    rank: number;
    progress: number | null;
  };
  metric: null | {
    epoch: number | null;
    train_loss: number | null;
    score: number | null;
    score_name: string | null;
  };
  events: Array<{
    severity: string;
    title: string;
    message: string;
  }>;
  desktop: {
    animation: string;
    status_label: string;
  };
};

export type ConfigPayload = {
  path: string;
  config: DanLingConfig;
  values: Record<string, string>;
};

const API_BASE = import.meta.env.VITE_DANLING_API ?? "http://127.0.0.1:8765";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getState(): Promise<DanLingState> {
  return requestJson<DanLingState>("/api/state");
}

export function getConfig(): Promise<ConfigPayload> {
  return requestJson<ConfigPayload>("/api/config");
}

export function saveConfig(values: Record<string, string>): Promise<ConfigPayload> {
  return requestJson<ConfigPayload>("/api/config", {
    method: "PUT",
    body: JSON.stringify(values)
  });
}

export async function suggestConfig(
  payload: Record<string, string>
): Promise<Record<string, string>> {
  const response = await requestJson<{ suggested: Record<string, string> }>(
    "/api/config/suggest",
    {
      method: "POST",
      body: JSON.stringify(payload)
    }
  );
  return response.suggested;
}
