import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent } from "react";
import {
  ConfigPayload,
  DanLingState,
  getConfig,
  getState,
  saveConfig,
  suggestConfig
} from "./api/client";
import { PetWindow } from "./pet/PetWindow";

type Panel = "state" | "config";
type ConfigTab = "manual" | "auto" | "datasource";

const fields = [
  ["pet_name", "丹灵名称"],
  ["primary_score", "主指标"],
  ["primary_loss", "主 loss"],
  ["baseline_score", "Baseline"],
  ["sota_score", "SOTA"],
  ["realm_threshold_炼器期", "炼器期阈值"],
  ["realm_threshold_筑基期", "筑基期阈值"],
  ["realm_threshold_结丹期", "结丹期阈值"],
  ["realm_threshold_元婴期", "元婴期阈值"],
  ["realm_threshold_化神期", "化神期阈值"],
  ["watch_interval", "刷新间隔"],
  ["loss_window", "loss 窗口"],
  ["no_improve_patience", "无提升耐心"],
  ["stale_seconds", "过期秒数"],
  ["high_memory_ratio", "高显存比例"],
  ["hot_util_percent", "高利用率"],
  ["source", "数据源"],
  ["data_path", "数据路径"]
] as const;

const sourceOptions = [
  ["auto", "自动检测"],
  ["csv", "Ultralytics results.csv"],
  ["ultralytics-log", "Ultralytics log/txt"],
  ["tensorboard", "TensorBoard event/logdir"]
] as const;

export function App() {
  const [state, setState] = useState<DanLingState | null>(null);
  const [config, setConfig] = useState<ConfigPayload | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [panel, setPanel] = useState<Panel>("state");
  const [tab, setTab] = useState<ConfigTab>("manual");
  const [autoPath, setAutoPath] = useState("");
  const [autoMetric, setAutoMetric] = useState("mAP50");
  const [autoSource, setAutoSource] = useState("csv");
  const [message, setMessage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const triedStartingApi = useRef(false);

  useEffect(() => {
    let cancelled = false;
    async function refreshState() {
      try {
        const next = await getState();
        if (!cancelled) {
          setState(next);
          setError(null);
        }
      } catch (exc) {
        if (!triedStartingApi.current) {
          triedStartingApi.current = true;
          await startApiSidecar();
          return;
        }
        if (!cancelled) setError(exc instanceof Error ? exc.message : String(exc));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    refreshState();
    const timer = window.setInterval(refreshState, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    getConfig()
      .then((payload) => {
        setConfig(payload);
        setValues(payload.values);
      })
      .catch((exc) => setMessage(exc instanceof Error ? exc.message : String(exc)));
  }, []);

  const eventText = useMemo(() => {
    const event = state?.events?.[0];
    return event ? `${event.title}: ${event.message}` : "训练状态平稳";
  }, [state]);

  async function startDrag(event: MouseEvent) {
    const target = event.target as HTMLElement;
    if (target.closest("button,input,select,textarea")) return;
    try {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      await getCurrentWindow().startDragging();
    } catch {
      return;
    }
  }

  async function openConfigTui() {
    try {
      const { Command } = await import("@tauri-apps/plugin-shell");
      await Command.create("danling-config-tui").spawn();
      setMessage("已启动 danling config tui");
    } catch (exc) {
      setMessage(exc instanceof Error ? exc.message : "当前环境无法启动 config tui");
    }
  }

  async function startApiSidecar() {
    try {
      const { Command } = await import("@tauri-apps/plugin-shell");
      await Command.create("danling-api", [
        "api",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
        "--no-hardware"
      ]).spawn();
      setMessage("已启动 DanLing API sidecar");
    } catch {
      return;
    }
  }

  async function closeWindow() {
    try {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      await getCurrentWindow().close();
    } catch {
      window.close();
    }
  }

  function updateValue(key: string, value: string) {
    setValues((current) => ({ ...current, [key]: value }));
  }

  async function handleSave(nextValues = values) {
    try {
      const payload = await saveConfig(nextValues);
      setConfig(payload);
      setValues(payload.values);
      setMessage(`已保存 ${payload.path}`);
    } catch (exc) {
      setMessage(exc instanceof Error ? exc.message : String(exc));
    }
  }

  async function handleSuggest() {
    try {
      const suggested = await suggestConfig({
        path: autoPath,
        metric: autoMetric,
        source: autoSource
      });
      setValues((current) => ({ ...current, ...suggested }));
      setMessage("已生成 baseline/SOTA，可继续保存");
      setTab("manual");
    } catch (exc) {
      setMessage(exc instanceof Error ? exc.message : String(exc));
    }
  }

  async function saveDatasource() {
    await handleSave({
      ...values,
      source: values.source || "auto",
      data_path: values.data_path || ""
    });
  }

  return (
    <main className="app-shell" onMouseDown={startDrag} data-tauri-drag-region>
      <nav className="topbar">
        <button className={panel === "state" ? "active" : ""} onClick={() => setPanel("state")}>
          状态
        </button>
        <button className={panel === "config" ? "active" : ""} onClick={() => setPanel("config")}>
          配置
        </button>
        <button onClick={openConfigTui}>TUI</button>
        <button onClick={closeWindow}>退出</button>
      </nav>

      {panel === "state" ? (
        <>
          <PetWindow state={state} loading={loading} error={error} />
          <p className="event-line">{eventText}</p>
        </>
      ) : (
        <section className="config-panel">
          <div className="tabbar">
            <button className={tab === "manual" ? "active" : ""} onClick={() => setTab("manual")}>
              手动
            </button>
            <button className={tab === "auto" ? "active" : ""} onClick={() => setTab("auto")}>
              自动生成
            </button>
            <button
              className={tab === "datasource" ? "active" : ""}
              onClick={() => setTab("datasource")}
            >
              数据源
            </button>
          </div>

          {tab === "manual" && (
            <form className="config-form" onSubmit={(event) => event.preventDefault()}>
              {fields.map(([key, label]) => (
                <label key={key}>
                  <span>{label}</span>
                  <input value={values[key] ?? ""} onChange={(event) => updateValue(key, event.target.value)} />
                </label>
              ))}
              <button className="primary" type="button" onClick={() => handleSave()}>
                保存配置
              </button>
            </form>
          )}

          {tab === "auto" && (
            <div className="config-form compact">
              <label>
                <span>数据源</span>
                <select value={autoSource} onChange={(event) => setAutoSource(event.target.value)}>
                  {sourceOptions.slice(1).map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>历史路径</span>
                <input value={autoPath} onChange={(event) => setAutoPath(event.target.value)} />
              </label>
              <label>
                <span>关心指标</span>
                <input value={autoMetric} onChange={(event) => setAutoMetric(event.target.value)} />
              </label>
              <button className="primary" type="button" onClick={handleSuggest}>
                生成 baseline/SOTA
              </button>
            </div>
          )}

          {tab === "datasource" && (
            <div className="config-form compact">
              <label>
                <span>默认数据源</span>
                <select value={values.source ?? "auto"} onChange={(event) => updateValue("source", event.target.value)}>
                  {sourceOptions.map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>默认路径</span>
                <input value={values.data_path ?? ""} onChange={(event) => updateValue("data_path", event.target.value)} />
              </label>
              <button className="primary" type="button" onClick={saveDatasource}>
                保存数据源
              </button>
            </div>
          )}

          <p className="message">{message || config?.path || "读取配置中"}</p>
        </section>
      )}
    </main>
  );
}
