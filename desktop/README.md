# DanLing Desktop

This is the first desktop pet shell from `docs/UI/桌面端开发计划.md`.

Full Chinese development, runtime, and packaging guide:
`docs/UI/桌面端开发文档.md`.

## Development

Start the Python sidecar:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run danling api --no-hardware
```

Start the web UI:

```bash
cd desktop
npm install
npm run dev
```

Run the Tauri shell:

```bash
cd desktop
npm run tauri -- dev
```

The pet window uses the sprite strips copied from `output/hatch-pet/danling/decoded`.
