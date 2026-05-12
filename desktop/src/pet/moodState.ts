export type AnimationSpec = {
  key: string;
  src: string;
  frames: number;
  duration: string;
};

const animations: Record<string, AnimationSpec> = {
  idle: {
    key: "idle",
    src: "/pets/danling/idle.png",
    frames: 6,
    duration: "1.2s"
  },
  running: {
    key: "running",
    src: "/pets/danling/running-right.png",
    frames: 8,
    duration: "0.72s"
  },
  "running-left": {
    key: "running-left",
    src: "/pets/danling/running-left.png",
    frames: 8,
    duration: "0.72s"
  },
  waiting: {
    key: "waiting",
    src: "/pets/danling/idle.png",
    frames: 6,
    duration: "1.5s"
  },
  failed: {
    key: "failed",
    src: "/pets/danling/idle.png",
    frames: 6,
    duration: "1.8s"
  }
};

export function animationFor(key: string | undefined): AnimationSpec {
  return animations[key || "running"] ?? animations.running;
}
