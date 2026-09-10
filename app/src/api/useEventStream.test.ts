import { describe, expect, it, vi } from "vitest";
import { INITIAL, reducer, type AppEvent, type RunState } from "./useEventStream";

// Real event shapes service/pipeline_events.py (parsing) and
// service/run_manager.py (terminal states) emit, exercised against the reducer
// they previously reached only through an untyped `payload: any` — this is the
// unlocked-by-typing test the remediation pass added.

function run(state: RunState, action: AppEvent): RunState {
  return reducer(state, action);
}

describe("run_started", () => {
  it("resets to a running state carrying the new prompt", () => {
    const dirty: RunState = { ...INITIAL, status: "complete", logLines: ["stale"] };
    const next = run(dirty, {
      type: "run_started",
      payload: {
        prompt: "a wooden crate",
        poly_budget: "medium",
        mcp_mode: false,
        use_concept_pipeline: true,
        started_at: 1000,
      },
    });
    expect(next.status).toBe("running");
    expect(next.prompt).toBe("a wooden crate");
    expect(next.logLines).toEqual([]);
    expect(next.startedAt).toBe(1_000_000);
  });
});

describe("stage", () => {
  const stage = (number: number, title: string): AppEvent => ({
    type: "stage",
    payload: { number, title, label: title.toUpperCase() },
  });

  it("tracks the current stage and appends unseen stage numbers once", () => {
    let state = INITIAL;
    state = run(state, { type: "stage", payload: { number: 1, title: "Plan", label: "PLAN" } });
    state = run(state, { type: "stage", payload: { number: 1, title: "Plan (retry)", label: "PLAN" } });
    expect(state.stages).toHaveLength(1);
    expect(state.currentStage?.title).toBe("Plan (retry)");
  });

  // The inspector's rail shows how long each station took. The backend only ever
  // announces starts, so these are the reducer's own arrival-to-arrival times and
  // the closing rules below are the whole of that measurement.
  it("closes the running stage when the next one is announced", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(0));
    let state = run(INITIAL, stage(1, "Plan"));
    vi.advanceTimersByTime(4000);
    state = run(state, stage(2, "Naming"));

    const plan = state.stages.find((s) => s.number === 1)!;
    expect(plan.elapsedMs).toBe(4000);
    expect(plan.openSince).toBeNull();
    // The newly announced stage is the open one and has no duration yet.
    expect(state.stages.find((s) => s.number === 2)!.openSince).toBe(4000);
    vi.useRealTimers();
  });

  it("accumulates time and counts passes when the loop re-enters a stage", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(0));
    let state = run(INITIAL, stage(55, "Visual"));
    vi.advanceTimersByTime(5000);
    state = run(state, stage(6, "Topology")); // closes the first visual pass
    vi.advanceTimersByTime(1000);
    state = run(state, stage(55, "Visual")); // loop comes back round
    vi.advanceTimersByTime(3000);
    state = run(state, stage(7, "Quality"));

    const visual = state.stages.find((s) => s.number === 55)!;
    expect(visual.passes).toBe(2);
    expect(visual.elapsedMs).toBe(8000); // 5s + 3s, not just the last pass
    expect(state.stages).toHaveLength(3);
    vi.useRealTimers();
  });

  it("closes the final stage on a terminal event, idempotently", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(0));
    let state = run(INITIAL, stage(8, "Reference"));
    vi.advanceTimersByTime(2000);
    state = run(state, {
      type: "complete",
      payload: {
        run_id: "r1",
        asset_name: "crate",
        glb_path: null,
        preview_path: null,
        visual_score: 8,
      },
    });
    expect(state.stages[0].elapsedMs).toBe(2000);

    // A second terminal event must not add the intervening time again — without
    // the openSince guard, "failed" arriving after "complete" would double it.
    vi.advanceTimersByTime(9000);
    state = run(state, { type: "failed", payload: { reason: "late" } });
    expect(state.stages[0].elapsedMs).toBe(2000);
    vi.useRealTimers();
  });
});

describe("score", () => {
  it("appends to history and tracks the latest", () => {
    let state = INITIAL;
    state = run(state, { type: "score", payload: { value: 5, kind: "run", source: "candidate 1" } });
    state = run(state, { type: "score", payload: { value: 8, kind: "ok", source: "candidate 2" } });
    expect(state.scores).toHaveLength(2);
    expect(state.latestScore?.value).toBe(8);
  });
});

describe("log_line", () => {
  it("caps accumulated lines at 500, keeping the most recent", () => {
    let state = INITIAL;
    for (let i = 0; i < 510; i++) {
      state = run(state, { type: "log_line", payload: { text: `line ${i}` } });
    }
    expect(state.logLines).toHaveLength(500);
    expect(state.logLines[0]).toBe("line 10");
    expect(state.logLines[499]).toBe("line 509");
  });
});

describe("cancelled", () => {
  it("keeps the completed payload when an asset was produced before cancel", () => {
    const next = run(INITIAL, {
      type: "cancelled",
      payload: {
        run_id: "20260101_120000_crate",
        asset_name: "crate",
        glb_path: "out/final/crate/crate.glb",
        preview_path: "out/final/crate/crate_preview.png",
        visual_score: 6,
      },
    });
    expect(next.status).toBe("cancelled");
    expect(next.completed?.run_id).toBe("20260101_120000_crate");
    expect(next.completed?.visual_score).toBe(6);
  });

  it("has no completed payload when cancelled before any asset existed", () => {
    const next = run(INITIAL, {
      type: "cancelled",
      payload: { reason: "Cancelled before an asset was produced" },
    });
    expect(next.status).toBe("cancelled");
    expect(next.completed).toBeNull();
    expect(next.message).toBe("Cancelled before an asset was produced");
  });
});

describe("terminal failure states", () => {
  it("rejected carries the reason as the message", () => {
    const next = run(INITIAL, { type: "rejected", payload: { reason: "unsafe prompt" } });
    expect(next.status).toBe("rejected");
    expect(next.message).toBe("unsafe prompt");
  });

  it("failed carries the reason as the message", () => {
    const next = run(INITIAL, { type: "failed", payload: { reason: "Pipeline reported failure" } });
    expect(next.status).toBe("failed");
    expect(next.message).toBe("Pipeline reported failure");
  });

  it("error carries the message", () => {
    const next = run(INITIAL, { type: "error", payload: { message: "RuntimeError: boom" } });
    expect(next.status).toBe("error");
    expect(next.message).toBe("RuntimeError: boom");
  });
});

describe("memory approval", () => {
  it("pending sets awaiting_approval and stores the request", () => {
    const next = run(INITIAL, {
      type: "memory_approval_pending",
      payload: {
        prompt: "a crate", asset_name: "crate", run_id: "r1",
        blueprint: null, preview_path: null, asset_path: null,
      },
    });
    expect(next.status).toBe("awaiting_approval");
    expect(next.approval?.run_id).toBe("r1");
  });

  it("resolved returns to running and clears the request", () => {
    const pending = run(INITIAL, {
      type: "memory_approval_pending",
      payload: {
        prompt: "a crate", asset_name: "crate", run_id: "r1",
        blueprint: null, preview_path: null, asset_path: null,
      },
    });
    const next = run(pending, { type: "memory_approval_resolved", payload: { approved: true } });
    expect(next.status).toBe("running");
    expect(next.approval).toBeNull();
  });
});

describe("__reset", () => {
  it("returns to the initial state from anything", () => {
    const dirty: RunState = { ...INITIAL, status: "error", message: "boom" };
    expect(run(dirty, { type: "__reset", payload: {} })).toEqual(INITIAL);
  });
});
