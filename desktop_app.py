from __future__ import annotations

import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

try:
    from core.gl_renderer import GLRenderer as _GLRenderer
    _GL_RENDERER_AVAILABLE = True
except Exception:
    _GLRenderer = None
    _GL_RENDERER_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageTk
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False


class _QueueWriter:
    def __init__(self, output_queue: queue.Queue[str]) -> None:
        self._output_queue = output_queue

    def write(self, text: str) -> int:
        if text:
            self._output_queue.put(text)
        return len(text)

    def flush(self) -> None:
        return


def _mono_font(size: int = 10) -> tuple[str, int]:
    for name in ("Cascadia Code", "Consolas", "Courier New"):
        try:
            tkfont.Font(family=name, size=size)
            return (name, size)
        except tk.TclError:
            continue
    return ("TkFixedFont", size)


def _sans_font(size: int, weight: str = "normal") -> tuple[str, int, str]:
    for name in ("Segoe UI", "Helvetica Neue", "Arial"):
        try:
            tkfont.Font(family=name, size=size, weight=weight)
            return (name, size, weight)
        except tk.TclError:
            continue
    return ("TkDefaultFont", size, weight)


THEME = {
    "bg": "#0E0E11",
    "root": "#09090B",
    "title": "#0B0B0E",
    "panel": "#0E0E11",
    "panel_deep": "#0D0D10",
    "viewport": "#0A0A0D",
    "viewport_deep": "#09090B",
    "input": "#101013",
    "input_alt": "#111114",
    "card": "#141417",
    "card_alt": "#18181C",
    "hover": "#1E1E23",
    "border": "#1F1F24",
    "border_alt": "#232329",
    "accent": "#8B5CF6",
    "accent_hover": "#A78BFA",
    "accent_dark": "#1E1633",
    "success": "#34D399",
    "success_soft": "#6EE7B7",
    "warning": "#FBBF24",
    "danger": "#F87171",
    "text": "#FAFAFA",
    "text_alt": "#F4F4F5",
    "muted": "#71717A",
    "muted_alt": "#A1A1AA",
    "dim": "#3F3F46",
    "button": "#1A1A1F",
    "button_hover": "#232329",
}


class ArgusDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ARGUS - Create")
        self.root.geometry("1920x1080")
        self.root.minsize(1280, 720)
        self.root.state("zoomed")

        self.output_queue: queue.Queue[str] = queue.Queue()
        self.is_running = False
        self.preview_photo: tk.PhotoImage | None = None
        self.preview_image_id: int | None = None
        self.preview_image_path: Path | None = None
        self.model_vertices = None
        self.model_faces = None
        self.model_face_colors = None
        self.model_face_normals = None
        self.view_mode: str = "wire"
        self.model_path: Path | None = None
        self.model_yaw = -0.65
        self.model_pitch = 0.35
        self.model_zoom = 1.0
        self._model_drag_last: tuple[int, int] | None = None
        self._view_btn_wire: tk.Label | None = None
        self._preview_scan_buffer = ""
        self._operator_event_ids: set[str] = set()
        self.nav_labels: dict[str, tk.Label] = {}
        self.sidebar_frame: ttk.Frame | None = None
        self.sidebar_drag_start: tuple[int, int] | None = None
        self.dashboard_frame: ttk.Frame | None = None
        self.projects_list_frame: tk.Frame | None = None
        self.viewport_frame: ttk.Frame | None = None
        self.right_panel: ttk.Frame | None = None
        self.project_thumbnails: list[tk.PhotoImage] = []
        self.live_log_path = Path("logs") / "argus_live.log"
        self.live_log_path.parent.mkdir(exist_ok=True)
        self.live_log_path.write_text("", encoding="utf-8")
        self.max_visible_log_lines = 2500

        self.mcp_var = tk.BooleanVar(value=False)
        self.concept_pipeline_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.last_prompt_var = tk.StringVar(value="")
        self.status_badge_var = tk.StringVar(value="IDLE")
        self.density_var = tk.StringVar(value="0 POLY")
        self.hierarchy_var = tk.StringVar(value="SEPARATE")
        self.asset_path_var = tk.StringVar(value="No asset loaded")
        self.stage_var = tk.StringVar(value="Ready to generate")
        self.model_name_var = tk.StringVar(value="No asset loaded")
        self.primitive_count_var = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value="Headless")
        self.quality_var = tk.StringVar(value="Balanced")
        self.format_var = tk.StringVar(value="GLB + FBX")
        self.sidebar_width = 350
        self.sidebar_min_width = 300
        self.sidebar_max_width = 540
        self.component_parts: list[str] = []
        self.graph_list_frame: tk.Frame | None = None
        self._nav_workbench_btn: tk.Label | None = None
        self._nav_dashboard_btn: tk.Label | None = None
        self._concept_toggle_lbl: tk.Label | None = None
        self.workbench_panel: tk.Frame | None = None
        self.dashboard_panel: tk.Frame | None = None
        self.left_panel: tk.Frame | None = None

        self._auto_rotate: bool = True
        self._rotate_btn: tk.Label | None = None
        self._rotation_after_id: str | None = None
        self._drag_was_rotating: bool = False
        self._view_pan_x: float = 0.0
        self._view_pan_y: float = 0.0

        self._ambient_strength: float  = 0.20
        self._bg_brightness:    float  = 1.0

        self._gl_renderer = None
        self._gpu_frame_photo: tk.PhotoImage | None = None
        self._gpu_frame_id: int | None = None
        self._timeline_expanded = True
        self._timeline_rows: list[tk.Frame] = []
        self._timeline_body: tk.Frame | None = None
        self._timeline_container: tk.Frame | None = None
        self._timeline_toggle_lbl: tk.Label | None = None
        self._run_started_at: datetime | None = None

        self._mono = _mono_font(10)
        self._sans_title = _sans_font(20, "normal")
        self._sans_heading = _sans_font(13, "bold")
        self._sans_sub = _sans_font(10, "normal")
        self._sans_btn = _sans_font(10, "normal")

        self._configure_styles()
        self._build_ui()
        self._show_generator_page()
        self._start_output_pump()
        self._boot_check()

    def _configure_styles(self) -> None:
        _BG       = THEME["root"]
        _PANEL    = THEME["panel_deep"]
        _TOP      = "#0B0B0E"
        _VIEWPORT = THEME["viewport_deep"]
        _CARD     = THEME["card"]
        _HOVER    = THEME["hover"]
        _ACCENT   = THEME["accent"]
        _TEXTPRI  = THEME["text_alt"]
        _TEXTSEC  = THEME["muted_alt"]
        _BTN      = THEME["button"]
        _BTNHOV   = THEME["button_hover"]

        self.root.configure(bg=_BG)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame",      background=_BG)
        style.configure("Top.TFrame",      background=_TOP)
        style.configure("Side.TFrame",     background=_PANEL)
        style.configure("Viewport.TFrame", background=_VIEWPORT)
        style.configure("Panel.TFrame",    background=_PANEL)
        style.configure("PanelInner.TFrame", background=_CARD)
        style.configure("Rail.TFrame",     background=_BG)
        style.configure("LeftPanel.TFrame", background=_PANEL)
        style.configure("GraphPanel.TFrame", background=_PANEL)

        style.configure("Brand.TLabel",    background=_PANEL, foreground=_TEXTPRI, font=self._sans_title)
        style.configure("BrandSub.TLabel", background=_PANEL, foreground=_ACCENT,  font=self._sans_sub)
        style.configure("CardTitle.TLabel",background=_PANEL, foreground=_TEXTPRI, font=self._sans_heading)
        style.configure("Muted.TLabel",    background=_PANEL, foreground=_TEXTSEC, font=self._sans_sub)
        style.configure("RailTitle.TLabel",background=_PANEL, foreground=_TEXTSEC, font=(*self._sans_btn[:2], "bold"))
        style.configure("RailMuted.TLabel",background=_PANEL, foreground=_TEXTSEC, font=self._sans_sub)
        style.configure("Footer.TLabel",   background=_BG,    foreground=_TEXTSEC, font=self._sans_sub)

        style.configure("TCheckbutton", background=_PANEL, foreground=_TEXTPRI, font=self._sans_sub)
        style.map("TCheckbutton", background=[("active", _PANEL)])

        style.configure(
            "Primary.TButton",
            background=_ACCENT, foreground="#FFFFFF",
            padding=(18, 12), font=(*self._sans_btn[:2], "bold"),
        )
        style.map("Primary.TButton",
                  background=[("active", "#7F5AEE"), ("disabled", "#232329")])

        style.configure(
            "Secondary.TButton",
            background=_BTN, foreground=_TEXTPRI,
            padding=(10, 7), font=self._sans_sub,
        )
        style.map("Secondary.TButton", background=[("active", _BTNHOV)])

        style.configure(
            "Chip.TButton",
            background=_CARD, foreground=_ACCENT,
            padding=(8, 5), font=self._sans_sub,
        )
        style.map("Chip.TButton", background=[("active", _HOVER)])

        style.configure(
            "TProgressbar",
            troughcolor=_CARD, background=_ACCENT,
            bordercolor=_CARD, lightcolor=_ACCENT, darkcolor=_ACCENT,
        )

    def _build_ui(self) -> None:
        _BG      = THEME["bg"]
        _TITLE   = THEME["title"]
        _PANEL   = THEME["panel"]
        _VPT     = THEME["viewport"]
        _BORDER  = THEME["border"]
        _ACCENT  = THEME["accent"]
        _TEXTPRI = THEME["text"]
        _TEXTSEC = THEME["muted"]

        self.root.configure(bg=_BG)

        titlebar = tk.Frame(self.root, bg=_TITLE, height=42)
        titlebar.pack(fill=tk.X)
        titlebar.pack_propagate(False)

        tb_left = tk.Frame(titlebar, bg=_TITLE)
        tb_left.pack(side=tk.LEFT, padx=16, fill=tk.Y)

        logo_mark = tk.Frame(tb_left, bg=_TITLE)
        logo_mark.pack(side=tk.LEFT, anchor=tk.CENTER)
        tk.Frame(logo_mark, bg="#7c3aed", width=3, height=10).pack(side=tk.LEFT, pady=(5,0))
        tk.Frame(logo_mark, bg=_ACCENT,  width=3, height=16).pack(side=tk.LEFT, padx=2)
        tk.Frame(logo_mark, bg="#A78BFA", width=3, height=12).pack(side=tk.LEFT, pady=(2,0))

        tk.Label(tb_left, text=" ARGUS", bg=_TITLE, fg=_TEXTPRI,
                 font=(*self._sans_btn[:2], "bold")).pack(side=tk.LEFT)
        tk.Label(tb_left, text=" v5.0", bg=_TITLE, fg=_TEXTSEC,
                 font=self._sans_sub).pack(side=tk.LEFT)

        tk.Label(tb_left, text="  /  ", bg=_TITLE, fg=_TEXTSEC,
                 font=self._sans_sub).pack(side=tk.LEFT)
        tk.Label(tb_left, text="Workbench  /  ", bg=_TITLE, fg=_TEXTSEC,
                 font=self._sans_sub).pack(side=tk.LEFT)
        tk.Label(tb_left, textvariable=self.model_name_var, bg=_TITLE, fg="#A1A1AA",
                 font=self._sans_sub).pack(side=tk.LEFT)

        tb_center = tk.Frame(titlebar, bg=_TITLE)
        tb_center.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        tabs_bg = tk.Frame(tb_center, bg=_TITLE)
        tabs_bg.pack(padx=4, pady=4)

        self._nav_workbench_btn = tk.Label(
            tabs_bg, text="WORKBENCH",
            bg="#1E1633", fg=_ACCENT,
            font=(*self._sans_sub[:2], "bold"), padx=14, pady=5, cursor="hand2",
        )
        self._nav_workbench_btn.pack(side=tk.LEFT, padx=3, pady=3)
        self._nav_workbench_btn.bind("<Button-1>", lambda _e: self._show_generator_page())

        self._nav_dashboard_btn = tk.Label(
            tabs_bg, text="DASHBOARD",
            bg=_TITLE, fg=_TEXTSEC,
            font=(*self._sans_sub[:2],), padx=14, pady=5, cursor="hand2",
        )
        self._nav_dashboard_btn.pack(side=tk.LEFT, pady=3)
        self._nav_dashboard_btn.bind("<Button-1>", lambda _e: self._show_dashboard_page())

        self._struct_mode = False
        self._struct_tab_btn = tk.Label(
            tabs_bg, text="STRUCT",
            bg=_TITLE, fg=_TEXTSEC,
            font=(*self._sans_sub[:2],), padx=14, pady=5, cursor="hand2",
        )
        self._struct_tab_btn.pack(side=tk.LEFT, padx=(0,3), pady=3)
        self._struct_tab_btn.bind("<Button-1>", lambda _e: self._toggle_struct_mode())

        tb_right = tk.Frame(titlebar, bg=_TITLE)
        tb_right.pack(side=tk.RIGHT, padx=0, fill=tk.Y)

        _mcp_bg = tk.Frame(tb_right, bg="#191527",
                           highlightthickness=1, highlightbackground="#27203D")
        _mcp_bg.pack(side=tk.LEFT, padx=10, pady=10)
        self._status_dot = tk.Label(_mcp_bg, text="●", bg="#191527", fg=_ACCENT,
                                    font=_sans_font(7))
        self._status_dot.pack(side=tk.LEFT, padx=(6,2), pady=3)
        tk.Label(_mcp_bg, textvariable=self.stage_var, bg="#191527", fg="#8B5CF6",
                 font=(*self._sans_sub[:2], "bold"), padx=(4), pady=3).pack(side=tk.LEFT, padx=(0,6))

        for txt, cmd, hov in (
            ("─", self._minimize_window, "#2A2A30"),
            ("□", self._toggle_maximize, "#2A2A30"),
            ("✕", lambda: self.root.destroy(), "#3F1D1D"),
        ):
            b = tk.Label(tb_right, text=txt, bg=_TITLE, fg="#555",
                         font=_sans_font(10), width=4, pady=0, cursor="hand2")
            b.pack(side=tk.LEFT, fill=tk.Y)
            if cmd:
                b.bind("<Button-1>", lambda _e, c=cmd: c())
            b.bind("<Enter>", lambda _e, w=b, c=hov: w.configure(bg=c, fg="#ccc"))
            b.bind("<Leave>", lambda _e, w=b: w.configure(bg=_TITLE, fg="#555"))

        tk.Frame(self.root, bg=_BORDER, height=1).pack(fill=tk.X)

        body = tk.Frame(self.root, bg=_BG)
        body.pack(fill=tk.BOTH, expand=True)
        self._body_ref = body
        self._left_collapsed = False

        self.left_panel = tk.Frame(body, bg=_PANEL, width=240)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
        self.left_panel.pack_propagate(False)
        tk.Frame(self.left_panel, bg=_BORDER, width=1).pack(side=tk.RIGHT, fill=tk.Y)

        self.workbench_panel = tk.Frame(self.left_panel, bg=_PANEL)
        self._build_left_workbench(self.workbench_panel)

        self.right_panel = tk.Frame(body, bg=_PANEL, width=195)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_panel.pack_propagate(False)
        tk.Frame(self.right_panel, bg=_BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)
        self._build_right_graph(self.right_panel)

        center = tk.Frame(body, bg=_BG)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._center_frame = center

        vp_toolbar = tk.Frame(center, bg="#101013", height=34)
        vp_toolbar.pack(fill=tk.X)
        vp_toolbar.pack_propagate(False)
        tk.Frame(center, bg=_BORDER, height=1).pack(fill=tk.X)

        self._view_btn_wire = tk.Label(
            vp_toolbar, text="MESH",
            bg="#1E1633", fg=_ACCENT,
            font=(*self._sans_sub[:2], "bold"), padx=10, pady=5, cursor="hand2",
        )
        self._view_btn_wire.pack(side=tk.LEFT, padx=(8, 0))
        self._view_btn_wire.bind("<Button-1>", lambda _e: self._set_view_mode("wire"))

        tk.Frame(vp_toolbar, bg=_BORDER, width=1, height=18).pack(side=tk.LEFT, padx=8)

        tk.Label(vp_toolbar, textvariable=self.model_name_var,
                 bg="#101013", fg="#71717A",
                 font=(*self._sans_sub[:2], "bold")).pack(side=tk.LEFT)
        tk.Label(vp_toolbar, textvariable=self.primitive_count_var,
                 bg="#101013", fg=_TEXTSEC,
                 font=self._sans_sub).pack(side=tk.LEFT, padx=(6, 0))

        _view3d = tk.Label(
            vp_toolbar, text="◈ View 3D", bg="#1E1633", fg=_ACCENT,
            font=(*self._sans_sub[:2], "bold"), padx=10, pady=3, cursor="hand2",
        )
        _view3d.pack(side=tk.RIGHT, padx=(0, 10))
        _view3d.bind("<Button-1>", lambda _e: (
            self._open_web_viewer(str(self.model_path)) if self.model_path
            else messagebox.showinfo("3D Viewer", "Generate or load an asset first.")))

        self._rotate_btn = tk.Label(
            vp_toolbar, text="⏸", bg="#101013", fg=_ACCENT,
            font=_sans_font(13), padx=6, cursor="hand2",
        )
        self._rotate_btn.pack(side=tk.RIGHT, padx=(0, 8))
        self._rotate_btn.bind("<Button-1>", lambda _e: self._toggle_rotation())

        for ico in ("⤡", "↺"):
            tk.Label(vp_toolbar, text=ico, bg="#101013", fg=_TEXTSEC,
                     font=_sans_font(12), padx=6, cursor="hand2").pack(side=tk.RIGHT)

        self.dashboard_panel = tk.Frame(body, bg="#0B0B0E")
        self.dashboard_frame = self.dashboard_panel
        self._build_dashboard_page(self.dashboard_panel)

        viewport_frame = tk.Frame(center, bg=_VPT)
        self.viewport_frame = viewport_frame
        viewport_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_canvas = tk.Canvas(
            viewport_frame, highlightthickness=0, bg=_VPT, bd=0,
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind("<Configure>", self._draw_preview_background)
        self.preview_canvas.bind("<ButtonPress-1>", self._start_model_drag)
        self.preview_canvas.bind("<B1-Motion>", self._drag_model)
        self.preview_canvas.bind("<ButtonRelease-1>", self._stop_model_drag)
        self.preview_canvas.bind("<MouseWheel>", self._zoom_model)

        self.preview_hint = tk.Label(
            self.preview_canvas, text="No asset loaded\nGenerate or load a .blend file",
            fg="#3F3F46", bg=_VPT, font=self._sans_sub, justify=tk.CENTER,
        )
        self._preview_hint_id = self.preview_canvas.create_window(
            0, 0, window=self.preview_hint, anchor=tk.CENTER,
        )

        _sb = "#0C0C0F"
        _sf = _ACCENT
        _sd = "#3F3F46"
        self._stats_frame = tk.Frame(self.preview_canvas, bg=_sb,
                                     highlightthickness=1, highlightbackground="#27203D")
        tk.Label(self._stats_frame, text="MESH STATS",
                 bg=_sb, fg=_sf,
                 font=(*self._sans_sub[:2], "bold"), padx=10, pady=6).pack(anchor=tk.W)
        tk.Frame(self._stats_frame, bg="#1E1E23", height=1).pack(fill=tk.X)
        self._stat_verts_lbl  = tk.Label(self._stats_frame, text="Verts    —", bg=_sb, fg=_sd, font=self._sans_sub, padx=10, pady=2, anchor=tk.W)
        self._stat_edges_lbl  = tk.Label(self._stats_frame, text="Edges    —", bg=_sb, fg=_sd, font=self._sans_sub, padx=10, pady=2, anchor=tk.W)
        self._stat_faces_lbl  = tk.Label(self._stats_frame, text="Faces    —", bg=_sb, fg=_sd, font=self._sans_sub, padx=10, pady=2, anchor=tk.W)
        self._stat_meshes_lbl = tk.Label(self._stats_frame, text="Objects  —", bg=_sb, fg=_sd, font=self._sans_sub, padx=10, pady=2, anchor=tk.W)
        for lbl in (self._stat_verts_lbl, self._stat_edges_lbl,
                    self._stat_faces_lbl, self._stat_meshes_lbl):
            lbl.pack(fill=tk.X)
        tk.Frame(self._stats_frame, bg=_sb, height=4).pack()
        self._stats_window_id = self.preview_canvas.create_window(
            0, 12, window=self._stats_frame, anchor=tk.NE, tags="stats_overlay",
        )
        self.preview_canvas.bind("<Configure>", self._on_canvas_resize, add="+")

        self.preview_canvas.bind("<ButtonPress-3>",   self._start_pan)
        self.preview_canvas.bind("<B3-Motion>",       self._pan_model)
        self.preview_canvas.bind("<ButtonRelease-3>", self._stop_pan)
        self.preview_canvas.bind("<Double-Button-1>", self._reset_view_pan)

        self._build_run_timeline(center)

    def _toggle_struct_mode(self) -> None:
        """Toggle STRUCT mode — when ON, structure memory is auto-saved after generation."""
        self._struct_mode = not self._struct_mode
        _A  = "#8B5CF6"
        _TS = "#71717A"
        if self._struct_mode:
            self._struct_tab_btn.configure(
                bg="#1E1633", fg=_A,
                font=(*self._sans_sub[:2], "bold"),
            )
            try:
                self._log_stage_lbl.configure(text="STRUCT")
                self._log_msg_lbl.configure(
                    text="Structure memory ON — next generation will auto-save blueprint"
                )
            except Exception:
                pass
        else:
            self._struct_tab_btn.configure(
                bg=THEME["title"], fg=_TS,
                font=(*self._sans_sub[:2],),
            )
            try:
                self._log_stage_lbl.configure(text="READY")
                self._log_msg_lbl.configure(text="Structure memory OFF")
            except Exception:
                pass

    def _export_on_demand(self, fmt: str) -> None:
        """Export GLB or FBX from the current .blend on button click."""
        if self.model_path is None:
            messagebox.showwarning("No asset", "Load an asset first.")
            return

        blend_path = self.model_path.parent / (self.model_path.stem + ".blend")
        if not blend_path.exists():
            messagebox.showerror("No .blend", f".blend not found:\n{blend_path}")
            return

        out_path = self.model_path.parent / f"{self.model_path.stem}.{fmt}"

        btn = self._export_glb_btn if fmt == "glb" else self._export_fbx_btn
        btn.configure(text=f"  ⏳  Exporting {fmt.upper()}...")
        self._export_status_lbl.configure(text="", fg="#34D399")

        def _worker():
            from core.blender import export_format_from_blend
            ok = export_format_from_blend(blend_path, fmt, out_path)
            label = f"  📦  Export GLB" if fmt == "glb" else f"  🗂️  Export FBX"
            if ok:
                self.root.after(0, lambda: btn.configure(text=label))
                self.root.after(0, lambda: self._export_status_lbl.configure(
                    text=f"✓ {fmt.upper()} saved  ({out_path.stat().st_size // 1024} KB)",
                    fg="#34D399",
                ))
                self.root.after(0, lambda: self._append_operator_log(
                    "EXPORT", f"{fmt.upper()} → {out_path.name}",
                    event_id=f"export:{out_path}",
                ))
            else:
                self.root.after(0, lambda: btn.configure(text=label))
                self.root.after(0, lambda: self._export_status_lbl.configure(
                    text=f"✗ Export failed", fg="#EF4444",
                ))

        threading.Thread(target=_worker, daemon=True).start()

    def _update_mesh_stats_overlay(self, n_verts: int, n_faces: int, n_objects: int) -> None:
        """Update the mesh stats overlay with current model metrics."""
        n_edges = int(n_faces * 1.5)
        _sf = "#8B5CF6"
        self._stat_verts_lbl.configure( text=f"Verts    {n_verts:>7,}", fg=_sf)
        self._stat_edges_lbl.configure( text=f"Edges    {n_edges:>7,}", fg=_sf)
        self._stat_faces_lbl.configure( text=f"Faces    {n_faces:>7,}", fg=_sf)
        self._stat_meshes_lbl.configure(text=f"Objects  {n_objects:>7,}", fg=_sf)

    def _on_canvas_resize(self, event) -> None:
        """Keep the stats overlay anchored to the top-right corner."""
        try:
            w = self.preview_canvas.winfo_width()
            self.preview_canvas.coords(self._stats_window_id, w - 12, 12)
        except Exception:
            pass

    def _toggle_left_panel(self) -> None:
        if getattr(self, "_panel_animating", False):
            return
        self._left_collapsed = not self._left_collapsed
        if self._left_collapsed:
            self._animate_left_panel(collapse=True)
        else:
            if self.viewport_frame:
                self.viewport_frame.pack_forget()
            self.left_panel.configure(width=0)
            self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
            if self.workbench_panel and not self.workbench_panel.winfo_ismapped():
                self.workbench_panel.pack(fill=tk.BOTH, expand=True)
            if self.viewport_frame:
                self.viewport_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self._animate_left_panel(collapse=False)

    def _animate_left_panel(self, collapse: bool) -> None:
        _FULL   = 240
        _STEPS  = 16
        _MS     = 11
        start   = _FULL if collapse else 0
        end     = 0    if collapse else _FULL

        self._panel_animating = True

        def _step(i: int) -> None:
            if i > _STEPS:
                self.left_panel.configure(width=end)
                self._panel_animating = False
                if collapse:
                    self.left_panel.pack_forget()
                return
            t = i / _STEPS
            t = t * t * (3 - 2 * t)
            w = int(start + (end - start) * t)
            self.left_panel.configure(width=max(0, w))
            self.root.after(_MS, lambda: _step(i + 1))

        _step(0)

    def _build_left_workbench(self, parent: tk.Frame) -> None:
        _P  = "#0E0E11"
        _IN = "#101013"
        _B  = "#1F1F24"
        _A  = "#8B5CF6"
        _TP = "#FAFAFA"
        _TS = "#71717A"

        def _sec_label(f, txt):
            row = tk.Frame(f, bg=_P)
            row.pack(fill=tk.X, pady=(12, 5))
            tk.Label(row, text=txt, bg=_P, fg=_TS,
                     font=(*self._sans_sub[:2], "bold"),
                     padx=0).pack(side=tk.LEFT)
            tk.Frame(row, bg=_B, height=1).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6,0))

        scroll_outer = tk.Frame(parent, bg=_P)
        scroll_outer.pack(fill=tk.BOTH, expand=True)

        inner = tk.Frame(scroll_outer, bg=_P)
        inner.pack(fill=tk.BOTH, expand=True, padx=14)

        _sec_label(inner, "PROMPT")
        self.prompt_text = tk.Text(
            inner, height=5, wrap=tk.WORD,
            bg=_IN, fg=_TP, insertbackground=_TP,
            relief=tk.FLAT, font=self._sans_sub,
            padx=12, pady=10,
            highlightthickness=1, highlightbackground=_B,
            highlightcolor=_A,
        )
        self.prompt_text.pack(fill=tk.X)

        self._prompt_placeholder = "Describe the 3D asset you want to create..."
        self._prompt_is_placeholder = False

        def _ph_clear(_e=None):
            if self._prompt_is_placeholder:
                self._prompt_is_placeholder = False
                self.prompt_text.delete("1.0", tk.END)
                self.prompt_text.configure(fg=_TP)

        def _ph_restore(_e=None):
            if not self.prompt_text.get("1.0", "end-1c").strip():
                self._prompt_is_placeholder = True
                self.prompt_text.configure(fg="#52525B")
                self.prompt_text.delete("1.0", tk.END)
                self.prompt_text.insert("1.0", self._prompt_placeholder)

        self._prompt_ph_clear = _ph_clear
        self.prompt_text.bind("<FocusIn>", _ph_clear)
        self.prompt_text.bind("<FocusOut>", _ph_restore)
        _ph_restore()

        _sec_label(inner, "SETTINGS")

        def _toggle_group(parent_f, label, opts, var, on_change=None):
            tk.Label(parent_f, text=label, bg=_P, fg=_TS,
                     font=self._sans_sub).pack(anchor=tk.W, pady=(0,3))
            row = tk.Frame(parent_f, bg=_IN, highlightthickness=1,
                           highlightbackground=_B)
            row.pack(fill=tk.X, pady=(0,6))
            btns = {}
            def _select(o):
                var.set(o)
                for k, b in btns.items():
                    if k == o:
                        b.configure(bg="#1E1633", fg=_A)
                    else:
                        b.configure(bg=_IN, fg=_TS)
                if on_change:
                    on_change()
            for o in opts:
                b = tk.Label(row, text=o, bg=_IN if var.get() != o else "#1E1633",
                             fg=_A if var.get() == o else _TS,
                             font=(*self._sans_sub[:2], "bold"),
                             padx=8, pady=4, cursor="hand2")
                b.pack(side=tk.LEFT, fill=tk.X, expand=True)
                b.bind("<Button-1>", lambda _e, opt=o: _select(opt))
                btns[o] = b
            return btns

        self._poly_var = tk.StringVar(value="Medium")
        _toggle_group(inner, "Poly Budget", ["Low", "Medium", "High"], self._poly_var)

        self._mode_var = tk.StringVar(value="Live MCP" if self.mcp_var.get() else "Headless")
        def _on_mode_change():
            self.mcp_var.set(self._mode_var.get() == "Live MCP")
            self._update_mode_label()
        _toggle_group(inner, "Mode", ["Headless", "Live MCP"],
                      self._mode_var, _on_mode_change)

        cp_row = tk.Frame(inner, bg=_P)
        cp_row.pack(fill=tk.X, pady=(0, 6))
        tk.Label(cp_row, text="Visual Reference", bg=_P, fg=_TS,
                 font=self._sans_sub).pack(side=tk.LEFT)
        self._concept_toggle_lbl = tk.Label(
            cp_row, text="ON", bg="#153226", fg="#34D399",
            font=(*self._sans_sub[:2], "bold"), padx=8, pady=2, cursor="hand2",
        )
        self._concept_toggle_lbl.pack(side=tk.RIGHT)
        self._concept_toggle_lbl.bind("<Button-1>", lambda _e: self._toggle_concept_pipeline())

        tk.Frame(inner, bg=_B, height=1).pack(fill=tk.X, pady=(6, 10))

        self._btn_photos: dict[str, object] = {}
        self.run_button = tk.Label(
            inner, text="⚡  GENERATE ASSET",
            bg=_A, fg="#FFFFFF", compound="center", borderwidth=0,
            font=(*self._sans_btn[:2], "bold"), pady=10, cursor="hand2",
        )
        self.run_button.pack(fill=tk.X)
        self.run_button.bind("<Button-1>", lambda _e: self.on_run_clicked())
        self.run_button.bind(
            "<Enter>",
            lambda _e: None if self.is_running else self._set_run_visual("hover"),
        )
        self.run_button.bind(
            "<Leave>",
            lambda _e: None if self.is_running else self._set_run_visual("normal"),
        )

        self._run_btn_w = 0

        def _on_run_btn_resize(e):
            if e.width > 40 and e.width != self._run_btn_w:
                self._run_btn_w = e.width
                self._round_rect_photo(e.width, 44, 12, "#8B5CF6", "#6366F1", key="run_normal")
                self._round_rect_photo(e.width, 44, 12, "#A78BFA", "#818CF8", key="run_hover")
                self._round_rect_photo(e.width, 44, 12, "#2E2447", None, key="run_busy")
                self._set_run_visual("busy" if self.is_running else "normal")

        self.run_button.bind("<Configure>", _on_run_btn_resize)

        self._run_btn_ttk = ttk.Button(inner, command=self.on_run_clicked,
                                        style="Primary.TButton")

        # Packed only while a run is active — an idle indeterminate bar
        # parks its block at the left edge and reads as a stray sliver.
        self.progress = ttk.Progressbar(inner, mode="indeterminate")

        acts = tk.Frame(inner, bg=_P)
        acts.pack(fill=tk.X, pady=(8, 0))
        self._progress_anchor = acts
        for txt, cmd in (
            ("Batch", None),
            ("Output", self._open_output_folder),
            ("Log", self._open_cmd_log_window),
        ):
            b = tk.Label(acts, text=txt, bg=_IN, fg=_TS,
                         font=self._sans_sub, padx=8, pady=4,
                         highlightthickness=1, highlightbackground=_B,
                         cursor="hand2")
            b.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,3))
            if cmd:
                b.bind("<Button-1>", lambda _e, c=cmd: c())
            b.bind("<Enter>", lambda _e, w=b: w.configure(fg=_A))
            b.bind("<Leave>", lambda _e, w=b: w.configure(fg=_TS))

        _sec_label(inner, "RECENT")
        self._recent_list_frame = tk.Frame(inner, bg=_P)
        self._recent_list_frame.pack(fill=tk.X)

        for i, (name, t) in enumerate([
            ("No recent assets", ""),
        ]):
            row = tk.Frame(self._recent_list_frame, bg=_P)
            row.pack(fill=tk.X, pady=1)
            tk.Frame(row, bg=_B if i > 0 else _P,
                     width=7, height=7).pack(side=tk.LEFT, padx=(0,6), pady=4)
            tk.Label(row, text=name, bg=_P, fg=_TS,
                     font=self._sans_sub).pack(side=tk.LEFT)

        _hidden = tk.Frame(parent, bg=_P)
        self.output_text = tk.Text(_hidden, wrap=tk.WORD, bg=_IN,
                                   fg="#A1A1AA", relief=tk.FLAT,
                                   font=self._mono, padx=8, pady=8)
        self.output_text.configure(state=tk.DISABLED)

        self._badge_label = tk.Label(parent, textvariable=self.status_badge_var,
                                     fg=_TP, bg="#0E0E11",
                                     font=(*self._sans_sub[:2], "bold"),
                                     padx=12, pady=0)
        self._badge_label.pack(side=tk.BOTTOM, anchor=tk.W, padx=14, pady=(0,6))

        parent.pack(fill=tk.BOTH, expand=True)

    def _build_run_timeline(self, parent: tk.Frame) -> None:
        _T = THEME
        self._timeline_container = tk.Frame(
            parent,
            bg="#0C0C0F",
            highlightthickness=1,
            highlightbackground=_T["border"],
        )
        self._timeline_container.pack(fill=tk.X, side=tk.BOTTOM)

        header = tk.Frame(self._timeline_container, bg="#0C0C0F", height=36)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        header.bind("<Button-1>", lambda _e: self._toggle_timeline())

        self._log_stage_lbl = tk.Label(
            header,
            text="READY",
            bg="#191527",
            fg=_T["accent"],
            font=(*self._sans_sub[:2], "bold"),
            padx=10,
            pady=6,
            cursor="hand2",
        )
        self._log_stage_lbl.pack(side=tk.LEFT, padx=(8, 0), pady=5)
        self._log_stage_lbl.bind("<Button-1>", lambda _e: self._toggle_timeline())

        self._log_msg_lbl = tk.Label(
            header,
            text="Generate an asset to begin",
            bg="#0C0C0F",
            fg=_T["muted"],
            font=(*self._mono[:2],),
            padx=9,
            cursor="hand2",
        )
        self._log_msg_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._log_msg_lbl.bind("<Button-1>", lambda _e: self._toggle_timeline())

        self._log_ok_lbl = tk.Label(
            header,
            text="",
            bg="#0C0C0F",
            fg=_T["accent"],
            font=(*self._sans_sub[:2], "bold"),
            padx=10,
        )
        self._log_ok_lbl.pack(side=tk.RIGHT)

        self._score_badge_lbl = tk.Label(
            header,
            text="",
            bg="#0C0C0F",
            fg=_T["muted"],
            font=(*self._sans_sub[:2], "bold"),
            padx=10,
        )
        self._score_badge_lbl.pack(side=tk.RIGHT)

        self._elapsed_lbl = tk.Label(
            header,
            text="",
            bg="#0C0C0F",
            fg=_T["muted"],
            font=self._mono,
            padx=8,
        )
        self._elapsed_lbl.pack(side=tk.RIGHT)

        self._timeline_toggle_lbl = tk.Label(
            header,
            text="▾",
            bg="#0C0C0F",
            fg=_T["muted_alt"],
            font=(*self._sans_sub[:2], "bold"),
            padx=10,
            cursor="hand2",
        )
        self._timeline_toggle_lbl.pack(side=tk.RIGHT)
        self._timeline_toggle_lbl.bind("<Button-1>", lambda _e: self._toggle_timeline())

        self._timeline_body = tk.Frame(self._timeline_container, bg="#0A0A0C")
        self._timeline_body.pack(fill=tk.X)
        tk.Frame(self._timeline_body, bg=_T["border"], height=1).pack(fill=tk.X)
        self._add_timeline_event("SYSTEM", "ARGUS operator console ready", kind="idle")

    def _toggle_timeline(self) -> None:
        if self._timeline_body is None:
            return
        self._timeline_expanded = not self._timeline_expanded
        if self._timeline_expanded:
            self._timeline_body.pack(fill=tk.X)
            if self._timeline_toggle_lbl is not None:
                self._timeline_toggle_lbl.configure(text="▾")
        else:
            self._timeline_body.pack_forget()
            if self._timeline_toggle_lbl is not None:
                self._timeline_toggle_lbl.configure(text="▸")

    def _add_timeline_event(self, label: str, message: str, kind: str = "idle") -> None:
        if self._timeline_body is None:
            return

        colors = {
            "idle": THEME["muted_alt"],
            "run": THEME["warning"],
            "ok": THEME["success_soft"],
            "fail": THEME["danger"],
            "error": THEME["danger"],
            "accent": THEME["accent"],
        }
        elapsed = "00:00"
        if self._run_started_at is not None:
            delta = datetime.now() - self._run_started_at
            elapsed = f"{int(delta.total_seconds() // 60):02d}:{int(delta.total_seconds() % 60):02d}"

        row = tk.Frame(self._timeline_body, bg="#0A0A0C")
        row.pack(fill=tk.X, padx=8, pady=(5, 0))
        tk.Label(
            row,
            text=elapsed,
            bg="#0A0A0C",
            fg=THEME["dim"],
            font=self._mono,
            width=6,
            anchor=tk.W,
        ).pack(side=tk.LEFT)
        tk.Label(
            row,
            text=label[:12],
            bg="#141119" if kind in {"run", "accent"} else "#0F0F12",
            fg=colors.get(kind, THEME["muted_alt"]),
            font=(*self._sans_sub[:2], "bold"),
            padx=8,
            pady=3,
            width=12,
            anchor=tk.W,
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(
            row,
            text=message[:150],
            bg="#0A0A0C",
            fg=THEME["text"] if kind in {"ok", "run", "accent"} else "#9B9BA3",
            font=self._sans_sub,
            anchor=tk.W,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._timeline_rows.append(row)
        while len(self._timeline_rows) > 10:
            old_row = self._timeline_rows.pop(0)
            old_row.destroy()

    def _clear_timeline(self) -> None:
        for row in self._timeline_rows:
            row.destroy()
        self._timeline_rows.clear()


    def _minimize_window(self) -> None:
        try:
            self.root.iconify()
        except tk.TclError:
            pass

    def _toggle_maximize(self) -> None:
        try:
            self.root.state("normal" if self.root.state() == "zoomed" else "zoomed")
        except tk.TclError:
            pass


    @staticmethod
    def _extract_score(line: str) -> int | None:
        m = re.search(r"(-?\d+)\s*/\s*10", line)
        return int(m.group(1)) if m else None

    @staticmethod
    def _score_kind(score: int) -> str:
        if score >= 7:
            return "ok"
        if score >= 4:
            return "run"
        return "fail"

    def _set_score_badge(self, score: int) -> None:
        colors = {"ok": THEME["success_soft"], "run": THEME["warning"],
                  "fail": THEME["danger"]}
        if hasattr(self, "_score_badge_lbl"):
            self._score_badge_lbl.configure(
                text=f"SCORE {score}/10",
                fg=colors.get(self._score_kind(score), THEME["muted"]),
            )

    @staticmethod
    def _severity_color(severity: str) -> str:
        return {
            "clean": "#34D399",
            "medium": THEME["warning"],
            "high": "#f59e0b",
            "critical": THEME["danger"],
        }.get(severity.strip().lower(), THEME["muted_alt"])

    def _add_score_history(self, score: int) -> None:
        frame = getattr(self, "_score_history_frame", None)
        if frame is None:
            return
        placeholder = getattr(self, "_score_history_placeholder", None)
        if placeholder is not None:
            placeholder.destroy()
            self._score_history_placeholder = None
        self._score_history_count = getattr(self, "_score_history_count", 0) + 1
        colors = {"ok": THEME["success_soft"], "run": THEME["warning"],
                  "fail": THEME["danger"]}
        chip = tk.Label(
            frame, text=f"{self._score_history_count}\n{score}",
            bg="#1A1A1F", fg=colors.get(self._score_kind(score), THEME["muted_alt"]),
            font=(*self._sans_sub[:2], "bold"), padx=6, pady=2, justify=tk.CENTER,
        )
        chip.pack(side=tk.LEFT, padx=(0, 4))
        children = frame.winfo_children()
        if len(children) > 10:
            children[0].destroy()

    def _tick_run_elapsed(self) -> None:
        if not getattr(self, "is_running", False) or self._run_started_at is None:
            return
        delta = (datetime.now() - self._run_started_at).total_seconds()
        if hasattr(self, "_elapsed_lbl"):
            self._elapsed_lbl.configure(
                text=f"{int(delta // 60):02d}:{int(delta % 60):02d}")
        self.root.after(1000, self._tick_run_elapsed)

    def _build_right_graph(self, parent: tk.Frame) -> None:
        _P  = "#0E0E11"
        _IN = "#101013"
        _B  = "#1F1F24"
        _A  = "#8B5CF6"
        _TS = "#3F3F46"
        _TP = "#A1A1AA"

        content = tk.Frame(parent, bg=_P)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        _hidden = tk.Frame(content, bg=_P)
        self._inspector_panel  = _hidden
        self._rtab_inspector   = tk.Label(_hidden)
        self._rtab_lights      = tk.Label(_hidden)
        self._lights_panel     = tk.Frame(_hidden, bg=_P)
        self._graph_search_var = tk.StringVar()
        self._graph_mode       = tk.StringVar(value="tree")
        self._graph_tree_tab   = tk.Label(_hidden)
        self._graph_flat_tab   = tk.Label(_hidden)
        _list_canvas           = tk.Canvas(_hidden, bg=_P)
        self.graph_list_frame  = tk.Frame(_list_canvas, bg=_P)

        _scroll_stub = ttk.Scrollbar(_hidden, orient=tk.VERTICAL)
        self._graph_search_var.trace_add("write", lambda *_: self._filter_graph())

        canvas = tk.Canvas(content, bg=_P, highlightthickness=0)
        scroll = ttk.Scrollbar(content, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg=_P)
        win_id = canvas.create_window((0, 0), window=inner, anchor=tk.NW)
        inner.bind("<Configure>", lambda _e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win_id, width=e.width))

        def _sec(f, title):
            tk.Frame(f, bg=_B, height=1).pack(fill=tk.X, pady=(10, 6))
            tk.Label(f, text=title, bg=_P, fg=_TS,
                     font=(*self._sans_sub[:2], "bold"),
                     padx=13, pady=0).pack(anchor=tk.W, pady=(0, 5))

        def _prop_row(f, key, var_or_text, color=_TP):
            row = tk.Frame(f, bg=_P)
            row.pack(fill=tk.X, padx=13, pady=1)
            tk.Label(row, text=key, bg=_P, fg=_TS,
                     font=self._sans_sub).pack(side=tk.LEFT)
            if isinstance(var_or_text, tk.StringVar):
                lbl = tk.Label(row, textvariable=var_or_text, bg=_P,
                               fg=color, font=(*self._sans_sub[:2], "bold"))
            else:
                lbl = tk.Label(row, text=var_or_text, bg=_P,
                               fg=color, font=(*self._sans_sub[:2], "bold"))
            lbl.pack(side=tk.RIGHT)
            return lbl

        _sec(inner, "ASSET PROPERTIES")
        self._rp_category_lbl  = _prop_row(inner, "Category",  "—")
        self._rp_material_lbl  = _prop_row(inner, "Material",  "—")
        self._rp_scale_lbl     = _prop_row(inner, "Scale",     "—")
        self._rp_budget_lbl    = _prop_row(inner, "Budget",    "—")
        self._rp_score_lbl     = _prop_row(inner, "Score",     "—", "#A78BFA")
        self._rp_blueprint_lbl = _prop_row(inner, "Blueprint", "—", "#34D399")

        _sec(inner, "VALIDATION")
        self._rp_topo_lbl     = _prop_row(inner, "Status",     "—", "#34D399")
        self._rp_ngons_lbl    = _prop_row(inner, "N-gons",     "—", "#34D399")
        self._rp_edges_lbl    = _prop_row(inner, "Open Edges", "—", "#34D399")
        self._rp_ground_lbl   = _prop_row(inner, "Grounded",   "—", "#34D399")
        self._rp_repairs_lbl  = _prop_row(inner, "Repairs",    "—", "#34D399")

        _sec(inner, "SCORE HISTORY")
        self._score_history_frame = tk.Frame(inner, bg=_P)
        self._score_history_frame.pack(fill=tk.X, padx=13, pady=(0, 6))
        self._score_history_placeholder = tk.Label(
            self._score_history_frame, text="—", bg=_P, fg=_TS, font=self._sans_sub)
        self._score_history_placeholder.pack(side=tk.LEFT)
        self._score_history_count = 0

        _sec(inner, "EXPORTS")
        exp_frame = tk.Frame(inner, bg=_P)
        exp_frame.pack(fill=tk.X, padx=13, pady=(0, 4))

        _eb  = "#101013"
        _ebh = "#1A1426"

        self._export_glb_btn = tk.Label(
            exp_frame, text="Export GLB",
            bg=_eb, fg=_TS,
            font=(*self._sans_sub[:2], "bold"),
            padx=8, pady=7, cursor="hand2", anchor=tk.W,
            highlightthickness=1, highlightbackground=_B,
        )
        self._export_glb_btn.pack(fill=tk.X, pady=(0, 3))
        self._export_glb_btn.bind("<Button-1>", lambda _e: self._export_on_demand("glb"))
        self._export_glb_btn.bind("<Enter>",    lambda _e: self._export_glb_btn.configure(bg=_ebh, fg=_A))
        self._export_glb_btn.bind("<Leave>",    lambda _e: self._export_glb_btn.configure(bg=_eb,  fg=_TS))

        self._export_fbx_btn = tk.Label(
            exp_frame, text="Export FBX",
            bg=_eb, fg=_TS,
            font=(*self._sans_sub[:2], "bold"),
            padx=8, pady=7, cursor="hand2", anchor=tk.W,
            highlightthickness=1, highlightbackground=_B,
        )
        self._export_fbx_btn.pack(fill=tk.X, pady=(0, 3))
        self._export_fbx_btn.bind("<Button-1>", lambda _e: self._export_on_demand("fbx"))
        self._export_fbx_btn.bind("<Enter>",    lambda _e: self._export_fbx_btn.configure(bg=_ebh, fg=_A))
        self._export_fbx_btn.bind("<Leave>",    lambda _e: self._export_fbx_btn.configure(bg=_eb,  fg=_TS))

        self._export_status_lbl = tk.Label(exp_frame, text="", bg=_P,
                                           fg="#34D399", font=self._sans_sub)
        self._export_status_lbl.pack(anchor=tk.W, pady=(2, 0))

        _sec(inner, "PIPELINE")
        _STAGES = [
            "Planning",
            "Asset Naming",
            "Script Generation",
            "Critic Review",
            "Blender Execution",
            "MCP Validation",
            "Export",
            "Structure Memory",
        ]
        self._pipeline_stage_labels: dict = {}
        stages_frame = tk.Frame(inner, bg=_P)
        stages_frame.pack(fill=tk.X, padx=13, pady=(0, 14))
        for s in _STAGES:
            row = tk.Frame(stages_frame, bg=_P)
            row.pack(fill=tk.X, pady=2)
            dot = tk.Label(row, text="●", bg=_P, fg=_TS, font=_sans_font(7))
            dot.pack(side=tk.LEFT, padx=(0, 6))
            lbl = tk.Label(row, text=s, bg=_P, fg=_TS, font=self._sans_sub)
            lbl.pack(side=tk.LEFT)
            self._pipeline_stage_labels[s] = (dot, lbl)

        tk.Frame(inner, bg=_B, height=1).pack(fill=tk.X, pady=(6, 0))

    def _switch_right_tab(self, tab: str) -> None:
        _ACCENT  = "#8B5CF6"
        _TEXTSEC = "#9B9BA3"
        _ACT_BG  = "#161320"
        _IDL_BG  = "#0A0A0C"
        if tab == "inspector":
            self._rtab_inspector.configure(bg=_ACT_BG, fg=_ACCENT,
                                            font=(*self._sans_sub[:2], "bold"))
            self._rtab_lights.configure(   bg=_IDL_BG, fg=_TEXTSEC,
                                            font=self._sans_sub)
            self._lights_panel.pack_forget()
            self._inspector_panel.pack(fill=tk.BOTH, expand=True)
        else:
            self._rtab_lights.configure(   bg=_ACT_BG, fg=_ACCENT,
                                            font=(*self._sans_sub[:2], "bold"))
            self._rtab_inspector.configure(bg=_IDL_BG, fg=_TEXTSEC,
                                            font=self._sans_sub)
            self._inspector_panel.pack_forget()
            self._lights_panel.pack(fill=tk.BOTH, expand=True)

    def _build_lights_panel(self, parent: tk.Frame) -> None:
        _PANEL   = "#0D0D10"
        _CARD    = "#141417"
        _BORDER  = "#232329"
        _ACCENT  = "#8B5CF6"
        _TEXTPRI = "#E4E4E7"
        _TEXTSEC = "#9B9BA3"

        tk.Label(parent, text="LIGHTING",
                 bg=_PANEL, fg=_TEXTSEC,
                 font=(*self._sans_btn[:2], "bold"),
                 padx=14, pady=(12)).pack(anchor=tk.W)
        tk.Frame(parent, bg=_BORDER, height=1).pack(fill=tk.X, padx=14)

        def _slider_section(label: str, default_pct: int,
                            on_change, description: str) -> None:
            section = tk.Frame(parent, bg=_PANEL, padx=14, pady=12)
            section.pack(fill=tk.X)

            hrow = tk.Frame(section, bg=_PANEL)
            hrow.pack(fill=tk.X)
            tk.Label(hrow, text=label, bg=_PANEL, fg=_TEXTPRI,
                     font=(*self._sans_sub[:2], "bold")).pack(side=tk.LEFT)
            val_lbl = tk.Label(hrow, text=f"{default_pct}%", bg=_PANEL, fg=_ACCENT,
                               font=(*self._sans_sub[:2], "bold"))
            val_lbl.pack(side=tk.RIGHT)

            tk.Label(section, text=description, bg=_PANEL, fg=_TEXTSEC,
                     font=self._sans_sub, wraplength=210, justify=tk.LEFT,
                     pady=3).pack(anchor=tk.W)

            var = tk.IntVar(value=default_pct)

            def _cb(v, _var=var, _lbl=val_lbl, _fn=on_change):
                pct = int(float(v))
                _lbl.configure(text=f"{pct}%")
                _fn(pct / 100.0)

            scale = tk.Scale(
                section, variable=var, orient=tk.HORIZONTAL,
                from_=0, to=100, showvalue=False, resolution=1,
                bg=_CARD, fg=_ACCENT,
                activebackground=_ACCENT,
                troughcolor="#161619",
                highlightthickness=0, bd=0,
                sliderlength=16, sliderrelief=tk.FLAT,
                command=_cb,
            )
            scale.pack(fill=tk.X, pady=(4, 0))
            tk.Frame(parent, bg=_BORDER, height=1).pack(fill=tk.X, padx=14)

        _slider_section(
            "Ambient Light",
            int(self._ambient_strength * 100),
            self._set_ambient,
            "Fills shadow areas. Low = dramatic shadows,\nhigh = flat/evenly lit.",
        )
        _slider_section(
            "Background",
            int(self._bg_brightness * 100),
            self._set_bg_brightness,
            "Overall scene background brightness.\nDoes not affect the model's colour.",
        )

    def _set_ambient(self, value: float) -> None:
        self._ambient_strength = max(0.0, min(1.0, value))
        if self.model_vertices is not None:
            self._render_model()

    def _set_bg_brightness(self, value: float) -> None:
        self._bg_brightness = max(0.0, min(1.0, value))
        if self.model_vertices is not None:
            self._render_model()

    def _toggle_concept_pipeline(self) -> None:
        new_val = not self.concept_pipeline_var.get()
        self.concept_pipeline_var.set(new_val)
        if self._concept_toggle_lbl:
            if new_val:
                self._concept_toggle_lbl.configure(text="ON", bg="#153226", fg="#34D399")
            else:
                self._concept_toggle_lbl.configure(text="OFF", bg="#1F1A19", fg="#8E8E96")

    def _set_graph_mode(self, mode: str) -> None:
        self._graph_mode.set(mode)
        if mode == "tree":
            self._graph_tree_tab.configure(bg="#161320", fg="#8B5CF6",
                                            font=(*self._sans_sub[:2], "bold"))
            self._graph_flat_tab.configure(bg="#141417", fg="#9B9BA3",
                                            font=self._sans_sub)
        else:
            self._graph_flat_tab.configure(bg="#161320", fg="#8B5CF6",
                                            font=(*self._sans_sub[:2], "bold"))
            self._graph_tree_tab.configure(bg="#141417", fg="#9B9BA3",
                                            font=self._sans_sub)
        self._populate_component_graph(self.component_parts)

    def _filter_graph(self) -> None:
        self._populate_component_graph(self.component_parts)

    def _populate_component_graph(self, parts: list[str]) -> None:
        self.component_parts = parts
        if self.graph_list_frame is None:
            return
        for w in self.graph_list_frame.winfo_children():
            w.destroy()
        if not parts:
            tk.Label(self.graph_list_frame,
                     text="Select a part from the graph\nlist above to customise or\nrename elements in real-time.",
                     bg="#0D0D10", fg="#52525B",
                     font=self._sans_sub, justify=tk.CENTER).pack(pady=40)
            return

        query = self._graph_search_var.get().lower() if hasattr(self, "_graph_search_var") else ""
        filtered = [p for p in parts if query in p.lower()] if query else parts

        if not filtered:
            tk.Label(self.graph_list_frame, text=f'No parts match "{query}"',
                     bg="#0D0D10", fg="#52525B", font=self._sans_sub).pack(pady=20)
            return

        mode = getattr(self, "_graph_mode", tk.StringVar(value="tree")).get()

        if mode == "tree":
            groups: dict[str, list[str]] = {}
            for part in filtered:
                first_word = part.split()[0].rstrip("_-") if part.split() else "Other"
                groups.setdefault(first_word, []).append(part)

            sorted_groups = sorted(groups.items(), key=lambda kv: -len(kv[1]))

            for group_name, group_parts in sorted_groups:
                ghdr = tk.Frame(self.graph_list_frame, bg="#0D0D10", pady=3)
                ghdr.pack(fill=tk.X, pady=(4, 0))
                tk.Label(
                    ghdr, text=f"▼  {group_name}",
                    bg="#0D0D10", fg="#8B5CF6",
                    font=(*self._sans_sub[:2], "bold"), padx=4,
                ).pack(side=tk.LEFT)
                tk.Label(
                    ghdr, text=f"({len(group_parts)} parts)",
                    bg="#0D0D10", fg="#45454D",
                    font=self._sans_sub,
                ).pack(side=tk.LEFT, padx=(4, 0))

                for idx, part in enumerate(group_parts):
                    is_last = idx == len(group_parts) - 1
                    row = tk.Frame(self.graph_list_frame, bg="#111114", pady=4)
                    row.pack(fill=tk.X, padx=(8, 0), pady=(0, 1))
                    connector = "└─" if is_last else "├─"
                    tk.Label(row, text=f"  {connector} ", bg="#111114", fg="#52525B",
                             font=self._mono).pack(side=tk.LEFT)
                    tk.Label(row, text=part, bg="#111114", fg="#D4D4D8",
                             font=self._sans_sub, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
                    ptype = "cylinder" if any(k in part.lower() for k in ("hub", "pipe", "cyl", "rod", "shaft")) \
                        else "wedge" if any(k in part.lower() for k in ("wedge", "ramp", "slope")) \
                        else "box"
                    tk.Label(row, text=ptype, bg="#111114", fg="#34343B",
                             font=self._sans_sub, padx=6).pack(side=tk.RIGHT)
        else:
            tk.Label(
                self.graph_list_frame,
                text=f"All Parts  ({len(filtered)})",
                bg="#0D0D10", fg="#8B5CF6",
                font=(*self._sans_sub[:2], "bold"),
            ).pack(anchor=tk.W, pady=(4, 6))

            for part in filtered:
                row = tk.Frame(self.graph_list_frame, bg="#141417", pady=5)
                row.pack(fill=tk.X, pady=(0, 2))
                tk.Label(row, text="  — ", bg="#141417", fg="#52525B",
                         font=self._sans_sub).pack(side=tk.LEFT)
                tk.Label(row, text=part, bg="#141417", fg="#D4D4D8",
                         font=self._sans_sub, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
                tk.Label(row, text="mesh", bg="#141417", fg="#45454D",
                         font=self._sans_sub, padx=6).pack(side=tk.RIGHT)


    def _section_label(
        self,
        parent: ttk.Frame | tk.Frame,
        text: str,
        pady: tuple[int, int] = (0, 8),
    ) -> None:
        tk.Label(
            parent,
            text=text,
            bg="#101013",
            fg="#B8B8C0",
            font=(*self._sans_btn[:2], "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=pady)

    def _setting_row(self, parent: tk.Frame, label: str, value: tk.StringVar) -> tk.Frame:
        row = tk.Frame(parent, bg="#16161A", padx=12, pady=9)
        tk.Label(row, text=label, bg="#16161A", fg="#A1A1AA", font=self._sans_sub).pack(side=tk.LEFT)
        tk.Label(
            row,
            textvariable=value,
            bg="#16161A",
            fg="#FAFAFA",
            font=(*self._sans_btn[:2], "bold"),
        ).pack(side=tk.RIGHT)
        return row

    def _update_mode_label(self) -> None:
        self.mode_var.set("Live MCP" if self.mcp_var.get() else "Headless")

    def _start_rotation_loop(self) -> None:
        """Start the continuous spin loop. Safe to call multiple times."""
        if self._rotation_after_id is not None:
            self.root.after_cancel(self._rotation_after_id)
            self._rotation_after_id = None
        self._rotation_tick()

    def _rotation_tick(self) -> None:
        """Called every ~33 ms (~30 fps). Advances yaw when spinning is active."""
        if self.model_vertices is None:
            self._rotation_after_id = None
            return
        if self._auto_rotate:
            self.model_yaw -= 0.010
            self._render_model()
        self._rotation_after_id = self.root.after(33, self._rotation_tick)

    def _toggle_rotation(self) -> None:
        self._auto_rotate = not self._auto_rotate
        if self._rotate_btn:
            if self._auto_rotate:
                self._rotate_btn.configure(text="⏸", fg="#8B5CF6", bg="#171323")
            else:
                self._rotate_btn.configure(text="▶", fg="#9B9BA3", bg="#09090B")

    def _toggle_grid(self) -> None:
        self._show_grid = not getattr(self, "_show_grid", True)
        if hasattr(self, "_grid_btn"):
            if self._show_grid:
                self._grid_btn.configure(bg="#111114", fg="#8B5CF6",
                                         font=(*self._sans_sub[:2], "bold"))
            else:
                self._grid_btn.configure(bg="#111114", fg="#3F3F46",
                                         font=self._sans_sub)
        self._draw_preview_background()

    def _toggle_axes(self) -> None:
        self._show_axes = not getattr(self, "_show_axes", True)
        if hasattr(self, "_axes_btn"):
            if self._show_axes:
                self._axes_btn.configure(bg="#111114", fg="#8B5CF6",
                                         font=(*self._sans_sub[:2], "bold"))
            else:
                self._axes_btn.configure(bg="#111114", fg="#3F3F46",
                                         font=self._sans_sub)
        if self.model_vertices is not None:
            self._render_model()

    def _capture_snapshot(self) -> None:
        """Save the current viewport canvas as a PNG at ~4× scale via PostScript."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path("out") / "snapshots"
            out_dir.mkdir(parents=True, exist_ok=True)
            ps_path = out_dir / f"snapshot_{ts}.ps"
            png_path = out_dir / f"snapshot_{ts}.png"
            self.preview_canvas.postscript(file=str(ps_path), colormode="color")
            try:
                from PIL import Image
                img = Image.open(str(ps_path))
                img.save(str(png_path))
                ps_path.unlink(missing_ok=True)
                self._append_operator_log("SNAPSHOT", f"Saved → {png_path.name}")
            except Exception:
                self._append_operator_log("SNAPSHOT", f"PostScript saved → {ps_path.name}")
        except Exception as exc:
            self._append_operator_log("SNAPSHOT", f"Failed: {exc}")

    def _start_pan(self, event: tk.Event) -> None:
        self._pan_last: tuple[int, int] = (event.x, event.y)

    def _pan_model(self, event: tk.Event) -> None:
        """Right-click drag pans the view — shifts the camera anchor without rotating."""
        if not hasattr(self, "_pan_last"):
            return
        dx = event.x - self._pan_last[0]
        dy = event.y - self._pan_last[1]
        self._view_pan_x += dx
        self._view_pan_y += dy
        self._pan_last = (event.x, event.y)
        if self.model_vertices is not None:
            self._render_model()

    def _stop_pan(self, _event: tk.Event | None = None) -> None:
        pass

    def _reset_view_pan(self, _event: tk.Event | None = None) -> None:
        """Double-click resets pan so the model snaps back to screen centre."""
        self._view_pan_x = 0.0
        self._view_pan_y = 0.0
        if self.model_vertices is not None:
            self._render_model()

    def _show_generator_page(self) -> None:
        if self.dashboard_panel is not None:
            self.dashboard_panel.pack_forget()
        for w in (self.left_panel, self.right_panel, self.viewport_frame,
                  getattr(self, "_center_frame", None)):
            if w is not None:
                w.pack_forget()
        if self.right_panel is not None:
            self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        if not self._left_collapsed and self.left_panel is not None:
            self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
            if self.workbench_panel is not None and not self.workbench_panel.winfo_ismapped():
                self.workbench_panel.pack(fill=tk.BOTH, expand=True)
        if getattr(self, "_center_frame", None) is not None:
            self._center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        if self.viewport_frame is not None:
            self.viewport_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        if self._nav_workbench_btn:
            self._nav_workbench_btn.configure(bg="#1E1633", fg="#8B5CF6",
                                              font=(*self._sans_sub[:2], "bold"))
        if self._nav_dashboard_btn:
            self._nav_dashboard_btn.configure(bg=THEME["title"], fg="#71717A",
                                              font=(*self._sans_sub[:2],))

    def _show_dashboard_page(self) -> None:
        if self.left_panel is not None:
            self.left_panel.pack_forget()
        if self.viewport_frame is not None:
            self.viewport_frame.pack_forget()
        if self.right_panel is not None:
            self.right_panel.pack_forget()
        if getattr(self, "_center_frame", None) is not None:
            self._center_frame.pack_forget()
        if self.dashboard_panel is not None:
            self.dashboard_panel.pack(fill=tk.BOTH, expand=True)
        if self._nav_dashboard_btn:
            self._nav_dashboard_btn.configure(bg="#1E1633", fg="#8B5CF6",
                                              font=(*self._sans_sub[:2], "bold"))
        if self._nav_workbench_btn:
            self._nav_workbench_btn.configure(bg=THEME["title"], fg="#71717A",
                                              font=(*self._sans_sub[:2],))
        self._refresh_projects()

    def _build_dashboard_page(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, style="Viewport.TFrame", padding=(34, 26, 34, 14))
        header.pack(fill=tk.X)
        title_block = tk.Frame(header, bg="#0B0B0E")
        title_block.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            title_block,
            text="Asset Library",
            bg="#0B0B0E",
            fg="#FAFAFA",
            font=(*self._sans_title[:2], "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            title_block,
            text="Generated exports, previews, and reusable project folders.",
            bg="#0B0B0E",
            fg="#A1A1AA",
            font=self._sans_sub,
        ).pack(anchor=tk.W, pady=(4, 0))
        self._dash_stats_var = tk.StringVar(value="")
        tk.Label(
            title_block,
            textvariable=self._dash_stats_var,
            bg="#0B0B0E",
            fg="#7B7B83",
            font=self._mono,
        ).pack(anchor=tk.W, pady=(6, 0))
        self.dashboard_count_var = tk.StringVar(value="0 assets")
        tk.Label(
            header,
            textvariable=self.dashboard_count_var,
            bg="#18181C",
            fg="#FAFAFA",
            font=(*self._sans_btn[:2], "bold"),
            padx=14,
            pady=8,
        ).pack(side=tk.RIGHT, padx=(12, 0))
        ttk.Button(
            header,
            text="Refresh",
            style="Secondary.TButton",
            command=self._refresh_projects,
        ).pack(side=tk.RIGHT)
        self._dash_search_var = tk.StringVar()

        def _on_filter(*_a):
            self._dash_page = 0  # jump back to the first page on a new filter
            self._refresh_projects()
        self._dash_search_var.trace_add("write", _on_filter)
        search_box = tk.Frame(header, bg="#141417",
                              highlightthickness=1, highlightbackground="#26262C")
        search_box.pack(side=tk.RIGHT, padx=(0, 12))
        tk.Label(search_box, text="FILTER", bg="#141417", fg="#71717A",
                 font=(*self._sans_sub[:2], "bold"), padx=8).pack(side=tk.LEFT)
        tk.Entry(
            search_box,
            textvariable=self._dash_search_var,
            bg="#141417",
            fg="#FAFAFA",
            insertbackground="#FAFAFA",
            relief=tk.FLAT,
            font=self._sans_sub,
            width=22,
        ).pack(side=tk.LEFT, ipady=7, padx=(0, 8))

        canvas = tk.Canvas(parent, bg="#0B0B0E", highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(30, 8), pady=(0, 24))
        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 24))
        canvas.configure(yscrollcommand=scroll.set)

        self.projects_list_frame = tk.Frame(canvas, bg="#0B0B0E")
        list_window = canvas.create_window((0, 0), window=self.projects_list_frame, anchor=tk.NW)
        self.projects_list_frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(list_window, width=event.width),
        )

        def _dash_wheel(event: tk.Event) -> None:
            if self.dashboard_panel is not None and self.dashboard_panel.winfo_ismapped():
                canvas.yview_scroll(int(-event.delta / 120), "units")
        self.root.bind_all("<MouseWheel>", _dash_wheel, add="+")

    def _discover_projects(self) -> list[dict]:
        projects: list[dict] = []
        final_root = Path("out/final")
        if final_root.exists():
            for run_dir in final_root.iterdir():
                if not run_dir.is_dir():
                    continue
                previews = sorted(run_dir.glob("*_preview.png"))

                main_glb = ""
                blend = ""
                files: list[tuple[str, str, int]] = []
                for f in sorted(run_dir.iterdir()):
                    low = f.name.lower()
                    try:
                        size = f.stat().st_size
                    except OSError:
                        continue
                    if low.endswith("_lod1.glb"):
                        files.append(("LOD1", str(f), size))
                    elif low.endswith("_lod2.glb"):
                        files.append(("LOD2", str(f), size))
                    elif low.endswith("_collision.glb"):
                        files.append(("COLLISION", str(f), size))
                    elif low.endswith(".glb"):
                        files.append(("GLB", str(f), size))
                        main_glb = main_glb or str(f)
                    elif low.endswith(".fbx"):
                        files.append(("FBX", str(f), size))
                    elif low.endswith(".blend"):
                        files.append(("BLEND", str(f), size))
                        blend = blend or str(f)

                manifest: dict = {}
                manifest_path = run_dir / "manifest.json"
                if manifest_path.exists():
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except Exception:
                        manifest = {}

                projects.append(
                    {
                        "name": run_dir.name,
                        "display_name": self._format_project_name(run_dir.name),
                        "updated": run_dir.stat().st_mtime_ns,
                        "updated_label": datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%d %b %Y, %I:%M %p"),
                        "glb": main_glb,
                        "blend": blend,
                        "files": files,
                        "preview": str(previews[0]) if previews else "",
                        "path": str(run_dir),
                        "visual_score": manifest.get("visual_score"),
                        "topology_severity": manifest.get("topology_severity"),
                    }
                )
        projects.sort(key=lambda item: int(item["updated"]), reverse=True)
        return projects

    def _format_project_name(self, raw_name: str) -> str:
        name = re.sub(r"^\d{8}_\d{6}_", "", raw_name)
        match = re.match(r"^(.*?)(\(\d+\))?$", name)
        base = match.group(1) if match else name
        suffix = match.group(2) if match and match.group(2) else ""
        title = base.replace("_", " ").strip().title() or "Generated Asset"
        return f"{title} {suffix}".strip()

    def _refresh_projects(self) -> None:
        if self.projects_list_frame is None:
            return
        self.project_thumbnails.clear()
        for child in self.projects_list_frame.winfo_children():
            child.destroy()

        projects = self._discover_projects()

        query = ""
        if hasattr(self, "_dash_search_var"):
            query = self._dash_search_var.get().strip().lower()
        if query:
            projects = [
                p for p in projects
                if query in p["display_name"].lower() or query in p["name"].lower()
            ]

        if hasattr(self, "dashboard_count_var"):
            self.dashboard_count_var.set(f"{len(projects)} asset{'s' if len(projects) != 1 else ''}")
        if hasattr(self, "_dash_stats_var"):
            total_bytes = sum(f[2] for p in projects for f in p.get("files", []))
            with_lods = sum(1 for p in projects
                            if any(f[0] == "LOD1" for f in p.get("files", [])))
            with_coll = sum(1 for p in projects
                            if any(f[0] == "COLLISION" for f in p.get("files", [])))
            mb = total_bytes / 1_000_000
            self._dash_stats_var.set(
                f"{mb:,.1f} MB on disk   ·   {with_lods} with LODs   ·   "
                f"{with_coll} with collision   ·   game-ready exports"
            )
        if not projects:
            tk.Label(
                self.projects_list_frame,
                text="No generated projects yet.",
                bg="#0B0B0E",
                fg="#9B9BA3",
                font=self._sans_sub,
                padx=12,
                pady=16,
            ).pack(fill=tk.X)
            return

        # Fixed 4×4 matrix: 16 uniform cards per page, paginated.
        PER_ROW, ROWS = 4, 4
        per_page = PER_ROW * ROWS
        pages = max(1, (len(projects) + per_page - 1) // per_page)
        page = max(0, min(getattr(self, "_dash_page", 0), pages - 1))
        self._dash_page = page
        page_items = projects[page * per_page:(page + 1) * per_page]

        gap = 16
        avail = max(self.root.winfo_width() - 90, 700)
        card_w = max(210, (avail - (PER_ROW - 1) * gap - 8) // PER_ROW)

        grid = tk.Frame(self.projects_list_frame, bg="#0B0B0E")
        grid.pack(fill=tk.X, padx=4, pady=(0, 6))
        for c in range(PER_ROW):
            grid.grid_columnconfigure(c, weight=1, uniform="argus_cards")
        for idx, project in enumerate(page_items):
            r, c = divmod(idx, PER_ROW)
            self._project_card(project, parent=grid, width=card_w).grid(
                row=r, column=c, sticky="n",
                padx=(0, gap if c < PER_ROW - 1 else 0), pady=(0, gap))

        # Pagination footer (only when there's more than one page).
        if pages > 1:
            footer = tk.Frame(self.projects_list_frame, bg="#0B0B0E")
            footer.pack(pady=(2, 14))

            def _go(delta):
                self._dash_page = max(0, min(self._dash_page + delta, pages - 1))
                self._refresh_projects()

            def _nav(txt, delta, enabled):
                fg = "#FAFAFA" if enabled else "#3F3F46"
                bg = "#1E1633" if enabled else "#141417"
                b = tk.Label(footer, text=txt, bg=bg, fg=fg,
                             font=(*self._sans_sub[:2], "bold"), padx=16, pady=6,
                             cursor="hand2" if enabled else "arrow")
                if enabled:
                    b.bind("<Button-1>", lambda _e: _go(delta))
                return b

            _nav("‹ Prev", -1, page > 0).pack(side=tk.LEFT, padx=4)
            tk.Label(footer, text=f"Page {page + 1} / {pages}", bg="#0B0B0E",
                     fg="#A1A1AA", font=self._sans_sub, padx=12).pack(side=tk.LEFT)
            _nav("Next ›", +1, page < pages - 1).pack(side=tk.LEFT, padx=4)

    def _project_card(self, project: dict[str, str], parent: tk.Frame | None = None,
                      width: int = 296) -> tk.Frame:
        """Meshy-style gallery card at a FIXED uniform size: cover thumbnail on
        top, name + badges, then a quiet row of text-link actions. Every card
        in the 4×4 grid is identical in size regardless of content length."""
        _C = "#141417"
        tw = max(120, width - 2)
        th = int(tw * 0.52)
        card_h = th + 96
        card = tk.Frame(parent or self.projects_list_frame, bg=_C,
                        width=width, height=card_h,
                        highlightthickness=1, highlightbackground="#232329")
        card.pack_propagate(False)
        card.grid_propagate(False)

        thumb = tk.Canvas(card, width=tw, height=th, bg="#0E0E11",
                          highlightthickness=0, cursor="hand2")
        thumb.pack(side=tk.TOP)
        self._populate_project_thumbnail(thumb, project, tw, th)
        if project["glb"]:
            thumb.bind("<Button-1>",
                       lambda _e, p=project["glb"]: self._open_project_in_viewer(p))

        body = tk.Frame(card, bg=_C)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(8, 9))

        title_row = tk.Frame(body, bg=_C)
        title_row.pack(fill=tk.X)

        # Badges pack RIGHT first so a long title can't push them off the
        # fixed-width card — the title then fills and clips on the left.
        score = project.get("visual_score")
        if isinstance(score, (int, float)) and score >= 0:
            score_colors = {"ok": THEME["success_soft"], "run": THEME["warning"],
                            "fail": THEME["danger"]}
            tk.Label(
                title_row, text=f"{int(score)}/10", bg="#1A1A1F",
                fg=score_colors.get(self._score_kind(int(score)), THEME["muted_alt"]),
                font=(*self._sans_sub[:2], "bold"), padx=7, pady=1,
            ).pack(side=tk.RIGHT)
        severity = project.get("topology_severity")
        if severity:
            tk.Label(
                title_row, text=severity.upper(), bg="#1A1A1F",
                fg=self._severity_color(severity),
                font=(*self._sans_sub[:2], "bold"), padx=7, pady=1,
            ).pack(side=tk.RIGHT, padx=(0, 5))

        _name = project.get("display_name", project["name"])
        if len(_name) > 22:
            _name = _name[:21] + "…"
        tk.Label(
            title_row, text=_name, bg=_C, fg="#FAFAFA",
            font=(*self._sans_btn[:2], "bold"), anchor=tk.W,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        chip_colors = {
            "GLB": "#A78BFA", "FBX": "#C4B5FD", "BLEND": "#FBBF24",
            "LOD1": "#6EE7B7", "LOD2": "#34D399", "COLLISION": "#F87171",
        }
        meta = tk.Frame(body, bg=_C)
        meta.pack(fill=tk.X, pady=(4, 7))
        tk.Label(meta, text=project.get("updated_label", ""), bg=_C, fg="#71717A",
                 font=self._sans_sub, anchor=tk.W).pack(side=tk.LEFT)
        for chip_label, _fpath, _size in reversed(project.get("files", [])[:3]):
            tk.Label(meta, text=chip_label, bg=_C,
                     fg=chip_colors.get(chip_label, "#A1A1AA"),
                     font=(*self._sans_sub[:2], "bold")).pack(side=tk.RIGHT, padx=(7, 0))

        actions = tk.Frame(body, bg=_C)
        actions.pack(side=tk.BOTTOM, fill=tk.X)

        def _action(txt, cmd, danger=False):
            lbl = tk.Label(actions, text=txt, bg=_C, fg="#8E8E96",
                           font=self._sans_sub, cursor="hand2")
            lbl.pack(side=tk.LEFT, padx=(0, 14))
            hov = "#F87171" if danger else "#A78BFA"
            lbl.bind("<Enter>", lambda _e, w=lbl, c=hov: w.configure(fg=c))
            lbl.bind("<Leave>", lambda _e, w=lbl: w.configure(fg="#8E8E96"))
            lbl.bind("<Button-1>", lambda _e: cmd())

        if project["glb"]:
            _action("View 3D", lambda path=project["glb"]: self._open_web_viewer(path))
            _action("Load", lambda path=project["glb"]: self._open_project_in_viewer(path))
        if project.get("blend") or project.get("glb"):
            _action("Blender", lambda p=project: self._open_in_blender(p))
        _action("Folder", lambda path=project["path"]: subprocess.Popen(
            ["explorer", str(Path(path).resolve())]))
        _action("Delete", lambda name=project["name"]: self._delete_project(name),
                danger=True)
        return card

    def _open_in_blender(self, project: dict) -> None:
        """Open the asset in a full Blender GUI session.

        Prefers the .blend (quads, materials, lights intact); falls back to
        importing the GLB into a cleaned default scene.
        """
        from core.blender import BLENDER_PATH

        blender = str(BLENDER_PATH)
        if not Path(blender).exists():
            messagebox.showwarning(
                "Blender not found",
                f"BLENDER_PATH does not exist:\n{blender}\n\nSet it in your .env file.",
            )
            return

        blend = project.get("blend") or ""
        glb = project.get("glb") or ""
        try:
            if blend and Path(blend).exists():
                subprocess.Popen([blender, str(Path(blend).resolve())])
                self._append_operator_log(
                    "BLENDER", f"Opening {Path(blend).name} in Blender",
                    event_id=f"openblend:{blend}:{datetime.now().timestamp()}")
            elif glb and Path(glb).exists():
                expr = (
                    "import bpy; "
                    "bpy.ops.object.select_all(action='SELECT'); "
                    "bpy.ops.object.delete(use_global=False); "
                    f"bpy.ops.import_scene.gltf(filepath=r'{Path(glb).resolve()}')"
                )
                subprocess.Popen([blender, "--python-expr", expr])
                self._append_operator_log(
                    "BLENDER", f"Importing {Path(glb).name} into Blender",
                    event_id=f"openglb:{glb}:{datetime.now().timestamp()}")
            else:
                messagebox.showwarning(
                    "Nothing to open", "This project has no .blend or .glb file.")
        except OSError as exc:
            messagebox.showerror("Blender launch failed", str(exc))

    def _populate_project_thumbnail(self, canvas: tk.Canvas, project: dict[str, str],
                                    w: int = 190, h: int = 112) -> None:
        cx, cy0 = w // 2, h // 2
        preview_path = Path(project["preview"]) if project["preview"] else None
        if preview_path and preview_path.exists():
            if _PIL_AVAILABLE:
                try:
                    from PIL import ImageOps
                    img = ImageOps.fit(Image.open(str(preview_path)).convert("RGB"), (w, h))
                    photo = ImageTk.PhotoImage(img)
                    self.project_thumbnails.append(photo)
                    canvas.create_image(cx, cy0, image=photo, anchor=tk.CENTER)
                    return
                except Exception:
                    pass
            try:
                photo = tk.PhotoImage(file=str(preview_path))
                scale = max(1, math.ceil(photo.width() / (w - 10)), math.ceil(photo.height() / (h - 8)))
                if scale > 1:
                    photo = photo.subsample(scale, scale)
                self.project_thumbnails.append(photo)
                canvas.create_image(cx, cy0, image=photo, anchor=tk.CENTER)
                return
            except tk.TclError:
                pass

        glb_path = Path(project["glb"]) if project["glb"] else None
        if glb_path and glb_path.exists():
            try:
                import numpy as np
                import trimesh

                mesh = trimesh.load(glb_path, force="scene").to_geometry()
                vertices = np.asarray(mesh.vertices, dtype=float)
                faces = np.asarray(mesh.faces, dtype=int)
                if len(vertices) and len(faces):
                    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
                    vertices = vertices - center
                    span = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0
                    vertices = vertices / span

                    yaw = -0.65
                    pitch = 0.35
                    cy, sy = math.cos(yaw), math.sin(yaw)
                    cp, sp = math.cos(pitch), math.sin(pitch)
                    x = vertices[:, 0] * cy - vertices[:, 1] * sy
                    y = vertices[:, 0] * sy + vertices[:, 1] * cy
                    z = vertices[:, 2]
                    y2 = y * cp - z * sp
                    z2 = y * sp + z * cp
                    sx = cx + x * (w * 0.7)
                    sy2 = (cy0 + 4) - z2 * (w * 0.7)
                    order = np.argsort(y2[faces].mean(axis=1))
                    if len(order) > 450:
                        order = order[:: max(1, len(order) // 1200)]

                    for face_index in order:
                        face = faces[face_index]
                        coords: list[float] = []
                        for vertex_index in face:
                            coords.extend([float(sx[vertex_index]), float(sy2[vertex_index])])
                        canvas.create_polygon(coords, fill="#121215", outline="#8B5CF6", width=1)
                    return
            except Exception:
                pass

        canvas.create_text(cx, cy0, text="NO PREVIEW", fill="#71717A", font=self._sans_sub)

    def _open_project_in_viewer(self, glb_path: str) -> None:
        self._show_generator_page()
        self._reset_preview()
        self._load_model(Path(glb_path))

    def _open_web_viewer(self, glb_path: str) -> None:
        """Open the textured, rig-animated asset in a real WebGL viewer — the
        in-app canvas can't show materials or the spinning-wheel animation."""
        try:
            from core.web_viewer import open_3d_viewer
            if not open_3d_viewer(glb_path):
                messagebox.showwarning(
                    "3D Viewer", "Could not open the 3D viewer for this asset.")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("3D Viewer", str(exc))

    def _delete_project(self, project_name: str) -> None:
        confirmed = messagebox.askyesno(
            "Delete asset",
            (
                f"Delete '{project_name}'?\n\n"
                "This removes the exported asset and matching run files."
            ),
        )
        if not confirmed:
            return

        root = Path.cwd().resolve()
        targets = [
            root / "out" / "final" / project_name,
            root / "out" / "runs" / project_name,
        ]

        if self.model_path and project_name in self.model_path.parts:
            self._reset_preview()

        deleted_any = False
        locked_paths: list[str] = []

        for target in targets:
            resolved = target.resolve()
            allowed_roots = [
                (root / "out" / "final").resolve(),
                (root / "out" / "runs").resolve(),
            ]
            if not any(resolved == base or base in resolved.parents for base in allowed_roots):
                messagebox.showerror("Delete blocked", f"Unsafe delete path:\n{resolved}")
                return
            if not resolved.exists():
                continue
            try:
                shutil.rmtree(resolved)
                deleted_any = True
            except PermissionError as exc:
                locked_paths.append(str(resolved))

        if locked_paths:
            messagebox.showerror(
                "Delete partially failed",
                "These folders could not be deleted because a file inside is\n"
                "still open in another program (Blender, Explorer, etc.):\n\n"
                + "\n".join(locked_paths)
                + "\n\nClose the file in the other program and try again.",
            )

        self._refresh_projects()
        if deleted_any:
            self.status_var.set(f"Deleted {project_name}")
        elif not locked_paths:
            self.status_var.set("Project already removed")

    def _metric_card(self, parent: ttk.Frame, label: str, value: tk.StringVar) -> tk.Frame:
        card = tk.Frame(parent, bg="#131316", padx=12, pady=12)
        tk.Label(card, text=label, bg="#131316", fg="#8E8E96", font=self._sans_sub).pack(anchor=tk.W)
        tk.Label(card, textvariable=value, bg="#131316", fg="#FAFAFA", font=(*self._sans_btn[:2], "bold")).pack(anchor=tk.W, pady=(6, 0))
        return card

    def _draw_preview_background(self, _event: tk.Event | None = None) -> None:
        import math
        w = max(self.preview_canvas.winfo_width(), 2)
        h = max(self.preview_canvas.winfo_height(), 2)
        self.preview_canvas.delete("grid")

        self.preview_canvas.create_rectangle(0, 0, w, h, fill="#09090B", outline="", tags="grid")

        if self.model_vertices is not None and self._gl_renderer is not None:
            self.preview_canvas.coords(self._preview_hint_id, w // 2, int(h * 0.47))
            if self.preview_image_id is not None:
                self.preview_canvas.coords(self.preview_image_id, w // 2, h // 2)
            if self._gpu_frame_id is not None:
                self.preview_canvas.coords(self._gpu_frame_id, w // 2, h // 2)
            self._render_model()
            return

        if not getattr(self, "_show_grid", True):
            self.preview_canvas.create_text(22, 22, text="ARGUS VIEWPORT",
                fill="#3F3F46", font=self._mono, anchor=tk.W, tags="grid")
            self.preview_canvas.coords(self._preview_hint_id, w // 2, int(h * 0.47))
            if self.preview_image_id is not None:
                self.preview_canvas.coords(self.preview_image_id, w // 2, h // 2)
            if self.model_vertices is not None and self.model_faces is not None:
                self._render_model()
            return

        cam_h     = 3.5
        pitch     = 0.40
        fov_scale = h * 0.72
        cx, cy    = w / 2, h * 0.50

        sin_p = math.sin(pitch)
        cos_p = math.cos(pitch)

        def proj(wx: float, wy: float, wz: float):
            """Project a world point (wx, floor=0, wz) onto the canvas.
            Camera sits at world (0, cam_h, 0), looking down the +Z axis.
            Y-axis is up; pitch rotates the view downward.
            Returns (sx, sy) or None if behind the camera."""
            lx = wx
            ly = wy - cam_h
            lz = wz

            ry =  ly * cos_p + lz * sin_p
            rz = -ly * sin_p + lz * cos_p

            if rz <= 0.01:
                return None
            sx = cx + fov_scale * (lx / rz)
            sy = cy - fov_scale * (ry / rz)
            return sx, sy

        N        = 14
        CELL     = 1.0
        Z_NEAR   = 0.3
        Z_FAR    = N * CELL

        COL_MINOR = "#131316"
        COL_MAJOR = "#1E1E23"
        COL_AXIS  = "#26262C"

        for ix in range(-N, N + 1):
            x = ix * CELL
            color = COL_AXIS if ix == 0 else (COL_MAJOR if ix % 4 == 0 else COL_MINOR)
            pts: list[float] = []
            z = Z_NEAR
            while z <= Z_FAR + 1e-6:
                p = proj(x, 0.0, z)
                if p:
                    pts.extend(p)
                z += CELL / 6
            if len(pts) >= 4:
                self.preview_canvas.create_line(*pts, fill=color, width=1, smooth=False, tags="grid")

        for iz in range(0, N + 1):
            z = iz * CELL
            if z < Z_NEAR:
                continue
            color = COL_MAJOR if iz % 4 == 0 else COL_MINOR
            p0 = proj(-N * CELL, 0.0, z)
            p1 = proj( N * CELL, 0.0, z)
            if p0 and p1:
                self.preview_canvas.create_line(*p0, *p1, fill=color, width=1, tags="grid")

        horiz_y = cy
        for dy, alpha in ((0, "#161619"), (1, "#121215"), (2, "#0D0D10")):
            self.preview_canvas.create_line(0, int(horiz_y) - dy, w, int(horiz_y) - dy,
                                            fill=alpha, tags="grid")

        self.preview_canvas.create_text(
            22, 22,
            text="ARGUS VIEWPORT",
            fill="#3F3F46",
            font=self._mono,
            anchor=tk.W,
            tags="grid",
        )
        if self.model_path:
            self.preview_canvas.create_text(
                32,
                h - 34,
                text=self.model_path.name,
                fill="#8E8E96",
                font=self._mono,
                anchor=tk.W,
                tags="grid",
            )
        self.preview_canvas.coords(self._preview_hint_id, w // 2, int(h * 0.47))
        if self.preview_image_id is not None:
            self.preview_canvas.coords(self.preview_image_id, w // 2, h // 2)
        if self.model_vertices is not None and self.model_faces is not None:
            self._render_model()

    def _start_model_drag(self, event: tk.Event) -> None:
        self._model_drag_last = (event.x, event.y)
        self._drag_was_rotating = self._auto_rotate
        self._auto_rotate = False

    def _stop_model_drag(self, _event: tk.Event) -> None:
        self._model_drag_last = None
        if self._drag_was_rotating:
            self._auto_rotate = True
            if self._rotate_btn:
                self._rotate_btn.configure(text="⏸", fg="#8B5CF6", bg="#171323")

    def _drag_model(self, event: tk.Event) -> None:
        if self.model_vertices is None or self._model_drag_last is None:
            return
        last_x, last_y = self._model_drag_last
        self.model_yaw += (event.x - last_x) * 0.01
        self.model_pitch += (event.y - last_y) * 0.01
        self.model_pitch = max(-1.25, min(1.25, self.model_pitch))
        self._model_drag_last = (event.x, event.y)
        self._render_model()

    def _zoom_model(self, event: tk.Event) -> None:
        if self.model_vertices is None:
            return
        self.model_zoom *= 1.1 if event.delta > 0 else 0.9
        self.model_zoom = max(0.35, min(3.0, self.model_zoom))
        self._render_model()

    def _insert_quick_prompt(self, text: str) -> None:
        self._prompt_ph_clear()
        self.prompt_text.insert(tk.END, text)
        self.prompt_text.focus_set()

    def _get_prompt(self) -> str:
        if getattr(self, "_prompt_is_placeholder", False):
            return ""
        return self.prompt_text.get("1.0", "end-1c").strip()

    def _boot_check(self) -> None:
        self._append_operator_log("SYSTEM", "ARGUS studio initialized")
        venv = os.environ.get("VIRTUAL_ENV")
        expected_venv = (Path.cwd() / ".venv").resolve()
        active_local_venv = bool(venv and Path(venv).resolve() == expected_venv)
        stale_venv = bool(venv and not active_local_venv)
        self._append_operator_log(
            "RUNTIME",
            "Local environment ready" if active_local_venv else "Using external Python environment",
        )
        if stale_venv:
            self._append_operator_log(
                "WARN",
                "VIRTUAL_ENV points to another project; launch with run_desktop.bat",
            )
        try:
            from main import key_pool_status

            pools = key_pool_status()
            ready_count = sum(
                1
                for keys in pools.values()
                for key in keys
                if not key.get("exhausted") and key.get("cooling_for_sec", 0.0) <= 0
            )
            self._append_operator_log("PROVIDERS", f"{ready_count} generation key(s) available")
        except Exception as exc:  # noqa: BLE001
            self._append_operator_log("ERROR", f"Initialization failed: {exc}")
            if "No module named 'requests'" in str(exc):
                self._append_operator_log("ACTION", "Install requirements into this project's .venv")
            self.status_var.set("Initialization failed")
            self.status_badge_var.set("ERROR")
            self._set_badge_style("error")

    def _set_badge_style(self, kind: str) -> None:
        colors = {
            "idle": ("#C4B5FD", "#1A1A1F"),
            "run": ("#FBBF24", "#26200F"),
            "ok": ("#6EE7B7", "#12231B"),
            "fail": ("#F87171", "#271718"),
            "error": ("#F87171", "#271718"),
        }
        fg, bg = colors.get(kind, colors["idle"])
        self._badge_label.configure(fg=fg, bg=bg)

    def _start_output_pump(self) -> None:
        self._drain_output_queue()

    def _drain_output_queue(self) -> None:
        chunks: list[str] = []
        drained = 0
        while True:
            try:
                chunk = self.output_queue.get_nowait()
            except queue.Empty:
                break
            chunks.append(chunk)
            drained += 1
            if drained >= 200:
                break

        if chunks:
            text = "".join(chunks)
            self._detect_preview_path(text)
            self._append_output(text)

        self.root.after(20 if drained >= 200 else 120, self._drain_output_queue)

    def _detect_preview_path(self, chunk: str) -> None:
        self._preview_scan_buffer = (self._preview_scan_buffer + chunk)[-4000:]
        lines = self._preview_scan_buffer.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped:
                self._process_operator_log_line(stripped)
            if stripped.startswith("[STAGE "):
                self._update_stage_from_log(stripped)
            if stripped.startswith("[PREVIEW IMAGE]"):
                inline_path = stripped.removeprefix("[PREVIEW IMAGE]").strip()
                if inline_path:
                    self._load_preview_image(Path(inline_path))
                elif index + 1 < len(lines):
                    self._load_preview_image(Path(lines[index + 1].strip()))
                continue

            match = re.search(r"^Preview\s*:\s*(.+)$", stripped)
            if match:
                self._load_preview_image(Path(match.group(1).strip()))

            glb_match = re.search(r"([A-Za-z]:\\[^\r\n]+\.glb)\s*$", stripped)
            if glb_match:
                self._load_model(Path(glb_match.group(1)))

            final_match = re.search(r"^Final Asset\s*:\s*(.+\.glb)\s*$", stripped)
            if final_match:
                self._load_model(Path(final_match.group(1).strip()))

    def _process_operator_log_line(self, line: str) -> None:
        if line.startswith("Asset name"):
            self._append_operator_log("ASSET", line.split(":", 1)[-1].strip(), event_id=f"asset:{line}")
        elif line.startswith("Project folder"):
            self._append_operator_log("PROJECT", line.split(":", 1)[-1].strip(), event_id=f"project:{line}")
        elif line.startswith("Repair required"):
            verdict = line.split(":", 1)[-1].strip()
            self._append_operator_log("REVIEW", f"Repair required: {verdict}", event_id=f"repair_required:{line}")
            if hasattr(self, "_rp_repairs_lbl"):
                required = verdict.lower() == "yes"
                self._rp_repairs_lbl.configure(
                    text="Required" if required else "None",
                    fg=THEME["warning"] if required else "#34D399",
                )
        elif line.startswith("Poly budget"):
            if hasattr(self, "_rp_budget_lbl"):
                self._rp_budget_lbl.configure(text=line.split(":", 1)[-1].strip().title())
        elif line.startswith("Critic action"):
            action = line.split(":", 1)[-1].strip()
            action_labels = {
                "repair before execution": "Correction requested",
                "patch applied": "Correction applied",
                "repair unavailable; stopping before export": "No safe correction; export halted",
            }
            self._append_operator_log("REVIEW", action_labels.get(action, action), event_id=f"critic:{line}")
        elif line.startswith("Blender status"):
            self._append_operator_log("BLENDER", line.split(":", 1)[-1].strip(), event_id=f"blender:{line}")
        elif line.startswith("Topology status"):
            self._append_operator_log("TOPOLOGY", line.split(":", 1)[-1].strip(), event_id=f"topology:{line}")
        elif line.startswith("Severity"):
            sev = line.split(":", 1)[-1].strip()
            if hasattr(self, "_rp_topo_lbl"):
                self._rp_topo_lbl.configure(text=sev.title(), fg=self._severity_color(sev))
        elif line.startswith("N-gons"):
            val = line.split(":", 1)[-1].strip()
            if hasattr(self, "_rp_ngons_lbl"):
                self._rp_ngons_lbl.configure(
                    text=val, fg="#34D399" if val == "0" else THEME["warning"])
        elif line.startswith("Open edges"):
            val = line.split(":", 1)[-1].strip()
            if hasattr(self, "_rp_edges_lbl"):
                self._rp_edges_lbl.configure(
                    text=val, fg="#34D399" if val == "0" else THEME["warning"])
        elif line.startswith("Grounded"):
            grounded = line.split(":", 1)[-1].strip().lower() == "true"
            if hasattr(self, "_rp_ground_lbl"):
                self._rp_ground_lbl.configure(
                    text="Yes" if grounded else "No",
                    fg="#34D399" if grounded else THEME["danger"])
        elif line.startswith("Memory status"):
            self._append_operator_log("REFERENCE", line.split(":", 1)[-1].strip(), event_id=f"memory:{line}")
        elif line.startswith("Final Asset"):
            self._append_operator_log("COMPLETE", "Final asset exported", event_id=f"final:{line}")
        elif line.startswith("Run finished"):
            self._append_operator_log("RUN", line.replace("Run finished:", "").strip(), event_id=f"run:{line}")
        elif line.startswith("App error"):
            self._append_operator_log("ERROR", line.split(":", 1)[-1].strip(), event_id=f"app_error:{line}")
        elif line.startswith("Generation"):
            engine = line.split(":", 1)[-1].strip()
            short = "COMPILED (deterministic)" if "compiler" in engine else "LLM-written script"
            self._append_operator_log("ENGINE", short, event_id=f"engine:{line}")
        elif line.startswith("Build spec"):
            self._append_operator_log("SPEC", line.split(":", 1)[-1].strip(), event_id=f"spec:{line}")
        elif line.startswith("spec:"):
            self._append_operator_log("SPEC", line.split(":", 1)[-1].strip(), event_id=f"specwarn:{line}")
        elif line.startswith("[VISUAL IMPROVE"):
            m = re.match(r"\[VISUAL IMPROVE (\d+)/(\d+)\]", line)
            self._visual_iter = m.group(1) + "/" + m.group(2) if m else "?"
            self._append_operator_log("IMPROVE", f"Iteration {self._visual_iter} — regenerating",
                                      event_id=f"vimp:{self._visual_iter}")
        elif line.startswith("Visual score"):
            score = self._extract_score(line)
            if score is not None:
                self._set_score_badge(score)
                self._add_score_history(score)
                self._append_operator_log("SCORE", f"Visual score {score}/10",
                                          event_id=f"vscore:{line}",
                                          kind=self._score_kind(score))
        elif line.startswith("Candidate score"):
            score = self._extract_score(line)
            if score is not None:
                self._set_score_badge(score)
                self._add_score_history(score)
                it = getattr(self, "_visual_iter", "?")
                self._append_operator_log("SCORE", f"Iteration {it} scored {score}/10",
                                          event_id=f"cscore:{it}:{line}",
                                          kind=self._score_kind(score))
        elif line.startswith("Candidate") and "/10" in line:
            score = self._extract_score(line)
            if score is not None:
                self._add_score_history(score)
                self._append_operator_log("SCORE", line,
                                          event_id=f"bestofn:{line}",
                                          kind=self._score_kind(score))
        elif line.startswith("Best-of-N final"):
            m = re.search(r":\s*(-?\d+)/10\s*\(from (\d+) candidates\)", line)
            if m:
                score, count = int(m.group(1)), m.group(2)
                self._append_operator_log("SCORE", f"Best of {count}: {score}/10 selected",
                                          event_id=f"bestofn_final:{line}",
                                          kind=self._score_kind(score))
        elif line.startswith("Full regen") and "/10" in line:
            score = self._extract_score(line)
            if score is not None:
                self._set_score_badge(score)
                self._add_score_history(score)
                self._append_operator_log("SCORE", line,
                                          event_id=f"regen:{line}",
                                          kind=self._score_kind(score))
        elif line.startswith("Visual loop final"):
            score = self._extract_score(line)
            if score is not None:
                self._set_score_badge(score)
                self._add_score_history(score)
                self._append_operator_log("SCORE", f"Visual loop final: {score}/10",
                                          event_id=f"vfinal:{line}",
                                          kind=self._score_kind(score))
        elif line.startswith("Visual target"):
            self._append_operator_log("SCORE", line.split(":", 1)[-1].strip(),
                                      event_id=f"vtarget:{line}", kind="ok")
        elif line.startswith("Feedback"):
            fb = line.split(":", 1)[-1].strip()
            if fb:
                self._append_operator_log("CRITIC", fb[:120], event_id=f"fb:{line}")

    def _update_stage_from_log(self, line: str) -> None:
        stage_map = {
            "[STAGE 1]": ("PLAN", "Analyzing structure"),
            "[STAGE 2]": ("NAMING", "Preparing identity"),
            "[STAGE 3]": ("SCRIPT", "Generating script"),
            "[STAGE 4]": ("REVIEW", "Pre-export critique"),
            "[STAGE 5]": ("EXPORT", "Building and exporting"),
            "[STAGE 6]": ("TOPOLOGY", "Inspecting mesh health"),
            "[STAGE 7]": ("QUALITY", "Checking package"),
            "[STAGE 8]": ("REFERENCE", "Awaiting approval"),
            "[STAGE 54]": ("CANDIDATES", "Best-of-N candidates"),
            "[STAGE 55]": ("VISUAL", "Visual feedback loop"),
            "[STAGE 56]": ("REGEN", "Full regeneration"),
            "[STAGE 75]": ("VISUAL QA", "Final visual assessment"),
            "[STAGE 77]": ("PAINTER", "Applying diffusion textures"),
        }
        for prefix, (label, status) in stage_map.items():
            if line.startswith(prefix):
                self.stage_var.set(status)
                self._append_operator_log(label, status, event_id=f"stage:{prefix}")
                return

    @staticmethod
    def _extract_mesh_base_color(mesh) -> "list[float]":
        """Return [R, G, B] 0-1 from a trimesh mesh's material. Never raises."""
        try:
            import numpy as np
            vis = mesh.visual
            if hasattr(vis, "material"):
                mat = vis.material
                for attr in ("baseColorFactor", "main_color", "diffuse"):
                    c = getattr(mat, attr, None)
                    if c is not None:
                        c = np.asarray(c, dtype=float)
                        if c.max() > 1.0:
                            c = c / 255.0
                        rgb = np.clip(c[:3], 0.0, 1.0)
                        if rgb.min() > 0.88:
                            continue
                        return list(rgb)
            if hasattr(vis, "to_color"):
                fc = vis.to_color()
                vc = getattr(fc, "vertex_colors", None)
                if vc is not None and len(vc):
                    vc = np.asarray(vc, dtype=float)[:, :3]
                    if vc.max() > 1.0:
                        vc = vc / 255.0
                    mean = np.clip(vc.mean(axis=0), 0.0, 1.0)
                    if mean.min() <= 0.88:
                        return list(mean)
        except Exception:
            pass
        return [0.55, 0.58, 0.65]

    def _load_model(self, model_path: Path) -> None:
        if not model_path.exists():
            return
        # The pipeline overwrites the SAME glb path on every improvement
        # iteration, so dedupe by file signature, not path — otherwise the
        # viewer freezes on the first build and never shows the final asset.
        try:
            _st = model_path.stat()
            _sig = (_st.st_mtime_ns, _st.st_size)
        except OSError:
            return
        if model_path == self.model_path and _sig == getattr(self, "_model_sig", None):
            return

        try:
            import numpy as np
            import trimesh

            loaded = trimesh.load(model_path, force="scene")

            if isinstance(loaded, trimesh.Scene):
                mesh_list = [
                    (name, m) for name, m in loaded.geometry.items()
                    if isinstance(m, trimesh.Trimesh) and len(getattr(m, "vertices", [])) > 0
                ]
                if not mesh_list:
                    all_geoms = [
                        m for m in loaded.geometry.values()
                        if isinstance(m, trimesh.Trimesh) and len(getattr(m, "vertices", [])) > 0
                    ]
                    if not all_geoms:
                        all_geoms = [m for m in loaded.geometry.values() if hasattr(m, "vertices")]
                    if all_geoms:
                        try:
                            merged = trimesh.util.concatenate(all_geoms)
                        except Exception:
                            merged = all_geoms[0]
                        if len(getattr(merged, "vertices", [])) > 0:
                            mesh_list = [("merged", merged)]
            elif isinstance(loaded, trimesh.Trimesh) and len(getattr(loaded, "vertices", [])) > 0:
                mesh_list = [("mesh", loaded)]
            else:
                mesh_list = []

            self._append_output(f"\n[VIEWER] Scene has {len(getattr(loaded, 'geometry', {})) if hasattr(loaded, 'geometry') else 'N/A'} geometries, resolved {len(mesh_list)} mesh(es)\n")

            if not mesh_list:
                raise ValueError("No mesh geometry found in GLB")

            all_verts: list = []
            all_faces: list = []
            all_colors: list = []
            vert_offset = 0

            for _name, m in mesh_list:
                verts = np.asarray(m.vertices, dtype=np.float32)
                faces = np.asarray(m.faces, dtype=np.int32) + vert_offset
                color = self._extract_mesh_base_color(m)
                face_colors = np.tile(
                    np.array(color, dtype=np.float32), (len(faces), 1)
                )
                all_verts.append(verts)
                all_faces.append(faces)
                all_colors.append(face_colors)
                vert_offset += len(verts)

            vertices = np.concatenate(all_verts, axis=0)
            faces    = np.concatenate(all_faces, axis=0)
            colors   = np.concatenate(all_colors, axis=0)

            _v = vertices.copy()
            vertices[:, 0] =  _v[:, 0]
            vertices[:, 1] = -_v[:, 2]
            vertices[:, 2] =  _v[:, 1]

            center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
            vertices = vertices - center
            span = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
            vertices = vertices / (span if span > 0 else 1.0)
            self._model_floor_z: float = float(vertices[:, 2].min())

            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            lengths[lengths < 1e-8] = 1e-8
            normals = normals / lengths

            if len(faces) > 12000:
                step = max(1, math.ceil(len(faces) / 12000))
                faces   = faces[::step]
                colors  = colors[::step]
                normals = normals[::step]

        except Exception as exc:  # noqa: BLE001
            import traceback
            detail = traceback.format_exc()
            self._append_output(f"\n3D preview load failed: {exc}\n{detail}\n")
            self._append_operator_log("VIEWER", f"Load failed: {exc}")
            return

        self.model_vertices     = vertices
        self.model_faces        = faces
        self.model_face_colors  = colors
        self.model_face_normals = normals
        self.model_edge_vis     = None
        self.model_path         = model_path
        self._model_sig         = _sig
        self.density_var.set(f"{len(faces):,} POLY")
        self.hierarchy_var.set(f"{len(mesh_list)} MESH{'ES' if len(mesh_list) != 1 else ''}")
        self.asset_path_var.set(str(model_path))
        _mname = model_path.stem.replace("_", " ").title()
        self.model_name_var.set(_mname)
        self.primitive_count_var.set(f"  |  {len(faces):,} Primitives")
        self._update_mesh_stats_overlay(len(vertices), len(faces), len(mesh_list))
        self.preview_canvas.itemconfigure(self._preview_hint_id, state=tk.HIDDEN)
        if self.preview_image_id is not None:
            self.preview_canvas.delete(self.preview_image_id)
            self.preview_image_id = None
            self.preview_photo = None
        self._append_operator_log("VIEWER", f"Loaded {len(mesh_list)} mesh(es), {len(faces):,} faces", event_id=f"viewer:{model_path}")
        self._auto_rotate = True
        if self._rotate_btn:
            self._rotate_btn.configure(text="⏸", fg="#8B5CF6", bg="#171323")
        self._setup_gl_renderer()
        self._start_rotation_loop()


    def _setup_gl_renderer(self) -> None:
        """Create (or reuse) the GPU renderer and upload the current mesh."""
        if not _GL_RENDERER_AVAILABLE:
            return
        if self._gl_renderer is None:
            self._gl_renderer = _GLRenderer.create()
            if self._gl_renderer:
                self._append_operator_log(
                    "GPU",
                    f"Real-time renderer: {self._gl_renderer.gpu_name}",
                )
            else:
                self._append_operator_log("GPU", "GL context unavailable — using CPU renderer")
        if self._gl_renderer is not None and self.model_vertices is not None:
            self._gl_renderer.upload_mesh(
                self.model_vertices,
                self.model_faces,
                self.model_face_colors,
                self.model_face_normals,
                getattr(self, "model_edge_vis", None),
            )
            floor_z = getattr(self, "_model_floor_z", -0.5)
            self._gl_renderer.upload_floor(floor_z)

    def _display_gpu_frame(self, raw_rgba: bytes, w: int, h: int) -> None:
        """
        Convert raw RGBA bytes from the GPU FBO into a tkinter canvas image.
        OpenGL FBO is bottom-up; PIL flips it to screen-correct orientation.

        Uses tag "gpu_frame" (NOT "model") so that canvas.delete("model") never
        deletes this item — avoids the stale-ID bug where itemconfigure silently
        does nothing on a previously deleted item.
        """
        from PIL import Image, ImageTk
        arr = __import__("numpy").frombuffer(raw_rgba, dtype=__import__("numpy").uint8).reshape(h, w, 4)
        img = Image.fromarray(arr[::-1], "RGBA")
        photo = ImageTk.PhotoImage(img)
        self._gpu_frame_photo = photo

        if self._gpu_frame_id is None:
            self._gpu_frame_id = self.preview_canvas.create_image(
                w // 2, h // 2,
                image=photo, anchor=tk.CENTER,
                tags="gpu_frame",
            )
        else:
            self.preview_canvas.itemconfigure(self._gpu_frame_id, image=photo)
            self.preview_canvas.coords(self._gpu_frame_id, w // 2, h // 2)
        self.preview_canvas.tag_raise("gpu_frame")

    def _draw_axes_on_canvas(
        self,
        yaw_c: float, yaw_s: float,
        pit_c: float, pit_s: float,
        scale: float, scx: float, scy: float,
    ) -> None:
        """
        Draw the XYZ origin gizmo on the canvas (tagged 'model').
        The origin is planted at the bottom of the model (floor level) so the
        gizmo moves with the scene as you orbit.
        """
        if not getattr(self, "_show_axes", True):
            return
        import numpy as _np
        fz = float(getattr(self, "_model_floor_z", -0.35))
        ARM = 0.18
        _pts = _np.array([
            [0,    0,    fz],
            [ARM,  0,    fz],
            [0,    ARM,  fz],
            [0,    0,    fz + ARM],
        ], dtype=_np.float32)
        _ox  = _pts[:, 0] * yaw_c - _pts[:, 1] * yaw_s
        _oy  = _pts[:, 0] * yaw_s + _pts[:, 1] * yaw_c
        _oz  = _pts[:, 2]
        _oz2 = _oy * pit_s + _oz * pit_c
        _osx = scx + _ox * scale
        _osy = scy - _oz2 * scale
        for ai, col in enumerate(["#EF4444", "#34D399", "#8B5CF6"]):
            self.preview_canvas.create_line(
                float(_osx[0]), float(_osy[0]),
                float(_osx[ai + 1]), float(_osy[ai + 1]),
                fill=col, width=2, tags="model",
            )


    def _set_view_mode(self, mode: str) -> None:
        self.view_mode = mode
        if self._view_btn_wire:
            self._view_btn_wire.configure(bg="#1E1633", fg="#8B5CF6")
        if self.model_vertices is not None:
            self._render_model()

    def _render_model(self) -> None:
        if self.model_vertices is None or self.model_faces is None:
            return

        import numpy as np

        w = max(self.preview_canvas.winfo_width(), 2)
        h = max(self.preview_canvas.winfo_height(), 2)

        yaw_c = math.cos(self.model_yaw);  yaw_s = math.sin(self.model_yaw)
        pit_c = math.cos(self.model_pitch); pit_s = math.sin(self.model_pitch)
        scale = min(w, h) * 1.55 * self.model_zoom
        scx   = w * 0.5 + self._view_pan_x
        scy   = h * 0.5 + self._view_pan_y

        if self._gl_renderer is not None:
            try:
                raw = self._gl_renderer.render_frame(
                    self.model_yaw, self.model_pitch, self.model_zoom,
                    w, h, "mesh", self._view_pan_x, self._view_pan_y,
                    ambient=self._ambient_strength,
                    bg_brightness=self._bg_brightness,
                )
                if raw is not None:
                    self._display_gpu_frame(raw, w, h)
                    self.preview_canvas.delete("model")
                    self._draw_axes_on_canvas(yaw_c, yaw_s, pit_c, pit_s, scale, scx, scy)
                    return
            except Exception:
                pass

        verts = self.model_vertices
        rx  = verts[:, 0] * yaw_c - verts[:, 1] * yaw_s
        ry  = verts[:, 0] * yaw_s + verts[:, 1] * yaw_c
        rz  = verts[:, 2]
        ry2 = ry * pit_c - rz * pit_s
        rz2 = ry * pit_s + rz * pit_c

        screen_x = scx + rx   * scale
        screen_y = scy - rz2 * scale

        faces = self.model_faces
        face_depth = ry2[faces].mean(axis=1)
        order = np.argsort(face_depth)

        normals = self.model_face_normals
        ny2 = (normals[:, 0] * yaw_s + normals[:, 1] * yaw_c) * pit_c - normals[:, 2] * pit_s
        front_mask = ny2 < 0.05

        self.preview_canvas.delete("model")

        for fi in order:
            if not front_mask[fi]:
                continue
            face = faces[fi]
            coords = [c for vi in face for c in (float(screen_x[vi]), float(screen_y[vi]))]
            self.preview_canvas.create_polygon(coords, fill="#0B0B0E", outline="", tags="model")

        for fi in order:
            if not front_mask[fi]:
                continue
            face = faces[fi]
            x0, y0 = float(screen_x[face[0]]), float(screen_y[face[0]])
            x1, y1 = float(screen_x[face[1]]), float(screen_y[face[1]])
            x2, y2 = float(screen_x[face[2]]), float(screen_y[face[2]])
            self.preview_canvas.create_line(x0, y0, x1, y1, fill="#A78BFA", width=1, tags="model")
            self.preview_canvas.create_line(x1, y1, x2, y2, fill="#A78BFA", width=1, tags="model")
            self.preview_canvas.create_line(x2, y2, x0, y0, fill="#A78BFA", width=1, tags="model")

        self.preview_canvas.tag_raise("model")
        self._draw_axes_on_canvas(yaw_c, yaw_s, pit_c, pit_s, scale, scx, scy)

    def _load_preview_image(self, image_path: Path) -> None:
        if self.model_vertices is not None:
            return
        if not image_path.exists():
            return
        try:
            _st = image_path.stat()
            _sig = (_st.st_mtime_ns, _st.st_size)
        except OSError:
            return
        if image_path == self.preview_image_path and _sig == getattr(self, "_preview_sig", None):
            return

        try:
            photo = tk.PhotoImage(file=str(image_path))
        except tk.TclError as exc:
            self._append_output(f"\nPreview load failed: {exc}\n")
            self._append_operator_log("VIEWER", f"Preview load failed: {exc}")
            return

        w = max(self.preview_canvas.winfo_width(), 2)
        h = max(self.preview_canvas.winfo_height(), 2)
        scale = max(
            1,
            (photo.width() + w - 1) // w,
            (photo.height() + h - 1) // h,
        )
        if scale > 1:
            photo = photo.subsample(scale, scale)

        self.preview_photo = photo
        self.preview_image_path = image_path
        self._preview_sig = _sig

        if self.preview_image_id is None:
            self.preview_image_id = self.preview_canvas.create_image(
                w // 2,
                h // 2,
                image=self.preview_photo,
                anchor=tk.CENTER,
            )
        else:
            self.preview_canvas.itemconfigure(self.preview_image_id, image=self.preview_photo)
            self.preview_canvas.coords(self.preview_image_id, w // 2, h // 2)

        self.preview_canvas.itemconfigure(self._preview_hint_id, state=tk.HIDDEN)
        self.preview_canvas.tag_raise(self.preview_image_id)
        self._append_operator_log("VIEWER", "Preview image loaded", event_id=f"preview:{image_path}")

    def _append_output(self, text: str) -> None:
        try:
            with self.live_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(text)
        except OSError:
            pass

    def _append_operator_log(
        self,
        label: str,
        message: str,
        event_id: str | None = None,
        kind: str | None = None,
    ) -> None:
        if event_id:
            if event_id in self._operator_event_ids:
                return
            self._operator_event_ids.add(event_id)
        text = f"{label:<10} {message}\n"
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self._trim_visible_log()
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)
        try:
            self._log_stage_lbl.configure(text=label[:12])
            self._log_msg_lbl.configure(text=message[:90])
            upper_label = label.upper()
            if kind is not None:
                timeline_kind = kind
            elif upper_label in {"ERROR", "FAILED"}:
                timeline_kind = "error"
            elif upper_label in {"COMPLETE", "DONE", "VIEWER"} or "SUCCESS" in message.upper():
                timeline_kind = "ok"
            elif upper_label in {"RUN", "PROMPT"}:
                timeline_kind = "run"
            elif upper_label in {"PLAN", "NAMING", "SCRIPT", "REVIEW", "EXPORT", "QUALITY", "REFERENCE",
                                 "BLENDER", "TOPOLOGY", "CANDIDATES", "VISUAL", "REGEN", "ENGINE",
                                 "SPEC", "IMPROVE", "PAINTER"}:
                timeline_kind = "accent"
            else:
                timeline_kind = "idle"
            self._add_timeline_event(label, message, timeline_kind)
        except Exception:
            pass
        _stage_map = {
            "Planning":    "Planning",
            "STAGE 1":     "Planning",
            "STAGE 2":     "Asset Naming",
            "STAGE 3":     "Script Generation",
            "STAGE 4":     "Critic Review",
            "STAGE 5":     "Blender Execution",
            "STAGE 6":     "MCP Validation",
            "STAGE 7":     "Export",
            "STAGE 8":     "Structure Memory",
            "BLENDER":     "Blender Execution",
            "EXPORT":      "Export",
            "MCP":         "MCP Validation",
            "VIEWER":      "Export",
        }
        matched = _stage_map.get(label.strip())
        if matched and hasattr(self, "_pipeline_stage_labels"):
            for name, (dot, lbl) in self._pipeline_stage_labels.items():
                if name == matched:
                    dot.configure(fg="#8B5CF6")
                    lbl.configure(fg="#8B5CF6")

    def _trim_visible_log(self) -> None:
        end_line = int(self.output_text.index("end-1c").split(".", 1)[0])
        if end_line <= self.max_visible_log_lines:
            return
        delete_to = max(1, end_line - self.max_visible_log_lines)
        self.output_text.delete("1.0", f"{delete_to}.0")

    def on_run_clicked(self) -> None:
        if self.is_running:
            return

        prompt = self._get_prompt()
        if not prompt:
            messagebox.showwarning("Prompt required", "Enter a description in the prompt box.")
            return

        self.is_running = True
        self._run_started_at = datetime.now()
        self._update_mode_label()
        self._reset_preview()
        self.last_prompt_var.set(prompt)
        self._set_run_visual("busy")
        self.status_var.set("Generating...")
        self.status_badge_var.set("GENERATING")
        self.stage_var.set("Planning structure")
        self._set_badge_style("run")
        self.progress.pack(fill=tk.X, pady=(5, 0), before=self._progress_anchor)
        self.progress.start(10)
        self._operator_event_ids.clear()
        self._clear_timeline()
        self._visual_iter = "?"
        if hasattr(self, "_score_badge_lbl"):
            self._score_badge_lbl.configure(text="", fg=THEME["muted"])
        for _lbl_name in ("_rp_topo_lbl", "_rp_ngons_lbl", "_rp_edges_lbl",
                          "_rp_ground_lbl", "_rp_repairs_lbl"):
            _lbl = getattr(self, _lbl_name, None)
            if _lbl is not None:
                _lbl.configure(text="—", fg="#34D399")
        if hasattr(self, "_rp_budget_lbl"):
            self._rp_budget_lbl.configure(text="—", fg="#A1A1AA")
        if hasattr(self, "_score_history_frame"):
            for child in self._score_history_frame.winfo_children():
                child.destroy()
            self._score_history_count = 0
            self._score_history_placeholder = tk.Label(
                self._score_history_frame, text="—", bg="#0E0E11",
                fg=THEME["muted"], font=self._sans_sub)
            self._score_history_placeholder.pack(side=tk.LEFT)
        self.root.after(1000, self._tick_run_elapsed)
        if hasattr(self, "_pipeline_stage_labels"):
            for dot, lbl in self._pipeline_stage_labels.values():
                dot.configure(fg="#3F3F46")
                lbl.configure(fg="#3F3F46")
        self._append_operator_log("RUN", "Generation request accepted")
        self._append_operator_log("PROMPT", prompt[:140] + ("..." if len(prompt) > 140 else ""))
        self._append_output(f"\n=== New run ===\n{prompt}\n\n")

        worker = threading.Thread(
            target=self._run_pipeline_worker,
            args=(prompt, self.mcp_var.get()),
            daemon=True,
        )
        worker.start()

    def _run_pipeline_worker(self, prompt: str, mcp_mode: bool) -> None:
        try:
            from main import run_pipeline

            writer = _QueueWriter(self.output_queue)
            _rejection_reason: list[str] = []
            _parts_hint: list[list[str]] = []
            _orig_write = writer.write
            def _intercepting_write(text: str) -> None:
                if text.startswith("[ARGUS] Request rejected:"):
                    _rejection_reason.append(text.strip())
                if text.startswith("[GRAPH_PARTS]"):
                    import json as _json
                    try:
                        _parts_hint.append(_json.loads(text[len("[GRAPH_PARTS]"):].strip()))
                    except Exception:
                        pass
                _orig_write(text)
            writer.write = _intercepting_write

            _mem_cb = (lambda ctx: True) if self._struct_mode \
                      else self._confirm_structure_memory_save

            with redirect_stdout(writer), redirect_stderr(writer):
                success = run_pipeline(
                    prompt=prompt,
                    poly_budget=self._poly_var.get().lower(),
                    mcp_mode=mcp_mode,
                    use_concept_pipeline=self.concept_pipeline_var.get(),
                    memory_approval_callback=_mem_cb,
                )

            if _parts_hint:
                self.root.after(0, lambda p=_parts_hint[-1]: self._populate_component_graph(p))

            rejected = bool(_rejection_reason)
            if rejected:
                reason_text = _rejection_reason[0].replace("[ARGUS] Request rejected: ", "")
                self.output_queue.put(f"\nRun finished: REJECTED\n\n")
                self.root.after(0, lambda: self.status_var.set("Rejected"))
                self.root.after(0, lambda: self.status_badge_var.set("REJECTED"))
                self.root.after(0, lambda r=reason_text: self.stage_var.set(f"Rejected: {r}"))
                self.root.after(0, lambda: self._set_badge_style("fail"))
            else:
                self.output_queue.put(
                    "\nRun finished: SUCCESS\n\n" if success else "\nRun finished: FAILED\n\n"
                )
                self.root.after(0, lambda: self.status_var.set("Complete" if success else "Failed"))
                self.root.after(
                    0,
                    lambda: self.status_badge_var.set("DONE" if success else "FAILED"),
                )
                self.root.after(
                    0,
                    lambda: self.stage_var.set("Complete" if success else "Generation failed"),
                )
                self.root.after(0, lambda: self._set_badge_style("ok" if success else "fail"))
        except Exception as exc:  # noqa: BLE001
            self.output_queue.put(f"\nApp error: {exc}\n\n")
            self.root.after(0, lambda: self.status_var.set("Error"))
            self.root.after(0, lambda: self.status_badge_var.set("ERROR"))
            self.root.after(0, lambda: self._set_badge_style("error"))
        finally:
            self.root.after(0, self._reset_run_state)

    def _confirm_structure_memory_save(self, context: dict) -> bool:
        decision: dict[str, bool] = {"approved": False}
        ready = threading.Event()

        def ask_user() -> None:
            blueprint = context.get("blueprint") or "matched blueprint"
            asset_path = context.get("asset_path") or "the exported asset"
            preview_path = context.get("preview_path") or "the generated preview"
            dialog = tk.Toplevel(self.root)
            dialog.title("Save structure reference")
            dialog.configure(bg="#101013")
            dialog.resizable(False, False)
            dialog.transient(self.root)
            dialog.grab_set()

            shell = tk.Frame(dialog, bg="#101013", padx=22, pady=20)
            shell.pack(fill=tk.BOTH, expand=True)

            tk.Label(
                shell,
                text="Save as Future Reference?",
                bg="#101013",
                fg="#FAFAFA",
                font=self._sans_heading,
            ).pack(anchor=tk.W)
            tk.Label(
                shell,
                text="Approve only if the structure, attachments, and silhouette look correct.",
                bg="#101013",
                fg="#A8A8B0",
                font=self._sans_sub,
                wraplength=420,
                justify=tk.LEFT,
            ).pack(anchor=tk.W, pady=(6, 14))

            preview_box = tk.Frame(shell, bg="#16161A", width=420, height=236)
            preview_box.pack(fill=tk.X)
            preview_box.pack_propagate(False)

            preview_file = Path(preview_path) if preview_path else None
            if preview_file and preview_file.exists():
                try:
                    photo = tk.PhotoImage(file=str(preview_file))
                    scale = max(1, math.ceil(photo.width() / 400), math.ceil(photo.height() / 216))
                    if scale > 1:
                        photo = photo.subsample(scale, scale)
                    dialog.preview_photo = photo
                    tk.Label(preview_box, image=photo, bg="#16161A").pack(expand=True)
                except tk.TclError:
                    tk.Label(
                        preview_box,
                        text="Preview unavailable",
                        bg="#16161A",
                        fg="#8E8E96",
                        font=self._sans_sub,
                    ).pack(expand=True)
            else:
                tk.Label(
                    preview_box,
                    text="Preview unavailable",
                    bg="#16161A",
                    fg="#8E8E96",
                    font=self._sans_sub,
                ).pack(expand=True)

            meta = tk.Frame(shell, bg="#101013")
            meta.pack(fill=tk.X, pady=(14, 16))
            self._dialog_meta(meta, "Blueprint", blueprint).pack(fill=tk.X, pady=(0, 6))
            self._dialog_meta(meta, "Asset", asset_path).pack(fill=tk.X)

            actions = tk.Frame(shell, bg="#101013")
            actions.pack(fill=tk.X)

            def close(approved: bool) -> None:
                decision["approved"] = approved
                dialog.grab_release()
                dialog.destroy()
                ready.set()

            ttk.Button(
                actions,
                text="Skip",
                style="Secondary.TButton",
                command=lambda: close(False),
            ).pack(side=tk.RIGHT)
            ttk.Button(
                actions,
                text="Save Reference",
                style="Primary.TButton",
                command=lambda: close(True),
            ).pack(side=tk.RIGHT, padx=(0, 10))

            dialog.protocol("WM_DELETE_WINDOW", lambda: close(False))
            dialog.update_idletasks()
            x = self.root.winfo_x() + max(0, (self.root.winfo_width() - dialog.winfo_width()) // 2)
            y = self.root.winfo_y() + max(0, (self.root.winfo_height() - dialog.winfo_height()) // 2)
            dialog.geometry(f"+{x}+{y}")

        self.output_queue.put(
            "\nStructure memory candidate is ready. Waiting for your approval...\n"
        )
        self.root.after(0, ask_user)
        ready.wait()
        self.output_queue.put(
            "Structure memory approval: yes\n"
            if decision["approved"]
            else "Structure memory approval: no\n"
        )
        return decision["approved"]

    def _dialog_meta(self, parent: tk.Frame, label: str, value: str) -> tk.Frame:
        row = tk.Frame(parent, bg="#101013")
        tk.Label(
            row,
            text=label,
            bg="#101013",
            fg="#A1A1AA",
            font=self._sans_sub,
            width=10,
            anchor=tk.W,
        ).pack(side=tk.LEFT)
        tk.Label(
            row,
            text=value,
            bg="#101013",
            fg="#FAFAFA",
            font=self._mono,
            anchor=tk.W,
            wraplength=330,
            justify=tk.LEFT,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        return row

    def _round_rect_photo(self, width: int, height: int, radius: int,
                          c1: str, c2: str | None = None,
                          key: str | None = None):
        """Rounded-rectangle PhotoImage, optionally with a horizontal
        gradient — Tk has no rounded corners or gradients of its own, so the
        hero button is pre-rendered through PIL. Returns None without PIL."""
        if not _PIL_AVAILABLE or width < 2 or height < 2:
            return None
        ss = 3  # supersample for smooth corners
        w, h, r = width * ss, height * ss, radius * ss
        if c2:
            ca = tuple(int(c1[i:i + 2], 16) for i in (1, 3, 5))
            cb = tuple(int(c2[i:i + 2], 16) for i in (1, 3, 5))
            row = [tuple(int(ca[k] + (cb[k] - ca[k]) * x / max(w - 1, 1))
                         for k in range(3)) for x in range(w)]
            base = Image.new("RGB", (w, 1))
            base.putdata(row)
            base = base.resize((w, h))
        else:
            base = Image.new("RGB", (w, h), c1)
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=r, fill=255)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        img.paste(base, (0, 0), mask)
        photo = ImageTk.PhotoImage(img.resize((width, height), Image.LANCZOS))
        if key is not None:
            self._btn_photos[key] = photo  # hold a reference or Tk drops it
        return photo

    def _set_run_visual(self, state: str) -> None:
        """Switch the generate button between normal / hover / busy looks."""
        text = "⏳  GENERATING..." if state == "busy" else "⚡  GENERATE ASSET"
        fg = "#8E8E96" if state == "busy" else "#FFFFFF"
        img = getattr(self, "_btn_photos", {}).get(f"run_{state}")
        if img is not None:
            self.run_button.configure(image=img, text=text, fg=fg,
                                      bg=THEME["panel"], pady=0)
        else:
            flat = {"normal": "#8B5CF6", "hover": "#A78BFA", "busy": "#2E2447"}
            self.run_button.configure(image="", text=text, fg=fg,
                                      bg=flat.get(state, "#8B5CF6"), pady=10)

    def _reset_run_state(self) -> None:
        self.is_running = False
        self._set_run_visual("normal")
        self.progress.stop()
        self.progress.pack_forget()

    def _clear_output(self) -> None:
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state=tk.DISABLED)
        try:
            self.live_log_path.write_text("", encoding="utf-8")
        except OSError:
            pass
        self._operator_event_ids.clear()
        self._append_operator_log("SYSTEM", "Operator log cleared")

    def _reset_preview(self) -> None:
        self.preview_photo = None
        self.preview_image_path = None
        self.model_vertices     = None
        self.model_faces        = None
        self.model_face_colors  = None
        self.model_face_normals = None
        self.model_edge_vis     = None
        self.model_path         = None
        self.model_yaw   = -0.65
        self.model_pitch = 0.35
        self.model_zoom  = 1.0
        self._model_drag_last = None
        self._view_pan_x = 0.0
        self._view_pan_y = 0.0
        self.density_var.set("0 POLY")
        self.hierarchy_var.set("—")
        self.asset_path_var.set("No asset loaded")
        self._preview_scan_buffer = ""
        self._model_floor_z = -0.35
        self.preview_canvas.delete("model")
        if self.preview_image_id is not None:
            self.preview_canvas.delete(self.preview_image_id)
            self.preview_image_id = None
        self._gpu_frame_photo = None
        if self._gpu_frame_id is not None:
            self.preview_canvas.delete(self._gpu_frame_id)
            self._gpu_frame_id = None
        self.preview_canvas.itemconfigure(self._preview_hint_id, state=tk.NORMAL)

    def _open_output_folder(self) -> None:
        output_dir = Path("out").resolve()
        if not output_dir.exists():
            messagebox.showinfo("Output folder", f"No folder yet:\n{output_dir}")
            return
        subprocess.Popen(["explorer", str(output_dir)])  # noqa: S603

    def _open_cmd_log_window(self) -> None:
        self.live_log_path.parent.mkdir(exist_ok=True)
        self.live_log_path.touch(exist_ok=True)
        log_path = str(self.live_log_path.resolve()).replace("'", "''")
        command = (
            "title ARGUS Live Log & "
            "powershell -NoProfile -ExecutionPolicy Bypass "
            f"-Command \"Get-Content -LiteralPath '{log_path}' -Wait\""
        )
        subprocess.Popen(["cmd.exe", "/k", command])  # noqa: S603

    def _reuse_last_prompt(self) -> None:
        last_prompt = self.last_prompt_var.get().strip()
        if not last_prompt:
            messagebox.showinfo("Last prompt", "No previous prompt yet.")
            return
        self._prompt_ph_clear()
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", last_prompt)


def main() -> None:
    root = tk.Tk()
    ArgusDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
