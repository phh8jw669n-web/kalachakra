"""Unified CLI: ``python -m kalachakra.decoupled_engine <command>``.

Commands: ``train``, ``eval``, ``export``, ``serve`` -- thin wrappers over the
module's training loop, inference engine, exporter and FastAPI server.
"""

from __future__ import annotations

import argparse


def _cmd_train(args) -> int:
    from .config import EngineConfig
    from .training import train
    cfg = EngineConfig()
    cfg.train.out_dir = args.out_dir
    cfg.train.max_steps = args.steps
    cfg.train.batch_size = args.batch
    if args.device:
        cfg.train.device = args.device
    if args.timeline_start is not None:
        from ..ephemeris.calendar import parse_datetime
        cfg.data.start_jd = parse_datetime(args.timeline_start)
    if args.timeline_end is not None:
        from ..ephemeris.calendar import parse_datetime
        cfg.data.end_jd = parse_datetime(args.timeline_end)
    train(cfg, num_workers=args.workers, ephe_path=args.ephe_path,
          jpl_file=args.jpl_file)
    return 0


def _cmd_eval(args) -> int:
    from .inference import DecoupledInference
    eng = DecoupledInference.from_checkpoint(args.checkpoint, device=args.device,
                                             ephe_path=args.ephe_path,
                                             jpl_file=args.jpl_file)
    if args.lat is not None and args.lon is not None:
        r = eng.pinpoint(args.timestamp, args.lat, args.lon)
        print(f"OKLab={r['oklab']}  RGB={tuple(int(x) for x in r['rgb8'])}")
        top = sorted(r["attribution"].items(), key=lambda kv: -kv[1])[:5]
        print("top attribution:", ", ".join(f"{n} {w:.1%}" for n, w in top))
    else:
        tex = eng.global_texture(args.timestamp, width=args.width, height=args.height)
        print(f"texture {tex['width']}x{tex['height']} "
              f"({len(tex['bytes'])} bytes) OKLab mean L={tex['oklab'][..., 0].mean():.3f}")
    return 0


def _cmd_export(args) -> int:
    from .export import export_from_checkpoint
    written = export_from_checkpoint(args.checkpoint, args.out_dir, fmt=args.format,
                                     device=args.device or "cpu")
    print("exported:", written)
    return 0


def _cmd_serve(args) -> int:
    from .api import serve
    return serve(args.checkpoint, host=args.host, port=args.port, device=args.device,
                 bank_size=args.bank_size, ephe_path=args.ephe_path,
                 jpl_file=args.jpl_file)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kalachakra.decoupled_engine", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train the Sky Encoder + Earth Lens")
    t.add_argument("--out-dir", default="checkpoints/decoupled")
    t.add_argument("--steps", type=int, default=5000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--device", default="")
    t.add_argument("--timeline-start", default=None)
    t.add_argument("--timeline-end", default=None)
    t.add_argument("--ephe-path", default=None)
    t.add_argument("--jpl-file", default=None)
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("eval", help="evaluate a checkpoint (texture or pinpoint)")
    e.add_argument("checkpoint")
    e.add_argument("--timestamp", default="now")
    e.add_argument("--lat", type=float, default=None)
    e.add_argument("--lon", type=float, default=None)
    e.add_argument("--width", type=int, default=512)
    e.add_argument("--height", type=int, default=256)
    e.add_argument("--device", default="")
    e.add_argument("--ephe-path", default=None)
    e.add_argument("--jpl-file", default=None)
    e.set_defaults(func=_cmd_eval)

    x = sub.add_parser("export", help="export to TorchScript / ONNX")
    x.add_argument("checkpoint")
    x.add_argument("--out-dir", default="exports/decoupled")
    x.add_argument("--format", choices=["torchscript", "onnx", "both"],
                   default="torchscript")
    x.add_argument("--device", default="")
    x.set_defaults(func=_cmd_export)

    s = sub.add_parser("serve", help="run the FastAPI live-integration server")
    s.add_argument("checkpoint")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8100)
    s.add_argument("--device", default="")
    s.add_argument("--bank-size", type=int, default=64)
    s.add_argument("--ephe-path", default=None)
    s.add_argument("--jpl-file", default=None)
    s.set_defaults(func=_cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
