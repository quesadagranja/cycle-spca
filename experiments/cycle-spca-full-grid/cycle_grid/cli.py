from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import multiprocessing as mp
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

from .config import all_tasks, load_config, total_fits
from .dashboard import write_dashboard
from .database import (
    database_counts,
    export_components_csv,
    export_failures_csv,
    export_fits_csv,
    export_stability_csv,
    load_fit_records_enriched,
    sync_fit_records,
    sync_stability_records,
)
from .experiment import (
    RunnerLock,
    clear_stale_running_files,
    initialize_experiment,
    inspect_dataset,
    inspect_repository,
)
from .fit_worker import initialize_fit_worker, run_fit_task
from .stability import (
    build_stability_tasks,
    initialize_stability_worker,
    run_stability_task,
)


def _refresh_once(
    config: dict[str, Any],
    *,
    export_components: bool,
    include_plots: bool,
) -> None:
    output = Path(config["output_dir"])
    sync_fit_records(output)
    sync_stability_records(output)
    export_fits_csv(output)
    export_failures_csv(output)
    export_stability_csv(output)
    if export_components:
        export_components_csv(output)
    write_dashboard(
        output,
        config,
        total=total_fits(config),
        include_plots=include_plots,
    )


def _monitor_loop(config: dict[str, Any], stop_event) -> None:
    refresh = max(5, int(config["runtime"]["refresh_seconds"]))
    plot_refresh = max(refresh, int(config["runtime"]["plot_refresh_seconds"]))
    component_refresh = max(
        refresh, int(config["runtime"]["components_export_seconds"])
    )
    next_plots = 0.0
    next_components = 0.0
    while True:
        now = time.monotonic()
        try:
            _refresh_once(
                config,
                export_components=now >= next_components,
                include_plots=now >= next_plots,
            )
            if now >= next_plots:
                next_plots = now + plot_refresh
            if now >= next_components:
                next_components = now + component_refresh
        except Exception as error:
            print(f"[monitor] refresh failed: {error}", file=sys.stderr, flush=True)
        if stop_event.wait(refresh):
            break


def _start_monitor(context, config: dict[str, Any]):
    stop_event = context.Event()
    process = context.Process(
        target=_monitor_loop,
        args=(config, stop_event),
        name="cycle-grid-monitor",
        daemon=True,
    )
    process.start()
    return stop_event, process


def _stop_monitor(stop_event, process) -> None:
    stop_event.set()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)


def _context(config: dict[str, Any]):
    requested = config["runtime"].get("start_method", "fork")
    methods = mp.get_all_start_methods()
    if requested not in methods:
        requested = methods[0]
    return mp.get_context(requested)


def _print_preflight(config: dict[str, Any], workers: int, pending: int) -> None:
    output = Path(config["output_dir"])
    stat = os.statvfs(output)
    free_gib = stat.f_bavail * stat.f_frsize / (1024**3)
    available_inodes = stat.f_favail
    maximum_png = (
        len(config["sample_seeds"])
        * len(config["N_values"])
        * len(config["lambda_l1_values"])
        * len(config["lambda_tv_values"])
        * sum(map(int, config["K_values"]))
    )
    # One leaf directory contains six data files, DONE, and K PNGs.  This is
    # intentionally conservative and includes room for parent directories and
    # the flat completion/stability records.
    estimated_inodes = maximum_png + 9 * total_fits(config)
    print("CycleSPCA full grid", flush=True)
    print(f"  output:      {output}", flush=True)
    print(f"  total fits:  {total_fits(config):,}", flush=True)
    print(f"  pending:     {pending:,}", flush=True)
    print(f"  workers:     {workers}", flush=True)
    print(f"  maximum PNG: {maximum_png:,}", flush=True)
    print(f"  CPU visible: {os.cpu_count()}", flush=True)
    print(f"  disk free:   {free_gib:,.1f} GiB", flush=True)
    print(f"  free inodes: {available_inodes:,}", flush=True)
    print(f"  est. inodes: {estimated_inodes:,}", flush=True)
    print(
        "  live tables: " + str(output / "tables" / "fits.csv"), flush=True
    )
    print("  dashboard:   " + str(output / "dashboard.html"), flush=True)
    if available_inodes > 0 and available_inodes < estimated_inodes:
        raise RuntimeError(
            f"The output filesystem exposes only {available_inodes:,} free inodes, "
            f"but this configuration may need about {estimated_inodes:,}. Choose "
            "another output filesystem before starting."
        )


def run_grid(config: dict[str, Any], *, workers: int, force_lock: bool) -> int:
    output = Path(config["output_dir"])
    context = _context(config)
    interrupted = False
    with RunnerLock(output, force=force_lock):
        initialized = initialize_experiment(config)
        clear_stale_running_files(output)
        sync_fit_records(output)
        completed = {path.stem for path in (output / "records").glob("*.json")}
        tasks = [
            task
            for task in all_tasks(config, initialized["samples"])
            if task["fit_id"] not in completed
        ]
        _print_preflight(config, workers, len(tasks))
        _refresh_once(config, export_components=False, include_plots=False)

        if tasks:
            stop_event, monitor = _start_monitor(context, config)
            completed_this_run = 0
            failed_this_run = 0
            last_print = time.monotonic()
            pool = context.Pool(
                processes=workers,
                initializer=initialize_fit_worker,
                initargs=(config,),
                maxtasksperchild=int(config["runtime"]["maxtasksperchild"]),
            )
            try:
                for result in pool.imap_unordered(run_fit_task, tasks, chunksize=1):
                    if result["status"] in {"done", "already_done"}:
                        completed_this_run += 1
                    else:
                        failed_this_run += 1
                    now = time.monotonic()
                    if now - last_print >= 30 or (completed_this_run + failed_this_run) == len(tasks):
                        print(
                            f"[fits] processed {completed_this_run + failed_this_run:,}/"
                            f"{len(tasks):,} in this run; failures={failed_this_run:,}",
                            flush=True,
                        )
                        last_print = now
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                interrupted = True
                print("Interrupted: terminating workers after preserving completed fits.", flush=True)
                pool.terminate()
                pool.join()
            finally:
                _stop_monitor(stop_event, monitor)
                _refresh_once(config, export_components=False, include_plots=True)

        if interrupted:
            return 130

        completed_total = len(list((output / "records").glob("*.json")))
        if config["stability"].get("enabled", True) and completed_total:
            _run_stability_locked(config, workers=int(config["stability"].get("workers", workers)))

        _refresh_once(config, export_components=True, include_plots=True)
        print_status(config)
        return 0


def _run_stability_locked(config: dict[str, Any], *, workers: int) -> None:
    output = Path(config["output_dir"])
    sync_fit_records(output)
    records = load_fit_records_enriched(output)
    tasks = build_stability_tasks(records)
    completed = {path.stem for path in (output / "stability_records").glob("*.json")}
    pending = [task for task in tasks if task["pair_id"] not in completed]
    print(
        f"[stability] total available pairs={len(tasks):,}; pending={len(pending):,}; "
        f"workers={workers}",
        flush=True,
    )
    if not pending:
        return
    context = _context(config)
    stop_event, monitor = _start_monitor(context, config)
    pool = context.Pool(
        processes=workers,
        initializer=initialize_stability_worker,
        initargs=(str(output),),
        maxtasksperchild=int(config["runtime"]["maxtasksperchild"]),
    )
    done = 0
    failed = 0
    last_print = time.monotonic()
    try:
        for result in pool.imap_unordered(run_stability_task, pending, chunksize=1):
            if result["status"] in {"done", "already_done"}:
                done += 1
            else:
                failed += 1
            now = time.monotonic()
            if now - last_print >= 30 or done + failed == len(pending):
                print(
                    f"[stability] processed {done + failed:,}/{len(pending):,}; "
                    f"failures={failed:,}",
                    flush=True,
                )
                last_print = now
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise
    finally:
        _stop_monitor(stop_event, monitor)
        sync_stability_records(output)


def run_stability(config: dict[str, Any], *, workers: int, force_lock: bool) -> int:
    output = Path(config["output_dir"])
    with RunnerLock(output, force=force_lock):
        initialize_experiment(config)
        _run_stability_locked(config, workers=workers)
        _refresh_once(config, export_components=False, include_plots=True)
    return 0


def print_status(config: dict[str, Any]) -> None:
    output = Path(config["output_dir"])
    total = total_fits(config)
    records = list((output / "records").glob("*.json"))
    completed = len(records)
    running = len(list((output / "running").glob("*.json")))
    failure_paths = list((output / "failures").glob("*.json"))
    fit_failures = sum(not path.name.startswith("stability.") for path in failure_paths)
    stability_failures = len(failure_paths) - fit_failures
    converged = None
    local = repeat = 0
    database_path = output / "tables" / "results.sqlite"
    if database_path.exists():
        try:
            counts = database_counts(output)
            converged = counts["converged"]
            local = counts["local"]
            repeat = counts["repeat"]
        except Exception:
            pass

    now = time.time()
    mtimes = [path.stat().st_mtime for path in records]
    rate = None
    eta_hours = None
    if mtimes:
        window_start = max(min(mtimes), now - 3600)
        duration_hours = max((now - window_start) / 3600, 1 / 3600)
        recent_count = sum(value >= window_start for value in mtimes)
        rate = recent_count / duration_hours
        if rate > 0:
            eta_hours = max(total - completed, 0) / rate

    print(f"Status at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  completed fits: {completed:,}/{total:,} ({100*completed/max(total,1):.2f}%)")
    print(f"  running fits:   {running:,}")
    print(f"  pending fits:   {max(total-completed, 0):,}")
    if converged is not None:
        print(f"  converged fits: {converged:,}/{completed:,}")
    print(f"  failed attempts: fits={fit_failures:,}, stability={stability_failures:,}")
    print(f"  stability pairs: local={local:,}, repeats={repeat:,}")
    if rate is not None:
        print(f"  recent rate:    {rate:,.2f} fits/hour")
    if eta_hours is not None:
        print(f"  rough ETA:      {eta_hours:,.2f} hours at the recent rate")
    print(f"  live CSV:       {output / 'tables' / 'fits.csv'}")
    print(f"  dashboard:      {output / 'dashboard.html'}")


def query_results(config: dict[str, Any], args) -> int:
    output = Path(config["output_dir"])
    sync_fit_records(output)
    records = load_fit_records_enriched(output)

    def matches(record: dict[str, Any]) -> bool:
        filters = {
            "fit_id": args.fit_id,
            "repeat": args.repeat,
            "N": args.N,
            "K": args.K,
            "l1_index": args.l1_index,
            "ltv_index": args.ltv_index,
        }
        for key, expected in filters.items():
            if expected is not None and record.get(key) != expected:
                return False
        if args.lambda_l1 is not None and abs(record["lambda_l1"] - args.lambda_l1) > 1e-12:
            return False
        if args.lambda_tv is not None and abs(record["lambda_tv"] - args.lambda_tv) > 1e-12:
            return False
        return True

    selected = [record for record in records if matches(record)]
    selected.sort(
        key=lambda item: (
            -float(item.get(args.sort_by) or float("-inf")),
            item["fit_id"],
        )
    )
    for record in selected[: args.limit]:
        image = output / record["fit_relative_dir"] / "component_png" / "component_01.png"
        print(
            f"{record['fit_id']}  EV={record.get('explained_variance_percent'):.4g}%  "
            f"sparsity={record.get('mean_sparsity_active')}  "
            f"relTV={record.get('mean_relative_tv_active')}  "
            f"converged={record.get('converged')}"
        )
        print(f"  {image}")
    print(f"Matched {len(selected):,} fits; displayed {min(len(selected), args.limit):,}.")
    return 0


def serve(config: dict[str, Any], host: str, port: int) -> int:
    output = Path(config["output_dir"])
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(
        *args, directory=str(output), **kwargs
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving {output} at http://{host}:{port}/dashboard.html")
    print(
        "For a remote cluster, use an SSH tunnel: "
        f"ssh -L {port}:127.0.0.1:{port} USER@CLUSTER"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Complete, resumable and live-queryable CycleSPCA full grid."
    )
    parser.add_argument("--config", default="grid.json", help="Path to grid JSON.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate code, dataset and grid.")
    validate.add_argument("--hash-dataset", action="store_true")

    subparsers.add_parser("prepare", help="Freeze provenance and nested samples.")

    run = subparsers.add_parser("run", help="Run or resume fits, then stability.")
    run.add_argument("--workers", type=int)
    run.add_argument("--force-lock", action="store_true")

    subparsers.add_parser("status", help="Print a live status snapshot.")
    watch = subparsers.add_parser("watch", help="Continuously print live status.")
    watch.add_argument("--interval", type=int, default=60)

    aggregate = subparsers.add_parser("aggregate", help="Refresh tables and dashboard now.")
    aggregate.add_argument("--components", action="store_true")
    aggregate.add_argument("--plots", action="store_true")

    stability = subparsers.add_parser("stability", help="Run or resume stability only.")
    stability.add_argument("--workers", type=int)
    stability.add_argument("--force-lock", action="store_true")

    query = subparsers.add_parser("query", help="Find completed fits and their PNGs.")
    query.add_argument("--fit-id")
    query.add_argument("--repeat", type=int)
    query.add_argument("--N", type=int)
    query.add_argument("--K", type=int)
    query.add_argument("--l1-index", type=int)
    query.add_argument("--ltv-index", type=int)
    query.add_argument("--lambda-l1", type=float)
    query.add_argument("--lambda-tv", type=float)
    query.add_argument("--sort-by", default="explained_variance")
    query.add_argument("--limit", type=int, default=20)

    web = subparsers.add_parser("serve", help="Serve the live dashboard over HTTP.")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output = Path(config["output_dir"])

    if args.command == "validate":
        repository = inspect_repository(config)
        dataset = inspect_dataset(config, calculate_hash=args.hash_dataset)["info"]
        print(f"Repository commit: {repository['commit']}")
        print(
            f"Dataset: {dataset['shape']} {dataset['dtype']}; "
            f"eligible={dataset['eligible_rows']:,}; excluded={dataset['excluded_rows']:,}"
        )
        print(f"Calendar: shape={config['calendar_shape']}, order={config['order']}")
        print(f"Grid: {total_fits(config):,} fits")
        return 0
    if args.command == "prepare":
        with RunnerLock(output):
            initialized = initialize_experiment(config)
            _refresh_once(config, export_components=False, include_plots=False)
        print(f"Experiment prepared at {output}")
        return 0
    if args.command == "run":
        workers = args.workers or int(config["runtime"]["workers"])
        return run_grid(config, workers=workers, force_lock=args.force_lock)
    if args.command == "status":
        print_status(config)
        return 0
    if args.command == "watch":
        try:
            while True:
                print_status(config)
                time.sleep(max(5, args.interval))
        except KeyboardInterrupt:
            return 0
    if args.command == "aggregate":
        _refresh_once(
            config,
            export_components=args.components,
            include_plots=args.plots,
        )
        print_status(config)
        return 0
    if args.command == "stability":
        workers = args.workers or int(config["stability"]["workers"])
        return run_stability(config, workers=workers, force_lock=args.force_lock)
    if args.command == "query":
        return query_results(config, args)
    if args.command == "serve":
        return serve(config, args.host, args.port)
    parser.error(f"Unknown command: {args.command}")
    return 2
