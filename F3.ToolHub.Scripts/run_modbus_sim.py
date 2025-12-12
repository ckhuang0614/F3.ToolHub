import argparse
import asyncio
import contextlib
import signal
from pathlib import Path

from pymodbus.server.simulator.http_server import ModbusSimulatorServer

DEFAULT_CONFIG = Path(__file__).with_name("pymodbus_web_demo.json")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the pymodbus Web Simulator with graceful shutdown support."
    )
    parser.add_argument(
        "--json-file",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to simulator JSON configuration (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--modbus-server",
        default="server",
        help="Server section name in the JSON file (default: server)",
    )
    parser.add_argument(
        "--modbus-device",
        default="demo-device",
        help="Device section name in the JSON file (default: demo-device)",
    )
    parser.add_argument(
        "--http-host",
        default="0.0.0.0",
        help="HTTP host for the Web UI (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8081,
        help="HTTP port for the Web UI (default: 8081)",
    )
    parser.add_argument(
        "--log-file",
        default="web-sim.log",
        help="File used by pymodbus to persist simulator logs (default: web-sim.log)",
    )
    parser.add_argument(
        "--custom-actions",
        default=None,
        help="Optional dotted path to a module that exposes custom_actions_dict",
    )
    return parser.parse_args()

async def run_simulator(args: argparse.Namespace) -> None:
    config_path = args.json_file.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Simulator config not found: {config_path}")

    sim = ModbusSimulatorServer(
        modbus_server=args.modbus_server,
        modbus_device=args.modbus_device,
        http_host=args.http_host,
        http_port=args.http_port,
        log_file=args.log_file,
        json_file=str(config_path),
        custom_actions_module=args.custom_actions,
    )

    await sim.run_forever(only_start=True)
    await sim.ready_event.wait()
    print(
        "Simulator running | Modbus TCP: 0.0.0.0:1502 | Web UI: http://%s:%s"
        % (args.http_host, args.http_port)
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop_event.set)

    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await sim.stop()
        print("Simulator stopped.")

async def main() -> None:
    args = parse_args()
    await run_simulator(args)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
