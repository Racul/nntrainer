#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run kvcache_eval.py perf trials on an Android device over adb.

Each condition is an Android build staged from its own branch. Conditions,
settings, activation checks, validation, resume and reports are those of
kvcache_eval.py; this file only adds an Android target for its trial loop.

  stage  copy the current build_android.sh outputs to build-android/NAME
  check  print the device state; no inference
  perf   latency and memory trials (default 1 + 1 per input and condition);
         each trial waits until the device is as cool as at the first trial

kvcache_eval.py report DIR rewrites the summary of an Android run as well.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
import time

import kvcache_eval as kv

# Outputs of Applications/CausalLM/build_android.sh (main 7d14d001); the only
# place these paths are written.
BUILD_OUTPUTS = {
    "nntr_causallm": "builddir_app/cpu/nntr_causallm",
    "libcausallm.so": "builddir_app/cpu/libcausallm.so",
    "libquick_dot_ai_api.so": "builddir_app/cpu/libquick_dot_ai_api.so",
    "libnntrainer.so": "builddir/android_build_result/lib/arm64-v8a/libnntrainer.so",
    "libccapi-nntrainer.so": "builddir/android_build_result/lib/arm64-v8a/libccapi-nntrainer.so",
}
LIBCXX_IN_NDK = "toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
STAGED_FILES = sorted(set(BUILD_OUTPUTS) | {"libc++_shared.so"})
APP_KEYS = ("platform", "enable-qnn", "buildtype")
STAGE_ROOT = "build-android"
DEVICE_DIR = "/data/local/tmp/kvcache-eval"
REMOTE_DIR = re.compile(r"(?:/[A-Za-z0-9_.-]+)+")
# Room left free on the device beyond the files to push (tokenizer cache, logs).
FREE_MARGIN_KIB = 64 * 1024
TRIAL_COLUMNS = ["temp_start_c", "temp_end_c", "cool_wait_s", "battery_temp_c", "battery_level", "usb_powered"]


class AndroidDevice:
    """Thin adb wrapper for one device; no evaluation policy."""

    def __init__(self, adb, serial=None):
        self.adb, self.serial = adb, serial

    def argv(self, *args):
        return [self.adb] + (["-s", self.serial] if self.serial else []) + list(args)

    def _adb(self, *args, timeout=300):
        """Every adb call except a trial run goes through here."""
        return subprocess.run(self.argv(*args), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)

    def shell(self, command):
        return self._adb("shell", command)

    def read(self, command):
        result = self.shell(command)
        if result.returncode:
            raise RuntimeError(f"adb shell {command!r} failed ({result.returncode}): "
                               + (result.stderr or result.stdout).strip())
        return result.stdout

    def connect(self):
        """Select the only authorized device, or check that --serial is one."""
        result = self._adb("devices")
        if result.returncode:
            raise RuntimeError("adb devices failed: " + result.stderr.strip())
        serials = [line.split()[0] for line in result.stdout.splitlines()[1:] if line.split()[1:2] == ["device"]]
        if self.serial is not None and self.serial not in serials:
            raise RuntimeError(f"Device {self.serial} is not connected and authorized (adb devices)")
        if self.serial is None and len(serials) != 1:
            raise RuntimeError("Several devices are connected; choose one with --serial: " + ", ".join(serials)
                               if serials else "No authorized Android device; connect one with USB debugging "
                               "allowed (adb devices lists none)")
        self.serial = self.serial or serials[0]

    def host_path(self, local):
        """Windows adb.exe called from WSL reads Windows paths."""
        if self.adb.lower().endswith(".exe"):
            return subprocess.check_output(["wslpath", "-w", str(local)], text=True).strip()
        return str(local)

    def push(self, local, remote):
        result = self._adb("push", self.host_path(local), remote, timeout=None)
        if result.returncode:
            raise RuntimeError(f"adb push {local} {remote} failed: " + (result.stderr or result.stdout).strip())

    def sha256(self, paths):
        """sha256 of the device files that exist among paths."""
        if not paths:
            return {}
        listed = self.shell("sha256sum " + shlex.join(paths) + " 2>/dev/null").stdout
        return {path: digest for digest, path in (line.split() for line in listed.splitlines() if line.strip())}

    def info(self):
        props = dict(re.findall(r"^\[([^\]]+)\]: \[([^\]]*)\]$", self.read("getprop"), re.M))
        return dict(serial=self.serial, manufacturer=props.get("ro.product.manufacturer"),
                    model=props.get("ro.product.model"),
                    soc=props.get("ro.soc.model") or props.get("ro.board.platform") or props.get("ro.hardware"),
                    abi=props.get("ro.product.cpu.abi"), release=props.get("ro.build.version.release"),
                    sdk=props.get("ro.build.version.sdk"), fingerprint=props.get("ro.build.fingerprint"))

    def thermal_type(self, zone):
        return self.read(f"cat /sys/class/thermal/thermal_zone{zone}/type").strip()

    def temperature(self, zone):
        """thermal_zone<N> in °C; sysfs reports millidegrees."""
        text = self.read(f"cat /sys/class/thermal/thermal_zone{zone}/temp").strip()
        try:
            value = int(text) / 1000
        except ValueError:
            raise RuntimeError(f"thermal_zone{zone} temperature is not a number: {text!r}") from None
        if not 0 < value < 150:
            raise RuntimeError(f"thermal_zone{zone} reads {value} °C; choose another --thermal-zone")
        return value

    def battery(self):
        text = self.read("dumpsys battery")

        def field(name):
            match = re.search(r"^\s*" + name + r": (\S+)\s*$", text, re.M)
            return match[1] if match else None

        temperature, level = field("temperature"), field("level")
        return dict(battery_temp_c=int(temperature) / 10 if temperature and temperature.lstrip("-").isdigit()
                    else None, battery_level=int(level) if level and level.isdigit() else None,
                    usb_powered=field("USB powered") == "true")

    def mem_total_kib(self):
        match = re.search(r"^MemTotal:\s*(\d+) kB", self.read("cat /proc/meminfo"), re.M)
        return int(match[1]) if match else None

    def free_kib(self, path):
        """Free space of the file system that holds path (or its nearest existing parent)."""
        listed = self.read(f'd={shlex.quote(path)}; while [ ! -d "$d" ]; do d=$(dirname "$d"); done; df -k "$d"')
        return int(listed.strip().splitlines()[-1].split()[3])

    def pids(self, name):
        return self.shell(f"pidof {shlex.quote(name)}").stdout.split()

    def kill(self, pattern):
        self.shell(f"pkill -f {shlex.quote(pattern)}")

    def run(self, command, stdout_path, stderr_path):
        """A trial: the remote shell's output streams to the host log files."""
        return kv.run_model(self.argv("shell", command), dict(os.environ), stdout_path, stderr_path)


def git_revision(revision, repo=kv.REPO):
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", revision + "^{commit}"],
                            capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"Unknown git revision: {revision}")
    return result.stdout.strip()


def content_id(files):
    """Short id of a set of named files, from their names and hashes."""
    return hashlib.sha256(json.dumps(sorted(files.items())).encode()).hexdigest()[:12]


def app_build(build_dir):
    """The CausalLM app_build options that matter for comparison, and its compiler."""
    info = build_dir / "meson-info"
    if not (info / "intro-buildoptions.json").is_file():
        raise ValueError(f"No Meson build metadata in {build_dir}")
    options = {item["name"]: item["value"]
               for item in json.loads((info / "intro-buildoptions.json").read_text(encoding="utf-8"))}
    compilers = json.loads((info / "intro-compilers.json").read_text(encoding="utf-8"))
    return dict(options={key: options.get(key) for key in APP_KEYS},
                compiler=compilers.get("host", {}).get("cpp", {}).get("full_version"))


def stage_build(repo, name, revision, ndk, replace=False):
    """Copy the build_android.sh outputs of the current checkout to
    build-android/NAME and record how they were built in build-info.json."""
    if not kv.LABEL_PATTERN.fullmatch(name):
        raise ValueError(f"Stage name must match {kv.LABEL_PATTERN.pattern}: {name}")
    sources = {file: repo / path for file, path in BUILD_OUTPUTS.items()}
    sources["libc++_shared.so"] = ndk / LIBCXX_IN_NDK
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise ValueError("Build with Applications/CausalLM/build_android.sh first; missing: " + ", ".join(missing))
    properties = ndk / "source.properties"
    ndk_revision = (re.search(r"^Pkg\.Revision\s*=\s*(\S+)", properties.read_text(), re.M)
                    if properties.is_file() else None)
    root = repo / STAGE_ROOT
    target = root / name
    if target.exists() and not replace:
        raise ValueError(f"{target} exists; pass --replace to stage it again")
    root.mkdir(exist_ok=True)
    if not (root / ".gitignore").exists():
        (root / ".gitignore").write_text("*\n")
    staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=root))
    staging.chmod(0o755)  # mkdtemp creates it private
    for file, source in sources.items():
        shutil.copy2(source, staging / file)
    kv.write_json(staging / "build-info.json", dict(
        revision=revision, staged_at=kv.now(), ndk=ndk_revision[1] if ndk_revision else None,
        # The engine's compiler is Meson's host compiler; ndk-build compiles
        # the engine itself with the recorded NDK.
        engine=kv.load_build(sources["libnntrainer.so"]),
        app=app_build(repo / Path(BUILD_OUTPUTS["nntr_causallm"]).parent),
        built_at={file: datetime.fromtimestamp(source.stat().st_mtime).astimezone().isoformat()
                  for file, source in sources.items()},
        files={file: kv.sha(staging / file) for file in sources}))
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    return target


def load_stage(binary):
    """build-info.json of a staged build, after checking its files."""
    binary = Path(binary)
    path = binary.parent / "build-info.json"
    if binary.name != "nntr_causallm" or not path.is_file():
        raise ValueError(f"Not a staged Android build: {binary} (android_eval.py stage NAME)")
    info = json.loads(path.read_text(encoding="utf-8"))
    if sorted(info["files"]) != STAGED_FILES:
        raise ValueError(f"{path} lists {sorted(info['files'])}, expected {STAGED_FILES}; stage it again")
    for file, digest in info["files"].items():
        if kv.sha(binary.parent / file) != digest:
            raise ValueError(f"{binary.parent / file} changed after staging; stage it again")
    return info


def fill_revisions(args):
    """Declare each condition's revision from its stage; a given --revision
    must name the same commit."""
    binaries = kv.parse_pairs(args.condition, "--condition")
    declared = kv.parse_pairs(args.revision, "--revision")
    if set(declared) - set(binaries):
        raise ValueError("--revision names labels without a --condition: "
                         + ", ".join(sorted(set(declared) - set(binaries))))
    revisions = []
    for label, binary in binaries.items():
        staged = load_stage(Path(binary).resolve())["revision"]
        if label in declared and git_revision(declared[label]) != staged:
            raise ValueError(f"--revision {label}={declared[label]} is not the staged build's revision {staged}")
        revisions.append(f"{label}={staged}")
    args.revision = revisions


def wait_for_temperature(read, target, max_wait, poll, sleep=time.sleep, clock=time.monotonic, report=None):
    """Poll read() until it is at or below target; return (temperature, seconds waited)."""
    start = clock()
    temperature = read()
    while temperature > target:
        waited = clock() - start
        if waited >= max_wait:
            raise RuntimeError(f"The device stayed above {target:.1f} °C ({temperature:.1f} °C after {waited:.0f} s); "
                               "let it cool and resume the run")
        if report:
            report(temperature, waited)
        sleep(poll)
        temperature = read()
    return temperature, clock() - start


def remote_command(bin_dir, trial_dir, memory, script):
    """Shell command of one trial. The executable runs by absolute path, so
    pkill -f finds exactly it; exec keeps the echoed PID for the model (or for
    mem_usage.sh, which reports the model PID itself)."""
    env = dict(LD_LIBRARY_PATH=bin_dir, NNTR_ENGINE="cpu", **kv.THREADS)
    run = (["sh", script] if memory else []) + [f"{bin_dir}/nntr_causallm", trial_dir]
    return (f"cd {shlex.quote(trial_dir)} && echo device_pid=$$ && exec env "
            + " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items()) + " " + shlex.join(run))


def device_layout(protocol, device_dir):
    """Where a run's files live on the device.

    Returns the bundle directory per condition, the device path of each host
    model file, and device path -> (host path, sha256) of every file to deploy.
    """
    bins = {label: f"{device_dir}/bin/{bundle['id']}" for label, bundle in protocol["bundles"].items()}
    nntr = protocol["model_configs"]["nntr_config.json"]
    model_dir = f"{device_dir}/models/{protocol['model_files']['id']}"
    assets = {nntr[key]: f"{model_dir}/{Path(nntr[key]).name}" for key in protocol["asset_sha256"]}
    files = {f"{device_dir}/mem_usage.sh": (str(kv.MEMORY_SCRIPT), kv.MEMORY_SCRIPT_SHA)}
    for label, bundle in protocol["bundles"].items():
        host = Path(protocol["binaries"][label]).parent
        files.update({f"{bins[label]}/{name}": (str(host / name), digest) for name, digest in bundle["files"].items()})
    files.update({assets[nntr[key]]: (nntr[key], digest) for key, digest in protocol["asset_sha256"].items()})
    return bins, assets, files


class AndroidTarget:
    """Runs perf trials on one Android device (interface: kvcache_eval.LocalTarget).

    Builds come from android_eval.py stage. Each condition's executable and
    libraries form a bundle under <device-dir>/bin/<id>, the model files sit
    under <device-dir>/models/<id>, and only files missing or different on the
    device are pushed.
    """
    kind = "android"

    def __init__(self, device, device_dir=DEVICE_DIR, zone=0, max_cool_wait=1800, cool_poll=10):
        self.device, self.dir, self.zone = device, device_dir, zone
        self.max_cool_wait, self.cool_poll = max_cool_wait, cool_poll
        self.stages, self.missing = {}, []

    def check(self, kind):
        if kind != "perf":
            raise ValueError("Android runs support perf only")
        if kv.sha(kv.MEMORY_SCRIPT) != kv.MEMORY_SCRIPT_SHA:
            raise ValueError("The original Applications/utils/mem_usage.sh changed")
        self.device.connect()
        self.info = self.device.info()
        if self.info["abi"] != "arm64-v8a":
            raise RuntimeError(f"Expected an arm64-v8a device, got {self.info['abi']}")
        running = self.device.pids("nntr_causallm")
        if running:
            raise RuntimeError(f"nntr_causallm is already running on the device (pid {', '.join(running)})")
        self.device.temperature(self.zone)

    def builds(self, binaries):
        builds = {}
        for label, binary in binaries.items():
            stage = self.stages[label] = load_stage(binary)
            options = stage["engine"]["options"]
            if (options["platform"] != "android" or any(options[key] for key in kv.DEVICE_KEYS)
                    or stage["app"]["options"]["enable-qnn"]):
                raise ValueError(f"Stage {label} from a CPU-only Android build (no device backends, no QNN)")
            builds[label] = dict(engine=stage["engine"], app=stage["app"], ndk=stage["ndk"])
        return builds

    def describe(self, protocol):
        nntr = protocol["model_configs"]["nntr_config.json"]
        models = {Path(nntr[key]).name: digest for key, digest in protocol["asset_sha256"].items()}
        if len(models) != len(protocol["asset_sha256"]):
            raise ValueError("Model files need distinct file names on the device")
        described = dict(
            target="android", device=self.info,
            thermal_zone=dict(index=self.zone, type=self.device.thermal_type(self.zone)),
            device_dir=self.dir, cpu_affinity="not pinned",
            bundles={label: dict(id=content_id(stage["files"]), files=stage["files"])
                     for label, stage in self.stages.items()},
            model_files=dict(id=content_id(models), files=models),
            extra_columns=TRIAL_COLUMNS,
            environment_note=(f"Android device {self.info['model']} ({self.info['soc']}), CPU not pinned. Each trial "
                              f"starts only when thermal_zone{self.zone} is at or below its temperature at the start "
                              "of the run's first trial; temperatures, cooling wait and battery are recorded per "
                              "trial."),
            android_script_sha256=kv.sha(Path(__file__)))
        _, _, files = device_layout(dict(protocol, **described), self.dir)
        present = self.device.sha256(list(files))
        self.missing = [remote for remote, (_, digest) in files.items() if present.get(remote) != digest]
        need = sum(Path(files[remote][0]).stat().st_size for remote in self.missing) // 1024
        free = self.device.free_kib(self.dir)
        print(f"Deploy: {len(self.missing)} of {len(files)} files to push ({need / 1024:.1f} MiB); "
              f"{free / 1024:.1f} MiB free for {self.dir}", flush=True)
        if need + FREE_MARGIN_KIB > free:
            raise RuntimeError(f"Not enough space on the device for {self.dir}")
        return described

    def volatile(self):
        stages = {label: {key: stage[key] for key in ("staged_at", "built_at")} for label, stage in self.stages.items()}
        return dict(self.device.battery(), device_mem_total_kib=self.device.mem_total_kib(), stages=stages,
                    host=kv.environment())

    def open(self, protocol):
        return AndroidSession(self, protocol)


class AndroidSession:
    """Deploys a run's files, then runs its trials on the device."""

    def __init__(self, target, protocol):
        self.device, self.zone, self.missing = target.device, target.zone, target.missing
        self.max_cool_wait, self.cool_poll = target.max_cool_wait, target.cool_poll
        self.bins, self.assets, self.files = device_layout(protocol, target.dir)
        self.trial, self.script = f"{target.dir}/trial", f"{target.dir}/mem_usage.sh"
        self.active = None

    def __enter__(self):
        self.deploy()
        self.tmp = tempfile.TemporaryDirectory(prefix="nntr-kvcache-android-")
        return self

    def deploy(self):
        folders = {str(PurePosixPath(remote).parent) for remote in self.missing} | {self.trial}
        self.device.read("mkdir -p " + shlex.join(sorted(folders)))
        for remote in self.missing:
            host = Path(self.files[remote][0])
            print(f"PUSH {remote} ({host.stat().st_size / 2 ** 20:.1f} MiB)", flush=True)
            self.device.push(host, remote)
        pushed = self.device.sha256(self.missing)
        wrong = [remote for remote in self.missing if pushed.get(remote) != self.files[remote][1]]
        if wrong:
            raise RuntimeError("Pushed files differ on the device: " + ", ".join(wrong))
        self.device.read("chmod 755 " + shlex.join(sorted(f"{path}/nntr_causallm" for path in set(self.bins.values()))))

    def __exit__(self, *exc):
        if self.active:
            try:
                self.device.kill(self.active)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                print(f"Warning: could not stop {self.active} on the device: {error}", flush=True)
        self.tmp.cleanup()

    def put(self, name, value):
        if name == "nntr_config.json":
            value = dict(value, **{key: self.assets[value[key]] for key in kv.ASSET_KEYS if value.get(key)})
        local = Path(self.tmp.name) / name
        local.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        self.device.push(local, f"{self.trial}/{name}")

    def before_trial(self, manifest):
        """The run's first trial sets the target temperature (kept in the
        manifest across resumes); every later trial waits for it."""
        def read():
            return self.device.temperature(self.zone)

        reported = []

        def report(temperature, waited):
            # About once a minute while cooling.
            if not reported or waited - reported[-1] >= 60:
                reported.append(waited)
                print(f"COOL thermal_zone{self.zone} {temperature:.1f} °C > {target:.1f} °C, "
                      f"waited {waited:.0f} s", flush=True)

        if "thermal" not in manifest:
            temperature, waited = read(), 0.0
            manifest["thermal"] = dict(zone=self.zone, target_c=temperature, set_at=kv.now())
            manifest.setdefault("target_notes", []).append(
                f"Target temperature {temperature:.1f} °C (thermal_zone{self.zone}), set by the first trial at "
                f"{manifest['thermal']['set_at']}.")
        else:
            target = manifest["thermal"]["target_c"]
            temperature, waited = wait_for_temperature(read, target, self.max_cool_wait, self.cool_poll,
                                                       report=report)
        return dict(temp_start_c=temperature, cool_wait_s=round(waited, 1), **self.device.battery())

    def run(self, label, memory, stdout_path, stderr_path):
        self.active = f"{self.bins[label]}/nntr_causallm"
        code, _, duration, stdout = self.device.run(remote_command(self.bins[label], self.trial, memory, self.script),
                                                    stdout_path, stderr_path)
        self.active = None
        # adb.exe may write CRLF; the raw log keeps it, validation reads LF.
        stdout = stdout.replace("\r\n", "\n")
        pid = re.search(r"^device_pid=(\d+)$", stdout, re.M)
        return code, int(pid[1]) if pid else None, duration, stdout

    def after_trial(self):
        return dict(temp_end_c=self.device.temperature(self.zone))


def print_device(device, device_dir, zone):
    """Device state and what earlier runs deployed; no inference."""
    device.connect()
    info = device.info()
    print(f"device   {info['manufacturer']} {info['model']} ({info['soc']}), Android {info['release']} "
          f"(SDK {info['sdk']}), {info['abi']}, serial {info['serial']}")
    print(f"thermal  zone {zone} ({device.thermal_type(zone)}): {device.temperature(zone):.1f} °C")
    battery = device.battery()
    print(f"battery  {battery['battery_level']}%, {battery['battery_temp_c']} °C, "
          f"USB powered {battery['usb_powered']}")
    print(f"storage  {device.free_kib(device_dir) / 2 ** 20:.1f} GiB free for {device_dir}")
    for folder in ("bin", "models"):
        listed = device.shell(f"ls {shlex.quote(device_dir + '/' + folder)}")
        names = listed.stdout.split() if listed.returncode == 0 else []
        print(f"{folder:8} {', '.join(names) or 'none'}")
    running = device.pids("nntr_causallm")
    print("running  nntr_causallm " + (f"pid {', '.join(running)}" if running else "none"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    device = argparse.ArgumentParser(add_help=False)
    device.add_argument("--adb", default="adb", help="adb executable; Windows adb.exe works from WSL")
    device.add_argument("--serial", help="device serial (default: the only authorized device)")
    device.add_argument("--device-dir", default=DEVICE_DIR,
                        help=f"working directory on the device (default {DEVICE_DIR})")
    device.add_argument("--thermal-zone", type=int, default=0,
                        help="/sys/class/thermal/thermal_zone<N> read for the temperature (default 0)")

    stage = commands.add_parser("stage", help="copy the current Android build outputs to build-android/NAME")
    stage.add_argument("name", help="stage name, e.g. main or the method branch")
    stage.add_argument("--revision", required=True, help="git revision the outputs were just built from")
    stage.add_argument("--ndk", type=Path, default=os.environ.get("ANDROID_NDK"),
                       help="Android NDK root that built them (default $ANDROID_NDK), for libc++_shared.so")
    stage.add_argument("--replace", action="store_true", help="replace an existing stage of that name")

    commands.add_parser("check", parents=[device], help="print the device state; no inference")

    perf = commands.add_parser("perf", parents=[*kv.perf_parser_parents(), device],
                               help="latency and memory trials on the device")
    perf.add_argument("--max-cool-wait", type=float, default=1800,
                      help="seconds a trial may wait for the target temperature before the run stops (default 1800)")
    perf.add_argument("--cool-poll", type=float, default=10,
                      help="seconds between temperature readings while waiting (default 10)")
    perf.set_defaults(repeat=1)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command != "stage" and (not REMOTE_DIR.fullmatch(args.device_dir)
                                    or ".." in args.device_dir.split("/")):
        parser.error("--device-dir must be an absolute path of [A-Za-z0-9_.-] names without ..")
    if args.command == "perf" and (args.max_cool_wait < 0 or args.cool_poll <= 0):
        parser.error("--max-cool-wait must be nonnegative and --cool-poll positive")
    if args.command == "stage":
        if args.ndk is None:
            parser.error("stage needs --ndk or ANDROID_NDK")
        target = stage_build(kv.REPO, args.name, git_revision(args.revision), args.ndk.resolve(), args.replace)
        print(f"Staged {target} from {args.revision}")
    elif args.command == "check":
        print_device(AndroidDevice(args.adb, args.serial), args.device_dir, args.thermal_zone)
    else:
        kv.resolve_paths(args)
        kv.check_perf_args(parser, args)
        fill_revisions(args)
        kv.evaluate_perf(args, AndroidTarget(AndroidDevice(args.adb, args.serial), args.device_dir,
                                             args.thermal_zone, args.max_cool_wait, args.cool_poll))


if __name__ == "__main__":
    kv.cli(main)
