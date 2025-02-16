##
## ====================== Installer ======================
##
import os
import sys
import logging
import copy
import pkg_resources
import subprocess


PIXELOE_VERSION = "0.1.3"
python = sys.executable


def run(command) -> str:
    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ,
        "encoding": "utf8",
        "errors": "ignore",
    }
    run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE
    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        logger.error(
            f"Command failed with exit code {result.returncode}:\n{command}\n{result.stderr}"
        )
        raise RuntimeError(f"Command failed")

    return result.stdout or ""


def run_pip(command):
    return run(f'"{python}" -m pip {command}')


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


# Create a new logger
logger = logging.getLogger("PixelOE-installer")
logger.propagate = False

# Add handler if we don't have one.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter(
            "[%(name)s]-|%(asctime)s|-%(levelname)s: %(message)s", "%H:%M:%S"
        )
    )
    logger.addHandler(handler)

logger.setLevel(logging.INFO)
logger.debug("Logger initialized.")


def get_installed_version(package: str):
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None


def install_pixeloe():
    version = get_installed_version("pixeloe")
    if version is not None and version >= PIXELOE_VERSION:
        return
    logger.info("Attempting to install pixeloe")
    run_pip(f'install -U "pixeloe>={PIXELOE_VERSION}"')
