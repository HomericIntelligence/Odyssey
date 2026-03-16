import json
import subprocess
import sys
from pathlib import Path
import pytest

# -------------------------------
# Hook detection (mirrors original bash logic)
# -------------------------------
project_hook = Path(".claude/hooks/pre-bash-exec.py")
home_hook = Path.home() / ".claude/hooks/pre-bash-exec.py"

if project_hook.exists():
    HOOK_SCRIPT = project_hook
elif home_hook.exists():
    HOOK_SCRIPT = home_hook
else:
    pytest.skip("No hook script found in project or home directory", allow_module_level=True)

PROJECT_ROOT = Path.cwd()


# -------------------------------
# Helper to run the hook
# -------------------------------
def run_hook(cmd: str) -> bool:
    """
    Run the pre-bash-exec Python hook with a given command string.

    Returns True if allowed, False if blocked.
    """
    payload = {"tool": "bash", "input": {"command": cmd, "cwd": str(PROJECT_ROOT)}}
    proc = subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        input=json.dumps(payload).encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode == 0:
        return True
    else:
        # Optionally parse block reason
        try:
            json.loads(proc.stdout.decode()).get("reason", "")
        except json.JSONDecodeError:
            reason = ""
            print(reason)
        return False


# -------------------------------
# Original dangerous commands
# -------------------------------
dangerous_commands = [
    "rm -rf /",
    "rm -rf / ",
    "rm -rf ~/",
    "rm -rf $HOME",
    "rm .git",
    "rm .git/config",
    "rm /etc/passwd",
    "rm -rf /tmp/something",
    "sudo rm -rf /tmp/file",
    "rm -rf",
    "echo yes | rm file.txt",
    "cp README.md README.md && rm -rf ~/",
    "rm ./foo || echo yes",
    "rm ./foo; echo yes",
    "rm ./foo && rm ./bar",
    "dd if=/dev/zero of=test.img",
    "mkfs.ext4 /dev/sda",
    "wipefs -a /dev/sda",
    "mount /dev/sda /mnt",
    "umount /mnt",
    "chmod -R 777 ./tests",
    "chown -R user:user ./tests",
    "chgrp -R staff ./tests",
    "tar -xf archive.tar",
    "unzip file.zip",
    "find ./tests -delete",
    "find ./tests -exec rm {} \\;",
    "echo ./tests | xargs rm",
    "git clean -fdx",
    "rsync -av --delete src/ dst/",
    "echo $(ls)",
    "echo `ls`",
    "echo ${PATH}",
]

# -------------------------------
# Original safe commands
# -------------------------------
safe_commands = [
    "rm ./tests/README.md",
    "rm tests/README.md",
    "rm README.md CLAUDE.md",
    "ls -la",
    "git status",
    "echo hello world",
    "mkdir -p ./tmp && touch ./tmp/file",
    "echo foo | tee ./tmp/output",
    "cp README.md ./tmp/",
    f"rm -rf {PROJECT_ROOT}/build" if (PROJECT_ROOT / "build").exists() else None,
]
safe_commands = [c for c in safe_commands if c is not None]

# -------------------------------
# Additional complex corner-case commands
# -------------------------------
complex_dangerous_commands = [
    "rm ./foo && rm ./bar && echo done",
    "rm ./foo || rm ./bar",
    "rm ./foo; rm ./bar; echo hello",
    "echo ./foo | xargs rm",
    "cat list.txt | xargs rm -f",
    "ls | grep pattern | xargs rm",
    'rm "$HOME/testfile"',
    "rm '~/testfile'",
    "rm ${HOME}/testfile",
    "find ./tmp -name '*.log' -exec rm {} \\;",
    "find ./tmp -type f -delete",
    "git status && git clean -fdx || echo done",
    "mkfs.ext4 /dev/sdb1",
    "sudo mkfs.vfat /dev/sdc1",
    "dd if=/dev/zero of=foo.img",
    "echo hello | tee output.txt | rm ./foo",
    "cat file | grep foo | xargs rm",
]

complex_safe_commands = [
    "ls -la && echo done && pwd",
    "echo hello; mkdir -p ./tmp && touch ./tmp/file",
    "cat file.txt | grep pattern | sort",
    "echo foo | tee ./tmp/output | wc -l",
    "git status && git log --oneline",
    "echo 'hello' > output.txt",
    "mkdir -p ./build && cp README.md ./build/",
]


# -------------------------------
# Parametrized tests
# -------------------------------
@pytest.mark.parametrize("cmd", dangerous_commands)
def test_dangerous_commands_blocked(cmd):
    allowed = run_hook(cmd)
    assert not allowed, f"Dangerous command passed but should be blocked: {cmd}"


@pytest.mark.parametrize("cmd", safe_commands)
def test_safe_commands_allowed(cmd):
    allowed = run_hook(cmd)
    assert allowed, f"Safe command blocked incorrectly: {cmd}"


@pytest.mark.parametrize("cmd", complex_dangerous_commands)
def test_complex_dangerous_commands_blocked(cmd):
    allowed = run_hook(cmd)
    assert not allowed, f"Complex dangerous command passed but should be blocked: {cmd}"


@pytest.mark.parametrize("cmd", complex_safe_commands)
def test_complex_safe_commands_allowed(cmd):
    allowed = run_hook(cmd)
    assert allowed, f"Complex safe command blocked incorrectly: {cmd}"
