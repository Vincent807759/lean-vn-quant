#!/usr/bin/env python3
"""
================================================================================
LEAN VN-QUANT: ZERO-LEAK SECURITY SCANNER & PRE-COMMIT AUDITOR
================================================================================
Independent, zero-dependency security audit tool to ensure:
1. No .env, private keys, or credential files are ever committed to Git.
2. No Telegram Bot Tokens, proprietary secret keys, or hardcoded personal paths.
3. Exits with code 1 upon any detection; exits with code 0 if 100% clean.
================================================================================
"""

import os
import sys
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Structural & Heuristic Regex Patterns (Zero hardcoded user secrets in source code)
STATIC_REGEX_RULES = [
    ("TELEGRAM_BOT_TOKEN", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b"), "Exposed Telegram Bot Token"),
    ("WORLDQUANT_KEYWORD", re.compile(r"\b(?:worldquant|wq_brain)\b", re.IGNORECASE), "Forbidden WorldQuant / WQ_BRAIN keyword reference"),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA|EC|DSA|OPENSSH|PGP) PRIVATE KEY-----"), "Private Cryptographic Key"),
    ("HARDCODED_ADMIN_PATH", re.compile(r"C:[\\/]Users[\\/]ADMIN", re.IGNORECASE), "Hardcoded local Windows user path"),
    ("GENERIC_SECRET", re.compile(r"""(?i)(?:api_key|secret_key|bot_token|access_token)\s*=\s*['\"][A-Za-z0-9_\-]{20,}['\"]"""), "Hardcoded generic secret key")
]


def build_dynamic_env_rules(root_dir: str) -> List[Tuple[str, re.Pattern, str]]:
    """
    Dynamically loads secret values from the local untracked .env file (if present)
    at runtime so the scanner never stores any secret literals in its own source code.
    """
    dynamic_rules = []
    env_path = os.path.join(root_dir, ".env")
    if os.path.isfile(env_path):
        try:
            with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if v and len(v) >= 6 and not is_placeholder(v):
                            dynamic_rules.append((
                                f"ENV_LEAK_{k}",
                                re.compile(re.escape(v)),
                                f"Exposed value of {k} from local .env"
                            ))
        except Exception:
            pass
    return dynamic_rules

FORBIDDEN_FILES = {
    ".env.local", ".env.production", "id_rsa", "id_ed25519"
}

IGNORE_DIRS = {
    ".venv", "venv", "env", ".git", "__pycache__", "node_modules", "dist", "build", ".idea", ".vscode"
}

PLACEHOLDER_SUBSTRINGS = [
    "<your_", "your_", "example", "placeholder", "dummy", "test_token",
    "env.example", "<workspace_path>", "{chat_id}"
]


def is_placeholder(text: str) -> bool:
    """Checks if a detected match is just a safe documentation placeholder."""
    lowered = text.lower()
    return any(p in lowered for p in PLACEHOLDER_SUBSTRINGS)


def is_tracked_by_git(root_dir: str, file_path: str) -> bool:
    """Checks if a file is currently tracked by git."""
    try:
        rel_p = os.path.relpath(file_path, root_dir)
        proc = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel_p],
            cwd=root_dir,
            capture_output=True,
            text=True
        )
        return proc.returncode == 0
    except Exception:
        return False


def get_target_files(root_dir: str) -> Tuple[List[str], bool, int, int, bool]:
    """Retrieves all tracked git files or falls back to walking the directory."""
    files_to_scan = []
    is_git_mode = False
    tracked_count = 0
    untracked_count = 0
    git_config_checked = False
    
    # 1. Try git ls-files first
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=5
        )
        if proc.returncode == 0 and proc.stdout.strip():
            tracked = proc.stdout.strip().splitlines()
            tracked_count = len(tracked)
            for rel_path in tracked:
                full_p = os.path.join(root_dir, rel_path.replace("/", os.sep))
                if os.path.isfile(full_p):
                    files_to_scan.append(full_p)

            # B15: Also include untracked and staged files via git status --porcelain
            status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root_dir,
                capture_output=True,
                text=True,
                timeout=5
            )
            if status_proc.returncode == 0 and status_proc.stdout.strip():
                lines = status_proc.stdout.strip().splitlines()
                untracked_count = len(lines)
                for line in lines:
                    if len(line) >= 4:
                        rel_p = line[3:].strip().strip('"')
                        if " -> " in rel_p:
                            rel_p = rel_p.split(" -> ")[1].strip().strip('"')
                        full_p = os.path.join(root_dir, rel_p.replace("/", os.sep))
                        if os.path.isfile(full_p) and full_p not in files_to_scan:
                            if os.path.basename(full_p) != ".env":
                                files_to_scan.append(full_p)

            # B15: Directly scan .git/config for token/secret leakage in remote URLs
            git_config = os.path.join(root_dir, ".git", "config")
            if os.path.isfile(git_config):
                git_config_checked = True
                if git_config not in files_to_scan:
                    files_to_scan.append(git_config)

            if files_to_scan:
                is_git_mode = True
                return files_to_scan, is_git_mode, tracked_count, untracked_count, git_config_checked
    except Exception:
        pass

    # 2. Fallback: Walk directory excluding ignored folders and local .env
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            # In fallback mode, do not scan local uncommitted .env file
            if f == ".env" or f.endswith(".env"):
                continue
            full_p = os.path.join(root, f)
            files_to_scan.append(full_p)

    return files_to_scan, is_git_mode, len(files_to_scan), 0, False


def run_security_scan(root_dir: str = ".") -> int:
    """Scans all files for forbidden files and secret patterns."""
    root_path = os.path.abspath(root_dir)
    print("=" * 80)
    print(f"🔒 RUNNING ZERO-LEAK SECURITY SCAN: {root_path}")
    print("=" * 80)

    # Pre-check: verify .env is NEVER tracked by git
    if is_tracked_by_git(root_path, ".env"):
        print("\n" + "!" * 80)
        print("❌ [CRITICAL SECURITY VIOLATION] '.env' is currently tracked by Git!")
        print("   Run: git rm --cached .env")
        print("!" * 80)
        return 1

    files, is_git, n_tracked, n_untracked, has_git_config = get_target_files(root_path)
    active_rules = STATIC_REGEX_RULES + build_dynamic_env_rules(root_path)

    # L2: Audit .git/config first before printing status
    git_config_violations = 0
    if has_git_config:
        git_config_path = os.path.join(root_path, ".git", "config")
        if os.path.isfile(git_config_path):
            try:
                with open(git_config_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        for rule_name, pattern, desc in active_rules:
                            if pattern.search(line):
                                git_config_violations += 1
            except Exception:
                pass

    if is_git:
        print(f"-> Scan Mode: Git Tracked Files (ls-files) ({n_tracked} files)")
        print(f"-> Checked Untracked/Staged Files (git status --porcelain): {n_untracked} files (Protected by .gitignore)")
        if has_git_config:
            print(f"-> Audited .git/config: {git_config_violations} credentials / PAT tokens found.")
    else:
        print(f"-> Scan Mode: Filesystem Scan (Excluding .env & .venv) ({len(files)} files)")

    violations = []
    scanned_count = 0

    for file_path in files:
        rel_path = os.path.relpath(file_path, root_path)
        base_name = os.path.basename(file_path)

        # 1. Check for forbidden file names and private directories in git
        norm_rel = rel_path.replace("\\", "/").lower()
        if any(d in norm_rel for d in ["private_alphas/", "research/", "local_docs/"]):
            violations.append({
                "file": rel_path,
                "line": 0,
                "rule": "FORBIDDEN_PRIVATE_DIR",
                "desc": f"Private research/alpha directory '{rel_path}' must NOT be tracked by git!",
                "snippet": rel_path
            })
            continue

        if is_git and norm_rel.endswith((".parquet", ".csv")):
            violations.append({
                "file": rel_path,
                "line": 0,
                "rule": "FORBIDDEN_DATA_FILE",
                "desc": f"Proprietary dataset file '{rel_path}' must NOT be tracked by git!",
                "snippet": rel_path
            })
            continue

        if is_git and (base_name in FORBIDDEN_FILES or (base_name.startswith(".env") and base_name != ".env.example")):
            violations.append({
                "file": rel_path,
                "line": 0,
                "rule": "FORBIDDEN_FILE",
                "desc": f"Forbidden credential file '{base_name}' should NOT be in git repository!",
                "snippet": base_name
            })
            continue

        # Skip binary files like .png, .ico, .pdf
        if file_path.endswith((".png", ".jpg", ".ico", ".pdf")):
            continue

        # Skip this script itself only for static regex self-definitions
        is_self = (base_name == "security_scan.py")

        # 2. Scan file content
        scanned_count += 1
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, start=1):
                    for rule_id, pattern, desc in active_rules:
                        if is_self and not rule_id.startswith("ENV_LEAK_"):
                            continue
                        matches = pattern.findall(line)
                        for match in matches:
                            snippet = match if isinstance(match, str) else line.strip()
                            if is_placeholder(snippet):
                                continue
                            violations.append({
                                "file": rel_path,
                                "line": line_idx,
                                "rule": rule_id,
                                "desc": desc,
                                "snippet": snippet[:60]
                            })
        except Exception as e:
            print(f"[SCAN ERROR] Could not read {rel_path}: {e}")

    print(f"-> Audited {scanned_count} code and documentation files.")

    # Report results
    if violations:
        print("\n" + "!" * 80)
        print(f"❌ [SECURITY VIOLATIONS DETECTED] Found {len(violations)} security leak(s)!")
        print("!" * 80)
        for v in violations:
            print(f"  • [{v['rule']}] {v['file']}:{v['line']}")
            print(f"    Reason : {v['desc']}")
            print(f"    Snippet: {v['snippet']}")
        print("\n[ACTION REQUIRED] Please remove or sanitize all secrets before committing.")
        return 1
    else:
        print("\n" + "=" * 80)
        print("✅ [PASS] ZERO-LEAK SECURITY AUDIT PASSED 100%! NO SECRETS FOUND.")
        print("=" * 80)
        return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    exit_code = run_security_scan(target)
    sys.exit(exit_code)
