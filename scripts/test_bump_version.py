"""Offline release regression tests in temporary repositories, never the checkout."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="alphawinnow-release-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        shutil.copyfile(Path(__file__).with_name("bump-version.sh"), self.root / "scripts/bump-version.sh")
        self.git("init", "-q")
        self.git("config", "user.name", "Release test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.manifest("0.1.2")
        self.commit("fix: baseline")
        self.git("tag", "v0.1.2")
        # A stub keeps tests offline and checks the two Cargo invocation paths.
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        cargo = bin_dir / "cargo"
        cargo.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$RELEASE_TEST_CALLS"\nexit "${RELEASE_TEST_CARGO_EXIT:-0}"\n')
        cargo.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                        RELEASE_TEST_CALLS=str(self.root / "cargo-calls"))

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True, text=True)

    def manifest(self, version):
        (self.root / "Cargo.toml").write_text(f'[workspace.package]\nversion = "{version}"\n')

    def commit(self, message):
        self.git("add", "Cargo.toml")
        self.git("commit", "--allow-empty", "-qm", message)

    def run_bump(self, level="auto", check=True):
        return subprocess.run(["bash", "scripts/bump-version.sh", level], cwd=self.root,
                              env=self.env, capture_output=True, text=True, check=check)

    def test_prepared_minor_not_bumped_twice(self):
        self.manifest("0.2.0")
        self.commit("feat!: measured search")
        self.assertEqual(self.run_bump().stdout.strip(), "0.2.0")
        self.assertEqual(self.run_bump().stdout.strip(), "0.2.0")
        calls = (self.root / "cargo-calls").read_text()
        self.assertIn("--locked", calls)
        self.assertNotIn("update", calls)

    def test_prepared_patch_and_major(self):
        for version in ("0.1.3", "1.0.0"):
            with self.subTest(version=version):
                self.manifest(version)
                self.assertEqual(self.run_bump().stdout.strip(), version)

    def test_ordinary_auto_patch(self):
        self.commit("fix: parser")
        self.assertEqual(self.run_bump().stdout.strip(), "0.1.3")
        self.assertIn("update --workspace", (self.root / "cargo-calls").read_text())

    def test_ordinary_auto_breaking(self):
        self.commit("feat!: schema update")
        self.assertEqual(self.run_bump().stdout.strip(), "0.2.0")

    def test_explicit_level_still_bumps(self):
        self.manifest("0.2.0")
        self.assertEqual(self.run_bump("patch").stdout.strip(), "0.2.1")

    def test_regression_is_rejected(self):
        self.manifest("0.1.1")
        self.assertNotEqual(self.run_bump(check=False).returncode, 0)

    def test_prepared_lock_check_failure_is_fatal(self):
        self.manifest("0.2.0")
        self.env["RELEASE_TEST_CARGO_EXIT"] = "9"
        result = self.run_bump(check=False)
        self.assertEqual(result.returncode, 9)
        self.assertEqual(result.stdout, "")

    def test_invalid_level_is_rejected(self):
        self.assertNotEqual(self.run_bump("unknown", check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
