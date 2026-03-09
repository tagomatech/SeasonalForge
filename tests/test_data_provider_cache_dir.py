import os
import unittest
from pathlib import Path
from unittest.mock import patch

from data_provider import default_cache_dir


class CacheDirTests(unittest.TestCase):
    def test_default_cache_dir_uses_env_override(self) -> None:
        with patch.dict(os.environ, {"FUTURES_SEASONALS_CACHE_DIR": "~/seasonals-cache"}, clear=False):
            self.assertEqual(default_cache_dir(), Path("~/seasonals-cache").expanduser())

    def test_default_cache_dir_uses_local_appdata_when_available(self) -> None:
        env = {
            "FUTURES_SEASONALS_CACHE_DIR": "",
            "LOCALAPPDATA": r"C:\\CacheRoot",
            "XDG_CACHE_HOME": "",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(default_cache_dir(), Path(r"C:\\CacheRoot") / "FuturesSeasonals")

    def test_default_cache_dir_falls_back_to_home_dir(self) -> None:
        env = {
            "FUTURES_SEASONALS_CACHE_DIR": "",
            "LOCALAPPDATA": "",
            "XDG_CACHE_HOME": "",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(default_cache_dir(), Path.home() / ".futures_seasonals")


if __name__ == "__main__":
    unittest.main()