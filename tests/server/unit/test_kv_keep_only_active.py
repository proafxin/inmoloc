import os
import pathlib
import tempfile

import pytest
from utils import *

server = ServerPreset.tinyllama2()


class LogReader:
    def __init__(self, path) -> None:
        self.path = path
        self.pos = 0

    def drain(self):
        with pathlib.Path(self.path).open(encoding="utf-8") as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


@pytest.fixture(autouse=True)
def create_server() -> None:
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


LONG_PROMPT_2 = (
    "In a quiet village by the sea, an old fisherman mended his nets "
    "every morning and told stories to the children who gathered on "
    "the pier. One day a storm came from the north and the boats could "
    "not go out, so the whole village listened to his tales of whales "
    "and islands and a lighthouse that sang to the ships at night."
)


# with a unified cache an idle slot keeps its cells while there is room, and a new request reuses them from memory
def test_idle_slot_kept_while_room() -> None:
    global server
    server.start()
    log = LogReader(server.log_path)

    # verify feature is enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": LONG_PROMPT,
            "id_slot": 0,
            "cache_prompt": True,
        },
    )
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # the short prompt of slot 1 fits next to the idle slot 0: slot 0 is not cleared
    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "The quick brown fox",
            "id_slot": 1,
            "cache_prompt": True,
        },
    )
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # the same prompt again is served from the cells slot 0 still holds
    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": LONG_PROMPT,
            "cache_prompt": True,
        },
    )
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


def test_disabled_with_flag() -> None:
    global server
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    # Feature should not be enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" not in log.drain()

    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": LONG_PROMPT,
            "id_slot": 0,
            "cache_prompt": True,
        },
    )
    assert res.status_code == 200

    # Request on different slot — should NOT trigger clearing
    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "The quick brown fox",
            "id_slot": 1,
            "cache_prompt": True,
        },
    )
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()
