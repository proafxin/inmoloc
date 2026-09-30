import os
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()

class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


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
def test_idle_slot_kept_while_room():
    global server
    server.start()
    log = LogReader(server.log_path)

    # verify feature is enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # the short prompt of slot 1 fits next to the idle slot 0: slot 0 is not cleared
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # the same prompt again is served from the cells slot 0 still holds
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


# an idle slot gives its cells to a request that needs them, and its prompt is saved to cache-ram to restore later
def test_idle_slot_saved_when_cells_needed():
    global server
    server.n_ctx = 256  # one long prompt fits, two do not
    server.start()
    log = LogReader(server.log_path)

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT_2,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.drain()

    # the first prompt again is restored from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n


def test_disabled_with_flag():
    global server
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    # Feature should not be enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" not in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # Request on different slot — should NOT trigger clearing
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()
