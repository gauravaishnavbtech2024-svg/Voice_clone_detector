import numpy as np
import pytest

from callshield.core.decider import AlertDecider
from callshield.core.session import SessionEngine, SessionState


class ScriptedDetector:
    """Test double detector returning scripted scores or constant values."""

    def __init__(self, scores: list[float] | None = None, default_score: float = 1.0) -> None:
        self.scores = list(scores) if scores is not None else []
        self.default_score = default_score
        self.call_count = 0
        self.last_input: np.ndarray | None = None

    def score(self, x: np.ndarray) -> float:
        self.call_count += 1
        self.last_input = x.copy()
        if self.scores:
            return self.scores.pop(0)
        return self.default_score


@pytest.mark.unit
def test_session_engine_timing_and_buffering() -> None:
    """1 s of audio -> no state yet; 2 s -> first state; 3 s -> second state."""
    detector = ScriptedDetector(default_score=1.0)
    decider = AlertDecider(thr=0.0, k=4, n=6)
    engine = SessionEngine(detector=detector, decider=decider, sample_rate_in=16000)

    # Generate normal-level audible audio (sine wave)
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    chunk_1s = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    # Feed 1.0s (16,000 samples) -> under 2.0s buffer threshold -> returns empty list
    states_1 = engine.feed(chunk_1s)
    assert len(states_1) == 0
    assert detector.call_count == 0

    # Feed another 1.0s (total 2.0s) -> reaches 2.0s -> emits first state at t=2.0
    states_2 = engine.feed(chunk_1s)
    assert len(states_2) == 1
    assert detector.call_count == 1
    st1 = states_2[0]
    assert isinstance(st1, SessionState)
    assert st1.t == 2.0
    assert st1.score == 1.0
    assert st1.level == "OK"
    assert st1.flagged == 0
    assert st1.reason == "scored"

    # Window passed to detector must be 4 s (64,000 samples) repeat-padded and normalized
    assert detector.last_input is not None
    assert len(detector.last_input) == 64000
    rms = np.sqrt(np.mean(detector.last_input**2))
    dbfs = 20.0 * np.log10(rms)
    assert pytest.approx(dbfs, abs=0.1) == -26.0

    # Feed 3rd second (total 3.0s) -> emits second state at t=3.0
    states_3 = engine.feed(chunk_1s)
    assert len(states_3) == 1
    st2 = states_3[0]
    assert st2.t == 3.0


@pytest.mark.unit
def test_session_engine_silence_bypass() -> None:
    """Silent windows are NOT scored: level OK, reason='silence'."""
    detector = ScriptedDetector(default_score=-1.0)  # Would trigger suspicious if scored
    decider = AlertDecider(thr=0.0, k=4, n=6)
    engine = SessionEngine(detector=detector, decider=decider, sample_rate_in=16000)

    # Feed 2 seconds of silence
    silent_audio = np.zeros(32000, dtype=np.float32)
    states = engine.feed(silent_audio)

    assert len(states) == 1
    st = states[0]
    assert st.t == 2.0
    assert st.level == "OK"
    assert st.reason == "silence"
    assert st.score == 0.0
    # Detector was never called for silence
    assert detector.call_count == 0


@pytest.mark.unit
def test_session_engine_chunking_invariance() -> None:
    """Feeding in 20 ms chunks gives the identical sequence of states as feeding all at once."""
    rng = np.random.default_rng(999)
    total_samples = 16000 * 5  # 5 seconds
    # Audible audio
    audio = rng.standard_normal(total_samples).astype(np.float32) * 0.3

    # Engine 1: Fed in large 1-second chunks (16000 samples)
    decider1 = AlertDecider(thr=0.0, k=4, n=6)
    det1 = ScriptedDetector(scores=[0.8, 0.7, -0.5, -0.6])
    engine1 = SessionEngine(detector=det1, decider=decider1, sample_rate_in=16000)

    states_bulk: list[SessionState] = []
    for i in range(0, total_samples, 16000):
        states_bulk.extend(engine1.feed(audio[i : i + 16000]))

    # Engine 2: Fed in 20 ms telephony chunks (320 samples at 16 kHz)
    decider2 = AlertDecider(thr=0.0, k=4, n=6)
    det2 = ScriptedDetector(scores=[0.8, 0.7, -0.5, -0.6])
    engine2 = SessionEngine(detector=det2, decider=decider2, sample_rate_in=16000)

    states_chunked: list[SessionState] = []
    chunk_size = 320  # 20 ms
    for i in range(0, total_samples, chunk_size):
        states_chunked.extend(engine2.feed(audio[i : i + chunk_size]))

    assert len(states_bulk) == len(states_chunked)
    for s_bulk, s_chunk in zip(states_bulk, states_chunked, strict=True):
        assert pytest.approx(s_bulk.t) == s_chunk.t
        assert s_bulk.score == s_chunk.score
        assert s_bulk.level == s_chunk.level
        assert s_bulk.flagged == s_chunk.flagged
        assert s_bulk.reason == s_chunk.reason


@pytest.mark.unit
def test_session_engine_fake_sequence_reaches_alert() -> None:
    """A fake sequence reaches ALERT at the expected second."""
    # 6 seconds of audio -> 5 evaluations: at t=2, 3, 4, 5, 6
    # Scores: -0.5, -0.6, -0.7, -0.8 -> reaches 4 suspicious at t=5.0
    detector = ScriptedDetector(scores=[-0.5, -0.6, -0.7, -0.8, -0.9])
    decider = AlertDecider(thr=0.0, k=4, n=6)
    engine = SessionEngine(detector=detector, decider=decider, sample_rate_in=16000)

    rng = np.random.default_rng(42)
    one_sec = rng.standard_normal(16000).astype(np.float32) * 0.2

    all_states: list[SessionState] = []
    # Feed second 1 and 2
    all_states.extend(engine.feed(one_sec))  # t=1 -> no state
    all_states.extend(engine.feed(one_sec))  # t=2 -> 1st score (-0.5) -> flagged=1, OK
    assert all_states[-1].level == "OK"

    all_states.extend(engine.feed(one_sec))  # t=3 -> 2nd score (-0.6) -> flagged=2, WARNING
    assert all_states[-1].level == "WARNING"

    all_states.extend(engine.feed(one_sec))  # t=4 -> 3rd score (-0.7) -> flagged=3, WARNING
    assert all_states[-1].level == "WARNING"

    all_states.extend(engine.feed(one_sec))  # t=5 -> 4th score (-0.8) -> flagged=4, ALERT!
    assert all_states[-1].level == "ALERT"
    assert all_states[-1].t == 5.0


@pytest.mark.unit
def test_session_engine_instance_isolation() -> None:
    """Two engines never share state."""
    det1 = ScriptedDetector(default_score=1.0)
    dec1 = AlertDecider(thr=0.0, k=4, n=6)
    engine1 = SessionEngine(detector=det1, decider=dec1, sample_rate_in=16000)

    det2 = ScriptedDetector(default_score=-1.0)
    dec2 = AlertDecider(thr=0.0, k=4, n=6)
    engine2 = SessionEngine(detector=det2, decider=dec2, sample_rate_in=16000)

    audio = np.ones(32000, dtype=np.float32) * 0.1
    s1 = engine1.feed(audio)
    s2 = engine2.feed(audio)

    assert s1[0].level == "OK"
    assert s1[0].flagged == 0

    assert s2[0].level == "OK"
    assert s2[0].flagged == 1


@pytest.mark.unit
def test_session_engine_8k_telephony_ingestion() -> None:
    """Telephony 8 kHz input is automatically resampled to 16 kHz."""
    det = ScriptedDetector(default_score=0.9)
    dec = AlertDecider(thr=0.0, k=4, n=6)
    engine = SessionEngine(detector=det, decider=dec, sample_rate_in=8000)

    # 8 kHz audio: 16,000 samples = 2.0 seconds at 8 kHz
    audio_8k = (np.sin(np.linspace(0, 100, 16000)) * 0.2).astype(np.float32)
    states = engine.feed(audio_8k)

    # After resampling to 16 kHz, 16,000 samples at 8 kHz -> 32,000 samples at 16 kHz (2.0s)
    assert len(states) == 1
    assert states[0].t == 2.0
    assert states[0].level == "OK"


@pytest.mark.unit
def test_session_engine_empty_feed() -> None:
    """Feeding empty array returns empty list."""
    det = ScriptedDetector()
    dec = AlertDecider(thr=0.0, k=4, n=6)
    engine = SessionEngine(detector=det, decider=dec)
    assert engine.feed(np.array([], dtype=np.float32)) == []
