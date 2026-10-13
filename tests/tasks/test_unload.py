import weakref

import pytest

from flasktts.tasks.tasks import _unloaded_after


class Model:
    def synth(self, fail):
        if fail:
            raise RuntimeError("synthesis failed")
        return "out.wav"


class FakeHighlander:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Model()
        return cls._instance

    @classmethod
    def release(cls):
        cls._instance = None


def test_model_is_collected_after_the_task():
    with _unloaded_after(FakeHighlander):
        model = weakref.ref(FakeHighlander.get_instance())
        assert FakeHighlander.get_instance().synth(fail=False) == "out.wav"

    assert FakeHighlander._instance is None
    assert model() is None


def test_model_is_collected_while_the_task_error_propagates():
    with pytest.raises(RuntimeError) as raised:
        with _unloaded_after(FakeHighlander):
            model = weakref.ref(FakeHighlander.get_instance())
            FakeHighlander.get_instance().synth(fail=True)

    assert FakeHighlander._instance is None
    assert model() is None
    assert raised.value.__traceback__ is not None
