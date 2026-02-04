"""Test lens functionality."""

##
# Imports

import pytest

from dataclasses import dataclass
import webdataset as wds
import atdata

import numpy as np
from numpy.typing import NDArray


##
# Tests


def test_lens():
    """Test a lens between sample types"""

    # Set up the lens scenario

    @atdata.packable
    class Source:
        name: str
        age: int
        height: float

    @atdata.packable
    class View:
        name: str
        height: float

    @atdata.lens
    def polite(s: Source) -> View:
        return View(
            name=s.name,
            height=s.height,
        )

    @polite.putter
    def polite_update(v: View, s: Source) -> Source:
        return Source(
            name=v.name,
            height=v.height,
            #
            age=s.age,
        )

    # Test with an example sample

    test_source = Source(
        name="Hello World",
        age=42,
        height=182.9,
    )
    correct_view = View(
        name=test_source.name,
        height=test_source.height,
    )

    test_view = polite(test_source)
    assert test_view == correct_view, (
        f"Incorrect lens behavior: {test_view}, and not {correct_view}"
    )

    # This lens should be well-behaved

    update_view = View(
        name="Now Taller",
        height=192.9,
    )

    x = polite(polite.put(update_view, test_source))
    assert x == update_view, f"Violation of GetPut: {x} =/= {update_view}"

    y = polite.put(polite(test_source), test_source)
    assert y == test_source, f"Violation of PutGet: {y} =/= {test_source}"

    # PutPut law: put(v2, put(v1, s)) = put(v2, s)
    another_view = View(
        name="Different Name",
        height=165.0,
    )
    z1 = polite.put(another_view, polite.put(update_view, test_source))
    z2 = polite.put(another_view, test_source)
    assert z1 == z2, f"Violation of PutPut: {z1} =/= {z2}"


def test_conversion(tmp_path):
    """Test automatic interconversion between sample types"""

    @dataclass
    class Source(atdata.PackableSample):
        name: str
        height: float
        favorite_pizza: str
        favorite_image: NDArray

    @dataclass
    class View(atdata.PackableSample):
        name: str
        favorite_pizza: str
        favorite_image: NDArray

    @atdata.lens
    def polite(s: Source) -> View:
        return View(
            name=s.name,
            favorite_pizza=s.favorite_pizza,
            favorite_image=s.favorite_image,
        )

    # Map a test sample through the view
    test_source = Source(
        name="Larry",
        height=42.0,
        favorite_pizza="pineapple",
        favorite_image=np.random.randn(224, 224),
    )
    test_view = polite(test_source)

    # Create a test dataset

    k_test = 100
    test_filename = (tmp_path / "test-source.tar").as_posix()

    with wds.writer.TarWriter(test_filename) as dest:
        for i in range(k_test):
            # Create a new copied sample
            cur_sample = Source(
                name=test_source.name,
                height=test_source.height,
                favorite_pizza=test_source.favorite_pizza,
                favorite_image=test_source.favorite_image,
            )
            dest.write(cur_sample.as_wds)

    # Try reading the test dataset

    ds = atdata.Dataset[Source](test_filename).as_type(View)

    assert ds.sample_type == View, "Auto-mapped"

    sample: View | None = None
    for sample in ds.ordered(batch_size=None):
        # Load only the first sample
        break

    assert sample is not None, "Did not load any samples from `Source` dataset"

    assert sample.name == test_view.name, (
        f"Divergence on auto-mapped dataset: `name` should be {test_view.name}, but is {sample.name}"
    )
    assert sample.favorite_pizza == test_view.favorite_pizza, (
        f"Divergence on auto-mapped dataset: `favorite_pizza` should be {test_view.favorite_pizza}, but is {sample.favorite_pizza}"
    )
    assert np.all(sample.favorite_image == test_view.favorite_image), (
        "Divergence on auto-mapped dataset: `favorite_image`"
    )


##
# Composition tests


# Shared types for composition tests

@atdata.packable
class Person:
    name: str
    age: int
    city: str


@atdata.packable
class NameAge:
    name: str
    age: int


@atdata.packable
class NameOnly:
    name: str


# Lenses (not registered via @lens to avoid polluting the global network)


def _person_to_nameage_get(s: Person) -> NameAge:
    return NameAge(name=s.name, age=s.age)


def _person_to_nameage_put(v: NameAge, s: Person) -> Person:
    return Person(name=v.name, age=v.age, city=s.city)


def _nameage_to_name_get(s: NameAge) -> NameOnly:
    return NameOnly(name=s.name)


def _nameage_to_name_put(v: NameOnly, s: NameAge) -> NameAge:
    return NameAge(name=v.name, age=s.age)


_person_to_nameage = atdata.Lens(_person_to_nameage_get, put=_person_to_nameage_put)
_nameage_to_name = atdata.Lens(_nameage_to_name_get, put=_nameage_to_name_put)


class TestPipelineComposition:
    """Tests for the | (pipeline) operator."""

    def test_pipe_get(self):
        """Pipeline composition applies getters left-to-right."""
        composed = _person_to_nameage | _nameage_to_name
        src = Person(name="Alice", age=30, city="NYC")
        result = composed.get(src)
        assert result == NameOnly(name="Alice")

    def test_pipe_put(self):
        """Pipeline composition threads put correctly (nLab formula)."""
        composed = _person_to_nameage | _nameage_to_name
        src = Person(name="Alice", age=30, city="NYC")
        updated = composed.put(NameOnly(name="Bob"), src)
        assert updated == Person(name="Bob", age=30, city="NYC")

    def test_pipe_source_view_types(self):
        """Composite lens has correct source_type and view_type."""
        composed = _person_to_nameage | _nameage_to_name
        assert composed.source_type is Person
        assert composed.view_type is NameOnly

    def test_pipe_getput_law(self):
        """GetPut: put(get(s), s) == s for composite lens."""
        composed = _person_to_nameage | _nameage_to_name
        src = Person(name="Alice", age=30, city="NYC")
        assert composed.put(composed.get(src), src) == src

    def test_pipe_putget_law(self):
        """PutGet: get(put(v, s)) == v for composite lens."""
        composed = _person_to_nameage | _nameage_to_name
        src = Person(name="Alice", age=30, city="NYC")
        v = NameOnly(name="Zara")
        assert composed.get(composed.put(v, src)) == v

    def test_pipe_putput_law(self):
        """PutPut: put(v2, put(v1, s)) == put(v2, s) for composite lens."""
        composed = _person_to_nameage | _nameage_to_name
        src = Person(name="Alice", age=30, city="NYC")
        v1 = NameOnly(name="Bob")
        v2 = NameOnly(name="Carol")
        assert composed.put(v2, composed.put(v1, src)) == composed.put(v2, src)


class TestCategoricalComposition:
    """Tests for the @ (matmul) operator."""

    def test_matmul_get(self):
        """Categorical composition applies getters right-to-left."""
        composed = _nameage_to_name @ _person_to_nameage
        src = Person(name="Alice", age=30, city="NYC")
        assert composed.get(src) == NameOnly(name="Alice")

    def test_matmul_put(self):
        """Categorical composition threads put correctly."""
        composed = _nameage_to_name @ _person_to_nameage
        src = Person(name="Alice", age=30, city="NYC")
        updated = composed.put(NameOnly(name="Bob"), src)
        assert updated == Person(name="Bob", age=30, city="NYC")

    def test_pipe_matmul_equivalence(self):
        """(f | g) produces same results as (g @ f)."""
        src = Person(name="Alice", age=30, city="NYC")
        pipe = _person_to_nameage | _nameage_to_name
        matmul = _nameage_to_name @ _person_to_nameage

        assert pipe.get(src) == matmul.get(src)

        v = NameOnly(name="Zara")
        assert pipe.put(v, src) == matmul.put(v, src)


class TestAssociativity:
    """Verify (f | g) | h == f | (g | h) on concrete data."""

    def setup_method(self):
        @atdata.packable
        class Full:
            a: int
            b: str
            c: float

        @atdata.packable
        class AB:
            a: int
            b: str

        @atdata.packable
        class AOnly:
            a: int

        self.Full = Full
        self.AB = AB
        self.AOnly = AOnly

        def f_get(s: Full) -> AB:
            return AB(a=s.a, b=s.b)

        def f_put(v: AB, s: Full) -> Full:
            return Full(a=v.a, b=v.b, c=s.c)

        def g_get(s: AB) -> AOnly:
            return AOnly(a=s.a)

        def g_put(v: AOnly, s: AB) -> AB:
            return AB(a=v.a, b=s.b)

        def h_get(s: AOnly) -> AOnly:
            return AOnly(a=s.a * 2)

        def h_put(v: AOnly, s: AOnly) -> AOnly:
            return AOnly(a=v.a // 2)

        self.f = atdata.Lens(f_get, put=f_put)
        self.g = atdata.Lens(g_get, put=g_put)
        self.h = atdata.Lens(h_get, put=h_put)

        self.src = Full(a=5, b="hello", c=3.14)

    def test_associativity_get(self):
        left = (self.f | self.g) | self.h
        right = self.f | (self.g | self.h)
        assert left.get(self.src) == right.get(self.src)

    def test_associativity_put(self):
        left = (self.f | self.g) | self.h
        right = self.f | (self.g | self.h)
        v = self.AOnly(a=42)
        assert left.put(v, self.src) == right.put(v, self.src)


class TestIdentityLens:
    """Tests for Lens.identity()."""

    def test_identity_get(self):
        id_lens = atdata.Lens.identity(Person)
        src = Person(name="Alice", age=30, city="NYC")
        assert id_lens.get(src) == src

    def test_identity_put(self):
        id_lens = atdata.Lens.identity(Person)
        src = Person(name="Alice", age=30, city="NYC")
        new = Person(name="Bob", age=25, city="LA")
        assert id_lens.put(new, src) == new

    def test_identity_types(self):
        id_lens = atdata.Lens.identity(Person)
        assert id_lens.source_type is Person
        assert id_lens.view_type is Person

    def test_compose_with_identity_left(self):
        """id | f == f."""
        id_lens = atdata.Lens.identity(Person)
        composed = id_lens | _person_to_nameage
        src = Person(name="Alice", age=30, city="NYC")
        assert composed.get(src) == _person_to_nameage.get(src)

        v = NameAge(name="Bob", age=25)
        assert composed.put(v, src) == _person_to_nameage.put(v, src)

    def test_compose_with_identity_right(self):
        """f | id == f."""
        id_lens = atdata.Lens.identity(NameAge)
        composed = _person_to_nameage | id_lens
        src = Person(name="Alice", age=30, city="NYC")
        assert composed.get(src) == _person_to_nameage.get(src)

        v = NameAge(name="Bob", age=25)
        assert composed.put(v, src) == _person_to_nameage.put(v, src)


class TestCompositionCallable:
    """Verify __call__ works on composite lenses."""

    def test_pipe_callable(self):
        composed = _person_to_nameage | _nameage_to_name
        src = Person(name="Alice", age=30, city="NYC")
        assert composed(src) == NameOnly(name="Alice")

    def test_matmul_callable(self):
        composed = _nameage_to_name @ _person_to_nameage
        src = Person(name="Alice", age=30, city="NYC")
        assert composed(src) == NameOnly(name="Alice")


class TestCompositionTypeErrors:
    """Verify composition with non-Lens operands raises TypeError."""

    def test_pipe_non_lens_returns_not_implemented(self):
        with pytest.raises(TypeError):
            _person_to_nameage | 42

    def test_matmul_non_lens_returns_not_implemented(self):
        with pytest.raises(TypeError):
            _nameage_to_name @ "not a lens"

    def test_pipe_non_lens_string(self):
        with pytest.raises(TypeError):
            _person_to_nameage | "hello"

    def test_matmul_non_lens_none(self):
        with pytest.raises(TypeError):
            _nameage_to_name @ None


##
# Edge case tests for coverage


def test_lens_get_method():
    """Test calling lens.get() explicitly instead of lens()."""

    @atdata.packable
    class GetSource:
        value: int

    @atdata.packable
    class GetView:
        doubled: int

    @atdata.lens
    def doubler(s: GetSource) -> GetView:
        return GetView(doubled=s.value * 2)

    source = GetSource(value=5)

    # Test both calling conventions
    result_call = doubler(source)
    result_get = doubler.get(source)

    assert result_call == result_get
    assert result_get.doubled == 10


def test_lens_trivial_putter():
    """Test lens without explicit putter uses trivial putter."""

    @atdata.packable
    class TrivialSource:
        a: int
        b: str

    @atdata.packable
    class TrivialView:
        a: int

    # Create lens without putter
    @atdata.lens
    def extract_a(s: TrivialSource) -> TrivialView:
        return TrivialView(a=s.a)

    source = TrivialSource(a=10, b="hello")
    view = TrivialView(a=99)

    # Trivial putter should return source unchanged
    result = extract_a.put(view, source)
    assert result == source, "Trivial putter should return source unchanged"


def test_lens_network_missing_lens():
    """Test LensNetwork raises ValueError for unregistered lens."""
    from atdata.lens import LensNetwork

    @atdata.packable
    class UnregisteredSource:
        x: int

    @atdata.packable
    class UnregisteredView:
        y: int

    network = LensNetwork()

    with pytest.raises(ValueError, match="No lens transforms"):
        network.transform(UnregisteredSource, UnregisteredView)


##
