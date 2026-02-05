"""Integration tests for lens transformation chains.

Tests complex lens workflows including:
- Chained transformations (A → B → C)
- Batch lens application
- Optional field handling
- Bidirectional round-trips (lens laws)
- LensNetwork discovery
- NDArray transformations
"""

import pytest

import numpy as np
from numpy.typing import NDArray
import webdataset as wds

import atdata
from atdata.lens import LensNetwork


##
# Test sample types for lens chains


@atdata.packable
class FullRecord:
    """Complete record with all fields."""

    id: int
    name: str
    email: str
    age: int
    score: float
    embedding: NDArray


@atdata.packable
class ProfileView:
    """View with profile information only."""

    name: str
    email: str
    age: int


@atdata.packable
class NameView:
    """Minimal view with just name."""

    name: str


@atdata.packable
class ScoredRecord:
    """Record with score and embedding."""

    id: int
    score: float
    embedding: NDArray


@atdata.packable
class OptionalFieldSample:
    """Sample with optional fields."""

    name: str
    value: int
    extra: str | None = None
    embedding: NDArray | None = None


@atdata.packable
class OptionalView:
    """View of optional sample."""

    name: str
    extra: str | None = None


##
# Lens definitions


@atdata.lens
def full_to_profile(full: FullRecord) -> ProfileView:
    """Extract profile from full record."""
    return ProfileView(
        name=full.name,
        email=full.email,
        age=full.age,
    )


@full_to_profile.putter
def full_to_profile_put(view: ProfileView, source: FullRecord) -> FullRecord:
    """Update full record from profile view."""
    return FullRecord(
        id=source.id,
        name=view.name,
        email=view.email,
        age=view.age,
        score=source.score,
        embedding=source.embedding,
    )


@atdata.lens
def profile_to_name(profile: ProfileView) -> NameView:
    """Extract just name from profile."""
    return NameView(name=profile.name)


@profile_to_name.putter
def profile_to_name_put(view: NameView, source: ProfileView) -> ProfileView:
    """Update profile from name view."""
    return ProfileView(
        name=view.name,
        email=source.email,
        age=source.age,
    )


@atdata.lens
def full_to_scored(full: FullRecord) -> ScoredRecord:
    """Extract scoring data from full record."""
    return ScoredRecord(
        id=full.id,
        score=full.score,
        embedding=full.embedding,
    )


@full_to_scored.putter
def full_to_scored_put(view: ScoredRecord, source: FullRecord) -> FullRecord:
    """Update full record from scored view."""
    return FullRecord(
        id=view.id,
        name=source.name,
        email=source.email,
        age=source.age,
        score=view.score,
        embedding=view.embedding,
    )


@atdata.lens
def optional_to_view(opt: OptionalFieldSample) -> OptionalView:
    """Extract optional fields view."""
    return OptionalView(
        name=opt.name,
        extra=opt.extra,
    )


@optional_to_view.putter
def optional_to_view_put(
    view: OptionalView, source: OptionalFieldSample
) -> OptionalFieldSample:
    """Update optional sample from view."""
    return OptionalFieldSample(
        name=view.name,
        value=source.value,
        extra=view.extra,
        embedding=source.embedding,
    )


##
# Helper functions


def create_full_records(n: int) -> list[FullRecord]:
    """Create n full records with distinct values."""
    return [
        FullRecord(
            id=i,
            name=f"user_{i}",
            email=f"user_{i}@example.com",
            age=20 + (i % 50),
            score=float(i) * 0.1,
            embedding=np.random.randn(64).astype(np.float32),
        )
        for i in range(n)
    ]


def write_dataset(path, samples) -> str:
    """Write samples to tar file, return path."""
    tar_path = path.as_posix()
    with wds.writer.TarWriter(tar_path) as sink:
        for sample in samples:
            sink.write(sample.as_wds)
    return tar_path


##
# Chained Transformation Tests


class TestChainedTransformations:
    """Tests for chaining multiple lens transformations."""

    def test_manual_chain_two_lenses(self, tmp_path):
        """Manually chain two lenses: Full → Profile → Name."""
        n_samples = 20
        records = create_full_records(n_samples)

        tar_path = write_dataset(tmp_path / "full.tar", records)

        # Load as ProfileView first
        profile_ds = atdata.Dataset[FullRecord](tar_path).as_type(ProfileView)

        # Iterate and transform to NameView
        for i, profile in enumerate(profile_ds.ordered(batch_size=None)):
            assert isinstance(profile, ProfileView)

            # Apply second transformation manually
            name_view = profile_to_name(profile)
            assert isinstance(name_view, NameView)
            assert name_view.name == f"user_{i}"

            if i >= 10:
                break

    def test_chain_round_trip(self):
        """Chain of transformations should support round-trip."""
        original = FullRecord(
            id=1,
            name="Alice",
            email="alice@test.com",
            age=30,
            score=0.95,
            embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

        # Forward chain: Full → Profile → Name
        profile = full_to_profile(original)
        name = profile_to_name(profile)

        assert name.name == "Alice"

        # Reverse chain with updates
        new_name = NameView(name="Alice Updated")
        updated_profile = profile_to_name.put(new_name, profile)
        updated_full = full_to_profile.put(updated_profile, original)

        assert updated_full.name == "Alice Updated"
        assert updated_full.id == 1  # Preserved
        assert updated_full.score == 0.95  # Preserved

    def test_parallel_views_from_same_source(self, tmp_path):
        """Same source can have multiple views through different lenses."""
        n_samples = 15
        records = create_full_records(n_samples)

        tar_path = write_dataset(tmp_path / "multi.tar", records)

        # Create two different views of the same data
        profile_ds = atdata.Dataset[FullRecord](tar_path).as_type(ProfileView)
        scored_ds = atdata.Dataset[FullRecord](tar_path).as_type(ScoredRecord)

        profiles = list(profile_ds.ordered(batch_size=None))
        scored = list(scored_ds.ordered(batch_size=None))

        assert len(profiles) == n_samples
        assert len(scored) == n_samples

        for i in range(n_samples):
            # Both views from same source
            assert profiles[i].name == f"user_{i}"
            assert scored[i].id == i
            assert scored[i].score == float(i) * 0.1


class TestBatchLensApplication:
    """Tests for applying lenses to batched data."""

    def test_lens_with_batched_iteration(self, tmp_path):
        """Lens should apply correctly to batched dataset iteration."""
        n_samples = 32
        batch_size = 8
        records = create_full_records(n_samples)

        tar_path = write_dataset(tmp_path / "batch.tar", records)
        dataset = atdata.Dataset[FullRecord](tar_path).as_type(ProfileView)

        batches = list(dataset.ordered(batch_size=batch_size))

        for batch in batches:
            assert isinstance(batch, atdata.SampleBatch)
            assert batch.sample_type == ProfileView

            # Verify samples are ProfileView instances
            for sample in batch.samples:
                assert isinstance(sample, ProfileView)
                assert hasattr(sample, "name")
                assert hasattr(sample, "email")
                assert hasattr(sample, "age")
                # Should not have FullRecord fields
                assert not hasattr(sample, "id")
                assert not hasattr(sample, "score")

    def test_batch_aggregation_after_lens(self, tmp_path):
        """Batch aggregation should work on lens-transformed samples."""
        n_samples = 24
        batch_size = 6
        records = create_full_records(n_samples)

        tar_path = write_dataset(tmp_path / "agg.tar", records)
        dataset = atdata.Dataset[FullRecord](tar_path).as_type(ProfileView)

        batch_idx = 0
        for batch in dataset.ordered(batch_size=batch_size):
            # Access aggregated attributes
            names = batch.name
            emails = batch.email
            ages = batch.age

            assert isinstance(names, list)
            assert isinstance(emails, list)
            assert isinstance(ages, list)
            assert len(names) == len(batch.samples)

            # Verify values
            for i, name in enumerate(names):
                expected_idx = batch_idx * batch_size + i
                assert name == f"user_{expected_idx}"

            batch_idx += 1

    def test_ndarray_lens_with_batching(self, tmp_path):
        """Lens transforming NDArray fields should work with batching."""
        n_samples = 20
        batch_size = 5
        records = create_full_records(n_samples)

        tar_path = write_dataset(tmp_path / "ndarray.tar", records)
        dataset = atdata.Dataset[FullRecord](tar_path).as_type(ScoredRecord)

        for batch in dataset.ordered(batch_size=batch_size):
            # NDArray should be stacked
            embeddings = batch.embedding
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (batch_size, 64)


class TestLensLaws:
    """Tests for lens law compliance (well-behavedness)."""

    def test_getput_law(self):
        """GetPut law: put(get(s), s) == s."""
        source = FullRecord(
            id=42,
            name="Test",
            email="test@example.com",
            age=25,
            score=0.5,
            embedding=np.array([1.0, 2.0], dtype=np.float32),
        )

        view = full_to_profile(source)
        result = full_to_profile.put(view, source)

        assert result.id == source.id
        assert result.name == source.name
        assert result.email == source.email
        assert result.age == source.age
        assert result.score == source.score
        np.testing.assert_array_equal(result.embedding, source.embedding)

    def test_putget_law(self):
        """PutGet law: get(put(v, s)) == v."""
        source = FullRecord(
            id=42,
            name="Original",
            email="original@example.com",
            age=25,
            score=0.5,
            embedding=np.array([1.0, 2.0], dtype=np.float32),
        )

        new_view = ProfileView(
            name="Updated",
            email="updated@example.com",
            age=30,
        )

        updated = full_to_profile.put(new_view, source)
        retrieved = full_to_profile(updated)

        assert retrieved.name == new_view.name
        assert retrieved.email == new_view.email
        assert retrieved.age == new_view.age

    def test_putput_law(self):
        """PutPut law: put(v2, put(v1, s)) == put(v2, s)."""
        source = FullRecord(
            id=42,
            name="Original",
            email="original@example.com",
            age=25,
            score=0.5,
            embedding=np.array([1.0, 2.0], dtype=np.float32),
        )

        view1 = ProfileView(name="First", email="first@example.com", age=26)
        view2 = ProfileView(name="Second", email="second@example.com", age=27)

        # Two ways to get final result
        result1 = full_to_profile.put(view2, full_to_profile.put(view1, source))
        result2 = full_to_profile.put(view2, source)

        assert result1.name == result2.name
        assert result1.email == result2.email
        assert result1.age == result2.age


class TestOptionalFieldLenses:
    """Tests for lenses handling optional fields."""

    def test_optional_field_with_value(self, tmp_path):
        """Lens should handle optional fields that have values."""
        samples = [
            OptionalFieldSample(
                name=f"item_{i}",
                value=i * 10,
                extra=f"extra_{i}",
                embedding=np.random.randn(32).astype(np.float32),
            )
            for i in range(10)
        ]

        tar_path = write_dataset(tmp_path / "opt_filled.tar", samples)
        dataset = atdata.Dataset[OptionalFieldSample](tar_path).as_type(OptionalView)

        for i, view in enumerate(dataset.ordered(batch_size=None)):
            assert isinstance(view, OptionalView)
            assert view.name == f"item_{i}"
            assert view.extra == f"extra_{i}"

    def test_optional_field_with_none(self, tmp_path):
        """Lens should handle optional fields that are None."""
        samples = [
            OptionalFieldSample(
                name=f"item_{i}",
                value=i * 10,
                extra=None,
                embedding=None,
            )
            for i in range(10)
        ]

        tar_path = write_dataset(tmp_path / "opt_none.tar", samples)
        dataset = atdata.Dataset[OptionalFieldSample](tar_path).as_type(OptionalView)

        for view in dataset.ordered(batch_size=None):
            assert isinstance(view, OptionalView)
            assert view.extra is None

    def test_optional_field_lens_roundtrip(self):
        """Lens with optional fields should support round-trip."""
        source = OptionalFieldSample(
            name="test",
            value=42,
            extra="optional",
            embedding=np.array([1.0, 2.0], dtype=np.float32),
        )

        view = optional_to_view(source)
        assert view.extra == "optional"

        # Update with None
        new_view = OptionalView(name="updated", extra=None)
        updated = optional_to_view.put(new_view, source)

        assert updated.name == "updated"
        assert updated.extra is None
        assert updated.value == 42  # Preserved


class TestLensNetworkDiscovery:
    """Tests for LensNetwork registry and discovery."""

    def test_registered_lens_discoverable(self):
        """Registered lenses should be discoverable via LensNetwork."""
        network = LensNetwork()

        # The lenses defined above should be registered
        lens = network.transform(FullRecord, ProfileView)
        assert lens is not None
        assert lens.source_type == FullRecord
        assert lens.view_type == ProfileView

    def test_unregistered_lens_raises(self):
        """Querying unregistered lens should raise ValueError."""

        @atdata.packable
        class UnknownSource:
            x: int

        @atdata.packable
        class UnknownView:
            y: int

        network = LensNetwork()

        with pytest.raises(ValueError, match="No lens transforms"):
            network.transform(UnknownSource, UnknownView)

    def test_multiple_lenses_registered(self):
        """Multiple lenses can be registered and retrieved independently."""
        network = LensNetwork()

        # All our test lenses should be registered
        lens1 = network.transform(FullRecord, ProfileView)
        lens2 = network.transform(ProfileView, NameView)
        lens3 = network.transform(FullRecord, ScoredRecord)

        assert lens1 is not lens2
        assert lens2 is not lens3
        assert lens1.view_type == ProfileView
        assert lens2.view_type == NameView
        assert lens3.view_type == ScoredRecord


class TestNDArrayTransformations:
    """Tests for lenses that transform NDArray fields."""

    def test_ndarray_field_preserved(self, tmp_path):
        """NDArray fields should be correctly preserved through lens."""
        records = create_full_records(10)

        tar_path = write_dataset(tmp_path / "ndarray.tar", records)
        dataset = atdata.Dataset[FullRecord](tar_path).as_type(ScoredRecord)

        for i, scored in enumerate(dataset.ordered(batch_size=None)):
            assert isinstance(scored.embedding, np.ndarray)
            assert scored.embedding.shape == (64,)
            assert scored.embedding.dtype == np.float32
            np.testing.assert_array_almost_equal(
                scored.embedding,
                records[i].embedding,
            )

    def test_ndarray_transformation_lens(self):
        """Lens that transforms NDArray values."""

        @atdata.packable
        class RawData:
            values: NDArray

        @atdata.packable
        class NormalizedData:
            normalized: NDArray

        @atdata.lens
        def normalize(raw: RawData) -> NormalizedData:
            arr = raw.values
            normalized = (arr - arr.mean()) / (arr.std() + 1e-8)
            return NormalizedData(normalized=normalized)

        source = RawData(values=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
        view = normalize(source)

        assert isinstance(view.normalized, np.ndarray)
        # Normalized should have mean ~0 and std ~1
        assert abs(view.normalized.mean()) < 0.01
        assert abs(view.normalized.std() - 1.0) < 0.01


class TestComplexLensScenarios:
    """Complex integration scenarios combining multiple features."""

    def test_dataset_lens_chain_with_batching(self, tmp_path):
        """Full pipeline: Dataset → Lens → Batch iteration."""
        n_samples = 50
        batch_size = 10
        records = create_full_records(n_samples)

        tar_path = write_dataset(tmp_path / "complex.tar", records)

        # Create lens-transformed dataset
        dataset = atdata.Dataset[FullRecord](tar_path).as_type(ProfileView)

        total_samples = 0
        for batch in dataset.ordered(batch_size=batch_size):
            assert isinstance(batch, atdata.SampleBatch)
            assert batch.sample_type == ProfileView

            # Apply second lens to each sample
            for sample in batch.samples:
                name_view = profile_to_name(sample)
                assert isinstance(name_view, NameView)
                assert name_view.name.startswith("user_")

            total_samples += len(batch.samples)

        assert total_samples == n_samples

    def test_shuffled_iteration_with_lens(self, tmp_path):
        """Lens should work with shuffled iteration."""
        n_samples = 30
        records = create_full_records(n_samples)

        tar_path = write_dataset(tmp_path / "shuffle.tar", records)
        dataset = atdata.Dataset[FullRecord](tar_path).as_type(ProfileView)

        seen_names = set()
        for profile in dataset.shuffled(batch_size=None):
            assert isinstance(profile, ProfileView)
            seen_names.add(profile.name)
            if len(seen_names) >= 20:
                break

        # Should have seen multiple distinct names
        assert len(seen_names) >= 10

    def test_lens_preserves_all_fields(self, tmp_path):
        """Lens transformation should preserve all view fields exactly."""
        records = [
            FullRecord(
                id=i,
                name=f"name_{i}",
                email=f"email_{i}@test.com",
                age=20 + i,
                score=float(i) * 0.5,
                embedding=np.full(64, float(i), dtype=np.float32),
            )
            for i in range(10)
        ]

        tar_path = write_dataset(tmp_path / "preserve.tar", records)
        dataset = atdata.Dataset[FullRecord](tar_path).as_type(ScoredRecord)

        for i, scored in enumerate(dataset.ordered(batch_size=None)):
            assert scored.id == i
            assert scored.score == float(i) * 0.5
            np.testing.assert_array_equal(
                scored.embedding,
                np.full(64, float(i), dtype=np.float32),
            )
