"""Test runner for all data utilities tests.

Runs all test suites for datasets, loaders, transforms, and samplers.
This serves as the main entry point for running data utility tests.
"""

from tests.projectodyssey.data.datasets.test_base_dataset import (
    test_dataset_has_len_method,
    test_dataset_has_getitem_method,
    test_dataset_getitem_returns_tuple,
    test_dataset_getitem_index_validation,
    test_dataset_supports_negative_indexing,
    test_dataset_length_immutable,
    test_dataset_iteration_consistency,
)

from tests.projectodyssey.data.datasets.test_tensor_dataset import (
    test_tensor_dataset_creation,
    test_tensor_dataset_with_matching_sizes,
    test_tensor_dataset_size_mismatch_error,
    test_tensor_dataset_empty,
    test_tensor_dataset_getitem,
)

from tests.projectodyssey.data.loaders.test_base_loader import (
    test_loader_has_len_method,
    test_loader_batch_size_consistency,
    test_loader_empty_dataset,
    test_loader_single_sample,
    test_loader_batch_size_validation,
    test_loader_drop_last_option,
)

from tests.projectodyssey.data.transforms.test_pipeline import (
    test_pipeline_creation,
    test_pipeline_empty,
    test_pipeline_single_transform,
    test_pipeline_sequential_application,
    test_pipeline_output_feeds_next,
    test_pipeline_preserves_intermediate_values,
)

from tests.projectodyssey.data.transforms.test_augmentations import (
    test_random_augmentation_deterministic,
    test_random_augmentation_varies,
    test_random_rotation_range,
    test_random_rotation_no_change,
    test_random_rotation_fill_value,
    test_random_crop_varies_location,
    test_random_crop_with_padding,
)

from tests.projectodyssey.data.transforms.test_augmentations import (
    test_random_horizontal_flip_probability,
    test_random_flip_always,
    test_random_flip_never,
    test_random_erasing_basic,
    test_random_erasing_scale,
    test_compose_random_augmentations,
    test_augmentation_determinism_in_pipeline,
)

from tests.projectodyssey.data.samplers.test_sequential import (
    test_sequential_sampler_creation,
    test_sequential_sampler_empty,
    test_sequential_sampler_yields_all_indices,
    test_sequential_sampler_order,
    test_sequential_sampler_deterministic,
)


def main() raises:
    """Run all data utilities tests.

    Executes all test suites and reports results.
    """
    var total_tests = 0
    var passed_tests = 0
    var failed_tests = 0

    print("=" * 70)
    print("Running Data Utilities Test Suite")
    print("=" * 70)

    # Dataset Tests
    print("\n[1/5] Running Dataset Tests...")
    print("-" * 70)

    try:
        test_dataset_has_len_method()
        passed_tests += 1
        print("  ✓ test_dataset_has_len_method")
    except e:
        failed_tests += 1
        print("  ✗ test_dataset_has_len_method:", e)
    total_tests += 1

    try:
        test_dataset_has_getitem_method()
        passed_tests += 1
        print("  ✓ test_dataset_has_getitem_method")
    except e:
        failed_tests += 1
        print("  ✗ test_dataset_has_getitem_method:", e)
    total_tests += 1

    try:
        test_dataset_getitem_returns_tuple()
        passed_tests += 1
        print("  ✓ test_dataset_getitem_returns_tuple")
    except e:
        failed_tests += 1
        print("  ✗ test_dataset_getitem_returns_tuple:", e)
    total_tests += 1

    try:
        test_dataset_getitem_index_validation()
        passed_tests += 1
        print("  ✓ test_dataset_getitem_index_validation")
    except e:
        failed_tests += 1
        print("  ✗ test_dataset_getitem_index_validation:", e)
    total_tests += 1

    try:
        test_dataset_supports_negative_indexing()
        passed_tests += 1
        print("  ✓ test_dataset_supports_negative_indexing")
    except e:
        failed_tests += 1
        print("  ✗ test_dataset_supports_negative_indexing:", e)
    total_tests += 1

    try:
        test_dataset_length_immutable()
        passed_tests += 1
        print("  ✓ test_dataset_length_immutable")
    except e:
        failed_tests += 1
        print("  ✗ test_dataset_length_immutable:", e)
    total_tests += 1

    try:
        test_dataset_iteration_consistency()
        passed_tests += 1
        print("  ✓ test_dataset_iteration_consistency")
    except e:
        failed_tests += 1
        print("  ✗ test_dataset_iteration_consistency:", e)
    total_tests += 1

    try:
        test_tensor_dataset_creation()
        passed_tests += 1
        print("  ✓ test_tensor_dataset_creation")
    except e:
        failed_tests += 1
        print("  ✗ test_tensor_dataset_creation:", e)
    total_tests += 1

    try:
        test_tensor_dataset_with_matching_sizes()
        passed_tests += 1
        print("  ✓ test_tensor_dataset_with_matching_sizes")
    except e:
        failed_tests += 1
        print("  ✗ test_tensor_dataset_with_matching_sizes:", e)
    total_tests += 1

    try:
        test_tensor_dataset_size_mismatch_error()
        passed_tests += 1
        print("  ✓ test_tensor_dataset_size_mismatch_error")
    except e:
        failed_tests += 1
        print("  ✗ test_tensor_dataset_size_mismatch_error:", e)
    total_tests += 1

    try:
        test_tensor_dataset_empty()
        passed_tests += 1
        print("  ✓ test_tensor_dataset_empty")
    except e:
        failed_tests += 1
        print("  ✗ test_tensor_dataset_empty:", e)
    total_tests += 1

    try:
        test_tensor_dataset_getitem()
        passed_tests += 1
        print("  ✓ test_tensor_dataset_getitem")
    except e:
        failed_tests += 1
        print("  ✗ test_tensor_dataset_getitem:", e)
    total_tests += 1

    # Loader Tests
    print("\n[2/5] Running Loader Tests...")
    print("-" * 70)

    try:
        test_loader_has_len_method()
        passed_tests += 1
        print("  ✓ test_loader_has_len_method")
    except e:
        failed_tests += 1
        print("  ✗ test_loader_has_len_method:", e)
    total_tests += 1

    try:
        test_loader_batch_size_consistency()
        passed_tests += 1
        print("  ✓ test_loader_batch_size_consistency")
    except e:
        failed_tests += 1
        print("  ✗ test_loader_batch_size_consistency:", e)
    total_tests += 1

    try:
        test_loader_empty_dataset()
        passed_tests += 1
        print("  ✓ test_loader_empty_dataset")
    except e:
        failed_tests += 1
        print("  ✗ test_loader_empty_dataset:", e)
    total_tests += 1

    try:
        test_loader_single_sample()
        passed_tests += 1
        print("  ✓ test_loader_single_sample")
    except e:
        failed_tests += 1
        print("  ✗ test_loader_single_sample:", e)
    total_tests += 1

    try:
        test_loader_batch_size_validation()
        passed_tests += 1
        print("  ✓ test_loader_batch_size_validation")
    except e:
        failed_tests += 1
        print("  ✗ test_loader_batch_size_validation:", e)
    total_tests += 1

    try:
        test_loader_drop_last_option()
        passed_tests += 1
        print("  ✓ test_loader_drop_last_option")
    except e:
        failed_tests += 1
        print("  ✗ test_loader_drop_last_option:", e)
    total_tests += 1

    # Transform Tests
    print("\n[3/5] Running Transform Tests...")
    print("-" * 70)

    try:
        test_pipeline_creation()
        passed_tests += 1
        print("  ✓ test_pipeline_creation")
    except e:
        failed_tests += 1
        print("  ✗ test_pipeline_creation:", e)
    total_tests += 1

    try:
        test_pipeline_empty()
        passed_tests += 1
        print("  ✓ test_pipeline_empty")
    except e:
        failed_tests += 1
        print("  ✗ test_pipeline_empty:", e)
    total_tests += 1

    try:
        test_pipeline_single_transform()
        passed_tests += 1
        print("  ✓ test_pipeline_single_transform")
    except e:
        failed_tests += 1
        print("  ✗ test_pipeline_single_transform:", e)
    total_tests += 1

    try:
        test_pipeline_sequential_application()
        passed_tests += 1
        print("  ✓ test_pipeline_sequential_application")
    except e:
        failed_tests += 1
        print("  ✗ test_pipeline_sequential_application:", e)
    total_tests += 1

    try:
        test_pipeline_output_feeds_next()
        passed_tests += 1
        print("  ✓ test_pipeline_output_feeds_next")
    except e:
        failed_tests += 1
        print("  ✗ test_pipeline_output_feeds_next:", e)
    total_tests += 1

    try:
        test_pipeline_preserves_intermediate_values()
        passed_tests += 1
        print("  ✓ test_pipeline_preserves_intermediate_values")
    except e:
        failed_tests += 1
        print("  ✗ test_pipeline_preserves_intermediate_values:", e)
    total_tests += 1

    # Augmentation Tests
    print("\n[3b/5] Running Augmentation Tests...")
    print("-" * 70)

    try:
        test_random_augmentation_deterministic()
        passed_tests += 1
        print("  ✓ test_random_augmentation_deterministic")
    except e:
        failed_tests += 1
        print("  ✗ test_random_augmentation_deterministic:", e)
    total_tests += 1

    try:
        test_random_augmentation_varies()
        passed_tests += 1
        print("  ✓ test_random_augmentation_varies")
    except e:
        failed_tests += 1
        print("  ✗ test_random_augmentation_varies:", e)
    total_tests += 1

    try:
        test_random_rotation_range()
        passed_tests += 1
        print("  ✓ test_random_rotation_range")
    except e:
        failed_tests += 1
        print("  ✗ test_random_rotation_range:", e)
    total_tests += 1

    try:
        test_random_rotation_no_change()
        passed_tests += 1
        print("  ✓ test_random_rotation_no_change")
    except e:
        failed_tests += 1
        print("  ✗ test_random_rotation_no_change:", e)
    total_tests += 1

    try:
        test_random_rotation_fill_value()
        passed_tests += 1
        print("  ✓ test_random_rotation_fill_value")
    except e:
        failed_tests += 1
        print("  ✗ test_random_rotation_fill_value:", e)
    total_tests += 1

    try:
        test_random_crop_varies_location()
        passed_tests += 1
        print("  ✓ test_random_crop_varies_location")
    except e:
        failed_tests += 1
        print("  ✗ test_random_crop_varies_location:", e)
    total_tests += 1

    try:
        test_random_crop_with_padding()
        passed_tests += 1
        print("  ✓ test_random_crop_with_padding")
    except e:
        failed_tests += 1
        print("  ✗ test_random_crop_with_padding:", e)
    total_tests += 1

    try:
        test_random_horizontal_flip_probability()
        passed_tests += 1
        print("  ✓ test_random_horizontal_flip_probability")
    except e:
        failed_tests += 1
        print("  ✗ test_random_horizontal_flip_probability:", e)
    total_tests += 1

    try:
        test_random_flip_always()
        passed_tests += 1
        print("  ✓ test_random_flip_always")
    except e:
        failed_tests += 1
        print("  ✗ test_random_flip_always:", e)
    total_tests += 1

    try:
        test_random_flip_never()
        passed_tests += 1
        print("  ✓ test_random_flip_never")
    except e:
        failed_tests += 1
        print("  ✗ test_random_flip_never:", e)
    total_tests += 1

    try:
        test_random_erasing_basic()
        passed_tests += 1
        print("  ✓ test_random_erasing_basic")
    except e:
        failed_tests += 1
        print("  ✗ test_random_erasing_basic:", e)
    total_tests += 1

    try:
        test_random_erasing_scale()
        passed_tests += 1
        print("  ✓ test_random_erasing_scale")
    except e:
        failed_tests += 1
        print("  ✗ test_random_erasing_scale:", e)
    total_tests += 1

    try:
        test_compose_random_augmentations()
        passed_tests += 1
        print("  ✓ test_compose_random_augmentations")
    except e:
        failed_tests += 1
        print("  ✗ test_compose_random_augmentations:", e)
    total_tests += 1

    try:
        test_augmentation_determinism_in_pipeline()
        passed_tests += 1
        print("  ✓ test_augmentation_determinism_in_pipeline")
    except e:
        failed_tests += 1
        print("  ✗ test_augmentation_determinism_in_pipeline:", e)
    total_tests += 1

    # Sampler Tests
    print("\n[4/5] Running Sampler Tests...")
    print("-" * 70)

    try:
        test_sequential_sampler_creation()
        passed_tests += 1
        print("  ✓ test_sequential_sampler_creation")
    except e:
        failed_tests += 1
        print("  ✗ test_sequential_sampler_creation:", e)
    total_tests += 1

    try:
        test_sequential_sampler_empty()
        passed_tests += 1
        print("  ✓ test_sequential_sampler_empty")
    except e:
        failed_tests += 1
        print("  ✗ test_sequential_sampler_empty:", e)
    total_tests += 1

    try:
        test_sequential_sampler_yields_all_indices()
        passed_tests += 1
        print("  ✓ test_sequential_sampler_yields_all_indices")
    except e:
        failed_tests += 1
        print("  ✗ test_sequential_sampler_yields_all_indices:", e)
    total_tests += 1

    try:
        test_sequential_sampler_order()
        passed_tests += 1
        print("  ✓ test_sequential_sampler_order")
    except e:
        failed_tests += 1
        print("  ✗ test_sequential_sampler_order:", e)
    total_tests += 1

    try:
        test_sequential_sampler_deterministic()
        passed_tests += 1
        print("  ✓ test_sequential_sampler_deterministic")
    except e:
        failed_tests += 1
        print("  ✗ test_sequential_sampler_deterministic:", e)
    total_tests += 1

    # Print Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print("Total Tests:  ", total_tests)
    print("Passed:       ", passed_tests, "✓")
    print("Failed:       ", failed_tests, "✗")
    print("Success Rate: ", (passed_tests * 100) // total_tests, "%")
    print("=" * 70)

    if failed_tests > 0:
        raise Error("Some tests failed")

    print("\n✓ All tests passed!")
