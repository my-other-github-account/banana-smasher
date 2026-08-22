import inspect


def test_qtip3_public_batch_entrypoint_has_complete_runtime_import_closure():
    from banana_smasher import qtip25_native_v4_api, qtip3_api_producer

    assert qtip3_api_producer.build_qtip_native_cell is qtip25_native_v4_api.build_qtip_native_cell
    assert callable(qtip25_native_v4_api.build_qtip_native_cells)
    assert inspect.signature(qtip3_api_producer.run_cells_batched).parameters["batch_size"].default == 40
