import unittest
import numpy as np
from datafusion import SessionContext
from table_providers.metadata_provider import MetadataTableProvider
from table_providers.numpy_provider import NumpyTableProvider

class TestDataFusionBridge(unittest.TestCase):
    def test_python_metadata_hooks_are_exposed(self):
        provider = MetadataTableProvider()
        self.assertEqual(provider.constraints(), {"primary_key": [0]})
        self.assertEqual(provider.table_type(), "Base")
        self.assertEqual(
            provider.get_table_definition(),
            "CREATE TABLE metadata_table (id BIGINT, value DOUBLE)",
        )
        self.assertEqual(provider.statistics(), {"num_rows": 2, "total_byte_size": 16})

    def test_numpy_records_are_queryable(self):
        records = np.array(
            [(1, 1.5), (2, 2.5), (3, 3.5), (4, 4.5)],
            dtype=[("id", "<i8"), ("value", "<f8")],
        )

        provider = NumpyTableProvider(records)
        ctx = SessionContext()
        ctx.register_table("numpy_records", provider)

        result = ctx.sql(
            "SELECT id, value FROM numpy_records WHERE value >= 2.5 ORDER BY id"
        ).collect()

        self.assertEqual(
            result,
            [
                {"id": 2, "value": 2.5},
                {"id": 3, "value": 3.5},
                {"id": 4, "value": 4.5},
            ],
        )

    def test_python_index_hooks_are_exposed_and_used(self):
        records = np.array(
            [(1, 1.5), (2, 2.5), (3, 3.5), (4, 4.5)],
            dtype=[("id", "<i8"), ("value", "<f8")],
        )

        provider = NumpyTableProvider(records)
        ctx = SessionContext()
        ctx.register_table("indexed_records", provider)

        exact = ctx.sql("SELECT id, value FROM indexed_records WHERE id = 2").collect()
        self.assertEqual(exact, [{"id": 2, "value": 2.5}])
        self.assertEqual(provider.lookup_calls, 1)

        ranged = ctx.sql("SELECT id, value FROM indexed_records WHERE id >= 2 AND id <= 3").collect()
        self.assertEqual(
            ranged,
            [
                {"id": 2, "value": 2.5},
                {"id": 3, "value": 3.5},
            ],
        )
        self.assertGreaterEqual(provider.range_calls, 1)


if __name__ == "__main__":
    unittest.main()