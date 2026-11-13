import unittest

import numpy as np
import pyarrow as pa

from datafusion import PythonTableProvider, SessionContext


class MetadataTableProvider(PythonTableProvider):
    def schema(self):
        return pa.schema([("id", pa.int64()), ("value", pa.float64())])

    def rows(self):
        return [{"id": 1, "value": 1.5}, {"id": 2, "value": 2.5}]

    def constraints(self):
        return {"primary_key": [0]}

    def table_type(self):
        return "Base"

    def get_table_definition(self):
        return "CREATE TABLE metadata_table (id BIGINT, value DOUBLE)"

    def statistics(self):
        return {"num_rows": 2, "total_byte_size": 16}


class NumpyTableProvider(PythonTableProvider):
    def __init__(self, records):
        self.records = records
        self._table = pa.Table.from_arrays(
            [
                pa.array(records["id"]),
                pa.array(records["value"]),
            ],
            names=["id", "value"],
        )
        self.lookup_calls = 0
        self.range_calls = 0

    def schema(self):
        return self._table.schema

    def rows(self):
        return self._table.to_pylist()

    def index_lookup(self, keys):
        self.lookup_calls += 1
        key_set = set(keys)
        return [row for row in self.rows() if row["id"] in key_set]

    def index_range_scan(self, start=None, end=None, column=None):
        self.range_calls += 1
        column = column or "id"
        rows = self.rows()
        return [
            row for row in rows
            if (start is None or row[column] >= start)
            and (end is None or row[column] <= end)
        ]

    def scan(self, projection=None, filters=None, limit=None):
        rows = self.rows()
        filters = filters or []

        start = None
        end = None
        lookup_keys = []

        for expr in filters:
            expr_text = expr.strip()
            if "=" in expr_text and "id" in expr_text and not any(op in expr_text for op in (">", "<")):
                literal = expr_text.split("=", 1)[1].strip()
                literal = literal.removeprefix("Int64(").removesuffix(")")
                lookup_keys.append(int(literal))
                continue
            if ">=" in expr_text and "id" in expr_text:
                start = int(expr_text.split(">=", 1)[1].strip().removeprefix("Int64(").removesuffix(")"))
                continue
            if "<=" in expr_text and "id" in expr_text:
                end = int(expr_text.split("<=", 1)[1].strip().removeprefix("Int64(").removesuffix(")"))

        if lookup_keys:
            rows = self.index_lookup(lookup_keys)
        elif start is not None or end is not None:
            rows = self.index_range_scan(start=start, end=end, column="id")

        if projection:
            field_names = [self.schema().field(i).name for i in projection]
            rows = [{name: row[name] for name in field_names} for row in rows]

        if limit is not None:
            rows = rows[:limit]

        return rows

    def supports_filters_pushdown(self, filters):
        return ["Exact" if "id" in expr else "Unsupported" for expr in filters]


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
