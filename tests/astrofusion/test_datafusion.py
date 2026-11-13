import unittest

import numpy as np
import pyarrow as pa

from datafusion import PythonTableProvider, SessionContext


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

    def schema(self):
        return self._table.schema

    def rows(self):
        return self._table.to_pylist()

    def scan(self, projection=None, filters=None, limit=None):
        rows = self.rows()

        def normalize_literal(value):
            cleaned = value.strip()
            for prefix in ("Float64(", "Int64(", "Utf8(", "Boolean("):
                if cleaned.startswith(prefix) and cleaned.endswith(")"):
                    cleaned = cleaned[len(prefix) : -1]
                    if cleaned.startswith('"') and cleaned.endswith('"'):
                        cleaned = cleaned[1:-1]
                    break
            return cleaned

        if filters:
            filtered = []
            for row in rows:
                keep = True
                for expr in filters:
                    if "value" in expr and ">=" in expr:
                        threshold = float(normalize_literal(expr.split(">=")[1]))
                        keep = row["value"] >= threshold
                    elif "id" in expr and "=" in expr:
                        target = int(normalize_literal(expr.split("=")[1]))
                        keep = row["id"] == target
                    if not keep:
                        break
                if keep:
                    filtered.append(row)
            rows = filtered

        if projection:
            field_names = [self.schema().field(i).name for i in projection]
            rows = [{name: row[name] for name in field_names} for row in rows]

        if limit is not None:
            rows = rows[:limit]

        return rows

    def supports_filters_pushdown(self, filters):
        return ["Exact" if "value" in expr else "Unsupported" for expr in filters]


class TestDataFusionBridge(unittest.TestCase):
    def test_numpy_records_are_queryable(self):
        records = np.array(
            [(1, 1.5), (2, 2.5), (3, 3.5), (4, 4.5)],
            dtype=[("id", "<i8"), ("value", "<f8")],
        )

        ctx = SessionContext()
        ctx.register_table("numpy_records", NumpyTableProvider(records))

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


if __name__ == "__main__":
    unittest.main()
