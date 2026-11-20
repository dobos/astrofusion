import pyarrow as pa
from datafusion import PythonTableProvider

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