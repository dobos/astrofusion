import pyarrow as pa
from datafusion import PythonTableProvider

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