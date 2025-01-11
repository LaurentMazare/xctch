from executorch.devtools import Inspector

etrecord_path = "etrecord.bin"
etdump_path = "foo.etbin"
inspector = Inspector(etdump_path=etdump_path, etrecord=etrecord_path)

for event_block in inspector.event_blocks:
    df = event_block.to_dataframe()
    print(df.iloc[:2].T)
print(df.shape)
df = df[df["event_name"] != "OPERATOR_CALL"]
print(df["event_name"].value_counts())

print(df["avg"].sum())
print(">>>>")
_df = df[df["event_name"] == "Method::execute"]
print(_df.T)

_df = df[["event_name", "avg"]]
_df = (
    _df.groupby('event_name')
    .agg(
        avg=('avg', 'sum'),
        rows=('avg', 'count')
    )
    .sort_values(by='avg', ascending=False)
    .reset_index()
)

print(_df)
